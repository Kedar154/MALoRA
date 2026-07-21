"""
eval_tool_calls.py

Turn-by-turn tool-call generation eval for the Hermes-style
"<tool_call>{...}</tool_call>" coding-agent format.

For each sample:
  - Walk prompt_messages in order.
  - Every time the next ground-truth message is role="assistant",
    that's a "turn": ask the model to predict it given everything
    seen so far (teacher-forced on ground truth, not on the model's
    own previous guesses), then append the *ground truth* message
    (not the model's) before continuing.
  - After walking all of prompt_messages, do one final turn using
    gt_text as the ground-truth assistant message to predict.

Each prediction is checked for:
  - presence of >=1 well-formed <tool_call>...</tool_call> block(s)
  - each block's inner content parses as valid JSON
  - each parsed object has "name" (str) and "arguments" (dict) keys

Nothing here judges whether the *content* of the tool call is
correct/logical -- only syntax/parse validity, per your instructions.

Outputs:
  - outputs.json: each sample's messages, with a "model_answer" role
    entry inserted immediately after every ground-truth "assistant"
    entry that was used as a prediction point.
  - a summary printed to stdout (parse success rate, json validity rate).
"""

import argparse
import json
import os
import re
import sys
import time
import copy

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── reuse your existing loading setup ─────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configuration_lora_moe import LoraMoeConfig
from modelling import LoraMoeModel
from training_config import TrainingConfig
conf = TrainingConfig()
HF_TOKEN   = conf.HFT
HF_REPO_ID = conf.HF_REPO
BASE_MODEL = conf.MODEL_ID

MAX_NEW_TOKENS = 512

TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


# ── model loading (trimmed straight from your old script) ─────────────────
def load_model(hf_folder: str, attn_on: bool = False):
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    print(f"Loading: {HF_REPO_ID}/{hf_folder}")
    local_dir = snapshot_download(
        repo_id=HF_REPO_ID,
        token=HF_TOKEN,
        allow_patterns=[f"{hf_folder}/*", f"{hf_folder}/model.safetensors"],
        local_dir=f"./hf_cache/{hf_folder.replace('/', '_')}",
    )
    ckpt_path = os.path.join(local_dir, hf_folder, "model.safetensors")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(local_dir, "model.safetensors")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"model.safetensors not found under {local_dir}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    base_model.config.use_cache = True
    base_model.enable_input_require_grads()

    moe_config = LoraMoeConfig.from_pretrained(BASE_MODEL)
    moe_config.experts_rank         = 8
    moe_config.attention_rank       = 32
    moe_config.experts_scale        = 1.0
    moe_config.num_experts_per_tok  = 2
    moe_config.num_local_experts    = 8
    moe_config.output_router_logits = False
    moe_config.router_aux_loss_coef = 0.001
    moe_config.use_attention_lora   = attn_on

    moe_model = LoraMoeModel(base_model, moe_config)

    saved_sd = load_file(ckpt_path, device="cpu")
    model_sd = moe_model.state_dict()

    matched = {k: v for k, v in saved_sd.items() if k in model_sd}
    if len(matched) == 0:
        remapped = {}
        for k, v in saved_sd.items():
            new_k = "base_model." + k
            if new_k in model_sd:
                remapped[new_k] = v
                continue
            if k.startswith("base_model."):
                stripped = k[len("base_model."):]
                if stripped in model_sd:
                    remapped[stripped] = v
                    continue
            remapped[k] = v
        matched = {k: v for k, v in remapped.items() if k in model_sd}

    moe_model.load_state_dict(matched, strict=False)
    moe_model.eval()
    print("Model ready.\n")
    return moe_model, tokenizer


# ── generation ──────────────────────────────────────────────────────────────
def generate_turn(model, tokenizer, messages_so_far):
    formatted = tokenizer.apply_chat_template(
        messages_so_far, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    completion = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
    return completion


# ── syntax/parse validation only (no logic checking) ───────────────────────
def validate_tool_call_text(text):
    """
    Returns a dict:
      {
        "has_tool_call": bool,          # found >=1 <tool_call> block
        "num_blocks": int,
        "all_json_valid": bool,         # every block parses as JSON
        "all_schema_valid": bool,       # every parsed obj has name+arguments
        "parsed_calls": [ {...} ... ],  # successfully parsed objects
        "errors": [ "..." ]             # parse/schema error messages
      }
    """
    result = {
        "has_tool_call": False,
        "num_blocks": 0,
        "all_json_valid": True,
        "all_schema_valid": True,
        "parsed_calls": [],
        "errors": [],
    }

    blocks = TOOL_CALL_RE.findall(text)
    result["num_blocks"] = len(blocks)
    result["has_tool_call"] = len(blocks) > 0

    if not blocks:
        result["all_json_valid"] = False
        result["all_schema_valid"] = False
        result["errors"].append("no <tool_call>...</tool_call> block found")
        return result

    for i, raw in enumerate(blocks):
        raw_stripped = raw.strip()
        try:
            obj = json.loads(raw_stripped)
        except json.JSONDecodeError as e:
            result["all_json_valid"] = False
            result["all_schema_valid"] = False
            result["errors"].append(f"block {i}: invalid JSON ({e})")
            continue

        if not isinstance(obj, dict) or "name" not in obj or "arguments" not in obj:
            result["all_schema_valid"] = False
            result["errors"].append(f"block {i}: missing 'name'/'arguments' keys")
            continue

        if not isinstance(obj["name"], str) or not isinstance(obj["arguments"], dict):
            result["all_schema_valid"] = False
            result["errors"].append(f"block {i}: 'name' must be str, 'arguments' must be dict")
            continue

        result["parsed_calls"].append(obj)

    return result


# ── core eval loop ──────────────────────────────────────────────────────────
def eval_sample(model, tokenizer, sample):
    """
    Walks sample["prompt_messages"] turn by turn, then a final turn for
    gt_text. Returns:
      - output_messages: list of dicts for outputs.json (with model_answer
        entries interleaved after each ground-truth assistant turn)
      - turn_reports: list of per-turn validation dicts
    """
    prompt_messages = sample["prompt_messages"]
    gt_text = sample["gt_text"]

    output_messages = []
    context = []          # messages actually fed to the model at each step
    turn_reports = []
    turn_idx = 0

    def do_prediction_and_record(gt_message_role, gt_message_content):
        nonlocal turn_idx
        turn_idx += 1
        t0 = time.time()
        prediction_text = generate_turn(model, tokenizer, context)
        elapsed = time.time() - t0
        validation = validate_tool_call_text(prediction_text)

        turn_reports.append({
            "sample_id": sample.get("id"),
            "turn_index": turn_idx,
            "gen_seconds": round(elapsed, 2),
            **validation,
        })

        # ground truth turn goes in first
        output_messages.append({"role": gt_message_role, "content": gt_message_content})
        # model's answer goes right after it
        output_messages.append({"role": "model_answer", "content": prediction_text})

    for msg in prompt_messages:
        role = msg["role"]
        content = msg["content"]

        if role == "assistant":
            # this is a ground-truth tool-call turn -> predict it first
            do_prediction_and_record("assistant", content)
            # advance context with the *ground truth* (teacher forcing)
            context.append({"role": "assistant", "content": content})
        else:
            # system / user / tool -> just context, no prediction
            output_messages.append({"role": role, "content": content})
            context.append({"role": role, "content": content})

    # final turn: predict gt_text
    do_prediction_and_record("assistant", gt_text)

    return output_messages, turn_reports


def load_dataset(path):
    samples = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="path to the jsonl dataset")
    parser.add_argument("--hf-folder", required=True, help="checkpoint subfolder in the HF repo")
    parser.add_argument("--attn-on", action="store_true")
    parser.add_argument("--out", default="outputs.json")
    parser.add_argument("--limit", type=int, default=None, help="only eval first N samples")
    args = parser.parse_args()

    samples = load_dataset(args.dataset)
    if args.limit:
        samples = samples[: args.limit]

    model, tokenizer = load_model(args.hf_folder, attn_on=args.attn_on)

    all_outputs = []
    all_turn_reports = []

    for i, sample in enumerate(samples):
        print(f"[{i+1}/{len(samples)}] evaluating sample {sample.get('id')} ...")
        output_messages, turn_reports = eval_sample(model, tokenizer, sample)
        all_outputs.append({
            "id": sample.get("id"),
            "case_type": sample.get("case_type"),
            "messages": output_messages,
        })
        all_turn_reports.extend(turn_reports)

    with open(args.out, "w") as f:
        json.dump(all_outputs, f, indent=2)

    # ── summary stats ────────────────────────────────────────────────────
    total = len(all_turn_reports)
    has_call = sum(r["has_tool_call"] for r in all_turn_reports)
    json_valid = sum(r["all_json_valid"] for r in all_turn_reports)
    schema_valid = sum(r["all_schema_valid"] for r in all_turn_reports)

    print("\n" + "=" * 50)
    print(f"Total turns evaluated: {total}")
    print(f"Turns with >=1 tool_call block: {has_call} ({has_call/total:.1%})")
    print(f"Turns with all blocks valid JSON: {json_valid} ({json_valid/total:.1%})")
    print(f"Turns with all blocks schema-valid (name+arguments): {schema_valid} ({schema_valid/total:.1%})")
    print("=" * 50)

    # dump per-turn reports too, for your own logic-level review later
    with open("turn_reports.json", "w") as f:
        json.dump(all_turn_reports, f, indent=2)

    print(f"\nWrote {args.out} and turn_reports.json")


if __name__ == "__main__":
    main()