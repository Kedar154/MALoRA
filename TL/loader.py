import re
import json
import copy
import argparse
import statistics
import itertools
import os

from transformers import AutoTokenizer
from datasets import load_dataset

TOKENIZER_NAME = "Qwen/Qwen2.5-coder-3B-Instruct"  # swap for your actual base model

# =====================================================================
# PART 1 — Hermes conversion (unchanged core logic)
# =====================================================================

FUNC_BLOCK_RE = re.compile(r"<function>(.*?)</function>", re.DOTALL)
FIELD_RE = {
    "name": re.compile(r"<name>(.*?)</name>", re.DOTALL),
    "description": re.compile(r"<description>(.*?)</description>", re.DOTALL),
}
PARAM_BLOCK_RE = re.compile(r"<parameter>(.*?)</parameter>", re.DOTALL)
PARAM_FIELD_RE = {
    "name": re.compile(r"<name>(.*?)</name>", re.DOTALL),
    "type": re.compile(r"<type>(.*?)</type>", re.DOTALL),
    "description": re.compile(r"<description>(.*?)</description>", re.DOTALL),
    "enum": re.compile(r"<enum>(.*?)</enum>", re.DOTALL),
}
REQUIRED_RE = re.compile(r"<required>(.*?)</required>", re.DOTALL)


def parse_openhands_tools(system_prompt: str):
    tools = []
    for func_match in FUNC_BLOCK_RE.finditer(system_prompt):
        block = func_match.group(1)

        name = FIELD_RE["name"].search(block)
        desc = FIELD_RE["description"].search(block)
        name = name.group(1).strip() if name else "unnamed_tool"
        desc = desc.group(1).strip() if desc else ""

        properties = {}
        required = []
        for pmatch in PARAM_BLOCK_RE.finditer(block):
            pblock = pmatch.group(1)
            pname = PARAM_FIELD_RE["name"].search(pblock)
            ptype = PARAM_FIELD_RE["type"].search(pblock)
            pdesc = PARAM_FIELD_RE["description"].search(pblock)
            penum = PARAM_FIELD_RE["enum"].search(pblock)

            if not pname:
                continue
            pname = pname.group(1).strip()
            ptype = ptype.group(1).strip() if ptype else "string"
            pdesc = pdesc.group(1).strip() if pdesc else ""

            prop = {"type": ptype, "description": pdesc}
            if penum:
                enum_vals = re.findall(r"\[(.*?)\]", penum.group(1), re.DOTALL)
                if enum_vals:
                    prop["enum"] = [
                        v.strip().strip('"').strip("'")
                        for v in enum_vals[0].split(",")
                    ]
            properties[pname] = prop

            if re.search(rf"[Rr]equired parameter.*?\b{re.escape(pname)}\b", block):
                required.append(pname)

        req_block = REQUIRED_RE.search(block)
        if req_block:
            extra = [x.strip().strip('"').strip("'") for x in req_block.group(1).split(",")]
            required = list(dict.fromkeys(required + [e for e in extra if e]))

        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        })
    return tools


HERMES_TOOL_INSTRUCTIONS = """You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools:
<tools>
{tools_json}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-dict>}}
</tool_call>
"""


def build_hermes_system_prompt(original_system_prompt: str, tools: list) -> str:
    prose = re.sub(r"<tools>.*?</tools>", "", original_system_prompt, flags=re.DOTALL).strip()
    prose = re.sub(
        r"If you choose to call a function.*$",
        "",
        prose,
        flags=re.DOTALL,
    ).strip()

    tools_json = json.dumps(tools, indent=2)
    hermes_block = HERMES_TOOL_INSTRUCTIONS.format(tools_json=tools_json)

    return f"{prose}\n\n{hermes_block}"


FUNCTION_CALL_RE = re.compile(
    r"<function=(?P<name>[^>]+)>(?P<body>.*?)</function>", re.DOTALL
)
PARAMETER_RE = re.compile(
    r"<parameter=(?P<key>[^>]+)>\n?(?P<value>.*?)\n?</parameter>", re.DOTALL
)


def _coerce_value(raw: str):
    raw = raw.strip()
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


def convert_tool_call_block(text: str) -> str:
    def _replace_outer(match):
        inner = match.group(1)
        func_match = FUNCTION_CALL_RE.search(inner)
        if not func_match:
            return match.group(0)

        name = func_match.group("name").strip()
        body = func_match.group("body")

        args = {}
        for pmatch in PARAMETER_RE.finditer(body):
            key = pmatch.group("key").strip()
            val = _coerce_value(pmatch.group("value"))
            args[key] = val

        hermes_json = json.dumps({"name": name, "arguments": args})
        return f"<tool_call>\n{hermes_json}\n</tool_call>"

    outer_re = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    return outer_re.sub(_replace_outer, text)


def convert_tool_response(content: str) -> str:
    content = re.sub(r"</?tool_response>", "", content).strip()
    return content


def convert_entry_to_hermes(entry: dict) -> dict:
    """Convert a single {"messages": [...]} entry in place (deep-copied)."""
    entry = copy.deepcopy(entry)
    messages = entry["messages"]
    for i in range(len(messages)):
        msg = messages[i]
        if msg["role"] == "system":
            tools = parse_openhands_tools(msg["content"])
            msg["content"] = build_hermes_system_prompt(msg["content"], tools)
        elif msg["role"] == "assistant":
            msg["content"] = convert_tool_call_block(msg["content"])
        elif msg["role"] == "user" and "<tool_response>" in msg["content"]:
            msg["role"] = "tool"
            msg["content"] = convert_tool_response(msg["content"])
    return entry


# =====================================================================
# PART 2 — ramp/slide sample building (unchanged core logic)
# =====================================================================

def group_into_turns(messages):
    turns = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg["role"] == "assistant":
            turn = [msg]
            if i + 1 < len(messages) and messages[i + 1]["role"] == "tool":
                turn.append(messages[i + 1])
                i += 2
            else:
                i += 1
            turns.append(turn)
        else:
            i += 1
    return turns


def render_message(role, content):
    return f"<|im_start|>{role}\n{content}<|im_end|>\n"


def render_turn(turn):
    return "".join(render_message(m["role"], m["content"]) for m in turn)


def build_ramp_slide_samples(entry, max_context_turns=4):
    messages = entry["messages"]
    system_msg, user_msg = messages[0], messages[1]
    turns = group_into_turns(messages[2:])

    header = render_message(system_msg["role"], system_msg["content"]) + \
             render_message(user_msg["role"], user_msg["content"])

    samples = []
    for t in range(len(turns)):
        context_turns = turns[max(0, t - max_context_turns):t]
        target_assistant_msg = turns[t][0]

        prefix_text = header + "".join(render_turn(ct) for ct in context_turns)
        target_text = render_message(target_assistant_msg["role"], target_assistant_msg["content"])

        samples.append({
            "full_text": prefix_text + target_text,
            "prefix_text": prefix_text,
            "target_text": target_text,
            "target_turn_index": t,
            "num_context_turns": len(context_turns),
            "domain": entry.get("domain"),
            "source": entry.get("source"),
        })
    return samples


# =====================================================================
# PART 3 — streaming pipeline: HF dataset -> hermes -> samples -> tokens
# =====================================================================

def stream_entries_from_hf(dataset_name, dataset_config, split, num_samples, hf_token=None):
    """Stream `num_samples` raw entries directly from a Hugging Face dataset
    (streaming=True, no full download), yielding one entry dict at a time.
    Assumes each HF row already has the shape {"messages": [...], ...} —
    same shape your local nvidia_test.json dump had."""
    ds = load_dataset(
        dataset_name,
        name=dataset_config,
        split=split,
        streaming=True,
        token=hf_token,
    )
    print(f"[hf] connected to {dataset_name} (config={dataset_config}, split={split})")
    print(f"[hf] features: {ds.features}")

    for entry in itertools.islice(ds, num_samples):
        yield entry


def stream_hermes_samples(dataset_name, dataset_config, split, num_samples,
                           hf_token=None, max_context_turns=4):
    """Full streaming chain: HF row -> hermes-converted entry ->
    ramp/slide samples, yielded one sample at a time."""
    n_entries = 0
    for entry in stream_entries_from_hf(dataset_name, dataset_config, split, num_samples, hf_token):
        n_entries += 1
        hermes_entry = convert_entry_to_hermes(entry)
        for sample in build_ramp_slide_samples(hermes_entry, max_context_turns):
            yield sample
    print(f"[hf] pulled {n_entries} entries from stream")


def tokenize_sample(sample, tokenizer, max_length=None):
    prefix_ids = tokenizer(sample["prefix_text"], add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(sample["target_text"], add_special_tokens=False)["input_ids"]

    input_ids = prefix_ids + target_ids
    labels = [-100] * len(prefix_ids) + target_ids[:]
    attention_mask = [1] * len(input_ids)
    raw_length = len(input_ids)  # pre-truncation, used for length analysis
    char_length = len(sample["prefix_text"]) + len(sample["target_text"])  # NEW

    truncated = False
    if max_length is not None and len(input_ids) > max_length:
        overflow = len(input_ids) - max_length
        if overflow >= len(prefix_ids):
            return None, raw_length, char_length  # target alone doesn't fit; drop this sample
        input_ids = input_ids[overflow:]
        labels = labels[overflow:]
        attention_mask = attention_mask[overflow:]
        truncated = True

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "target_turn_index": sample["target_turn_index"],
        "domain": sample["domain"],
        "source": sample["source"],
        "truncated": truncated,
    }, raw_length, char_length


# =====================================================================
# PART 3b — char-prefilter -> tokenize -> approve, single sample path
# =====================================================================

def prefilter_and_build(sample, tokenizer, max_length, chars_per_token=4.15, safety_margin=0.9):
    full_len_chars = len(sample["prefix_text"]) + len(sample["target_text"])
    char_budget = max_length * chars_per_token * safety_margin

    if full_len_chars > char_budget:
        return None, "char_gate"

    prefix_ids = tokenizer(sample["prefix_text"], add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(sample["target_text"], add_special_tokens=False)["input_ids"]
    total_len = len(prefix_ids) + len(target_ids)

    if total_len > max_length:
        return None, "token_gate"

    input_ids = prefix_ids + target_ids
    labels = [-100] * len(prefix_ids) + target_ids[:]
    attention_mask = [1] * len(input_ids)

    row = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "target_turn_index": sample["target_turn_index"],
        "domain": sample["domain"],
        "source": sample["source"],
    }
    return (row, sample), "approved"   # CHANGED: bundle raw sample alongside row

# =====================================================================
# PART 3c — stream until exactly N approved samples
# =====================================================================

def stream_approved_samples_until_target(
    dataset_name, dataset_config, split, target_samples, tokenizer,
    max_length, hf_token=None, max_context_turns=4,
    chars_per_token=4.15, safety_margin=0.9, max_entries_cap=None,
):
    approved_count = 0
    entries_pulled = 0
    stats = {"char_gate": 0, "token_gate": 0, "approved": 0}

    ds = load_dataset(dataset_name, name=dataset_config, split=split,
                       streaming=True, token=hf_token)
    print(f"[hf] connected to {dataset_name} (config={dataset_config}, split={split})")

    for entry in ds:
        if max_entries_cap is not None and entries_pulled >= max_entries_cap:
            print(f"[warn] hit max_entries_cap={max_entries_cap} before reaching "
                  f"target_samples={target_samples}. Only {approved_count} approved samples collected.")
            break

        entries_pulled += 1
        hermes_entry = convert_entry_to_hermes(entry)
        print(f"[hf] pulled entry {entries_pulled} (domain={hermes_entry.get('domain')}, source={hermes_entry.get('source')}\nHermees: {hermes_entry})")
        
        for sample in build_ramp_slide_samples(hermes_entry, max_context_turns):
            result, reason = prefilter_and_build(
                sample, tokenizer, max_length, chars_per_token, safety_margin
            )
            stats[reason] += 1

            if result is not None:
                row, raw_sample = result           # CHANGED: unpack both
                approved_count += 1
                yield row, raw_sample               # CHANGED: yield both
                if approved_count >= target_samples:
                    print(f"[hf] pulled {entries_pulled} entries to reach {approved_count} approved samples")
                    print(f"[stats] char_gate rejects={stats['char_gate']}, "
                          f"token_gate rejects={stats['token_gate']}, approved={stats['approved']}")
                    return

    print(f"[hf] stream exhausted after {entries_pulled} entries — "
          f"only {approved_count}/{target_samples} approved samples collected")
    print(f"[stats] char_gate rejects={stats['char_gate']}, "
          f"token_gate rejects={stats['token_gate']}, approved={stats['approved']}")
    

# =====================================================================
# PART 4 — token length analysis
# =====================================================================

def percentile(sorted_vals, p):
    if not sorted_vals:
        return 0
    k = (len(sorted_vals) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def print_length_report(lengths, dropped_count):
    if not lengths:
        print("No samples tokenized — nothing to report.")
        return

    sorted_lengths = sorted(lengths)
    n = len(sorted_lengths)

    report = {
        "count": n,
        "dropped_target_overflow": dropped_count,
        "min": sorted_lengths[0],
        "max": sorted_lengths[-1],
        "mean": round(statistics.mean(sorted_lengths), 1),
        "median": round(statistics.median(sorted_lengths), 1),
        "stdev": round(statistics.stdev(sorted_lengths), 1) if n > 1 else 0,
        "p50": round(percentile(sorted_lengths, 50), 1),
        "p75": round(percentile(sorted_lengths, 75), 1),
        "p90": round(percentile(sorted_lengths, 90), 1),
        "p95": round(percentile(sorted_lengths, 95), 1),
        "p99": round(percentile(sorted_lengths, 99), 1),
    }

    print("\n===== TOKEN LENGTH ANALYSIS (pre-truncation, full_text) =====")
    for k, v in report.items():
        print(f"{k:>24}: {v}")

    print("\n----- suggested context windows -----")
    for p, label in [(90, "p90"), (95, "p95"), (99, "p99"), (100, "max")]:
        val = report["max"] if p == 100 else report[f"p{p}"]
        for bucket in [512, 1024, 2048, 4096, 8192, 16384, 32768]:
            if val <= bucket:
                print(f"  covering {label:>4} ({val:>7}) -> window {bucket}")
                break
        else:
            print(f"  covering {label:>4} ({val:>7}) -> exceeds 32768, needs larger window")

# =====================================================================
# PART 4b — max-fit-sample tracking (character/token calibration)
# =====================================================================

class MaxFitTracker:
    """Tracks the largest-by-tokens sample that still fits under max_length,
    so you can later filter by character count as a cheap proxy instead of
    tokenizing every sample."""

    def __init__(self, max_length=None):
        self.max_length = max_length
        self.best_tokens = -1
        self.best_chars = None
        self.best_domain = None
        self.best_source = None
        # also track absolute max (ignoring the limit) for contrast
        self.abs_max_tokens = -1
        self.abs_max_chars = None

    def update(self, raw_length, char_length, sample):
        if raw_length > self.abs_max_tokens:
            self.abs_max_tokens = raw_length
            self.abs_max_chars = char_length

        fits = (self.max_length is None) or (raw_length <= self.max_length)
        if fits and raw_length > self.best_tokens:
            self.best_tokens = raw_length
            self.best_chars = char_length
            self.best_domain = sample.get("domain")
            self.best_source = sample.get("source")

    def report(self):
        print("\n----- max-fit sample (chars-per-token calibration) -----")
        if self.best_tokens < 0:
            print("  No sample found that fits under max_length.")
        else:
            ratio = self.best_chars / self.best_tokens if self.best_tokens else 0
            limit_str = self.max_length if self.max_length is not None else "∞ (no limit set)"
            print(f"  max_length limit        : {limit_str}")
            print(f"  largest fitting sample  : {self.best_tokens} tokens / {self.best_chars} chars")
            print(f"  chars-per-token ratio   : {ratio:.3f}")
            print(f"  domain / source         : {self.best_domain} / {self.best_source}")
            print(f"  => as a fast proxy, samples with full_text longer than "
                  f"~{self.best_chars} characters are likely to exceed max_length.")
        if self.max_length is not None and self.abs_max_tokens > self.best_tokens:
            print(f"  (for reference, the true largest sample overall was "
                  f"{self.abs_max_tokens} tokens / {self.abs_max_chars} chars, "
                  f"which did NOT fit under the limit)")

def split_and_save(rows, out_path, eval_ratio=0.05, seed=42):
    """Splits rows into train/eval and saves as a DatasetDict at out_path."""
    from datasets import Dataset, DatasetDict

    ds = Dataset.from_list(rows)
    if eval_ratio and eval_ratio > 0 and len(ds) > 1:
        split = ds.train_test_split(test_size=eval_ratio, seed=seed)
        dsd = DatasetDict({"train": split["train"], "eval": split["test"]})
    else:
        dsd = DatasetDict({"train": ds, "eval": ds.select(range(0))})

    dsd.save_to_disk(out_path)
    print(f"Saved {len(dsd['train'])} train / {len(dsd['eval'])} eval samples -> {out_path} "
          f"(load with datasets.load_from_disk('{out_path}'))")
    return dsd


# =====================================================================
# PART 5 — driver: stream from HF -> tokenize -> save -> analyze
# =====================================================================
def run_pipeline(dataset_name, dataset_config, split, num_samples, hf_token,
                  tokenized_out=None, max_context_turns=4, max_length=None,
                  tokenizer_name=TOKENIZER_NAME, untokenized_out=None, eval_ratio=0.05):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    lengths = []
    dropped_count = 0
    tokenized_rows = []
    untokenized_rows = [] if untokenized_out else None
    max_fit_tracker = MaxFitTracker(max_length=max_length)

    for sample in stream_hermes_samples(
        dataset_name, dataset_config, split, num_samples, hf_token, max_context_turns
    ):
        if untokenized_rows is not None:
            untokenized_rows.append(sample)

        tok_row, raw_len, char_len = tokenize_sample(sample, tokenizer, max_length)
        lengths.append(raw_len)
        max_fit_tracker.update(raw_len, char_len, sample)

        if tok_row is None:
            dropped_count += 1
            continue
        tokenized_rows.append(tok_row)

    print(f"Streamed and tokenized {len(lengths)} samples "
          f"({len(tokenized_rows)} kept, {dropped_count} dropped due to target overflow).")

    if untokenized_out:
        with open(untokenized_out, "w", encoding="utf-8") as f:
            json.dump(untokenized_rows, f, indent=2, ensure_ascii=False)
        print(f"Saved untokenized samples -> {untokenized_out}")

    if tokenized_out:
        try:
            split_and_save(tokenized_rows, tokenized_out, eval_ratio=eval_ratio)
        except ImportError:
            jsonl_path = tokenized_out if tokenized_out.endswith(".jsonl") else tokenized_out + ".jsonl"
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for row in tokenized_rows:
                    f.write(json.dumps(row) + "\n")
            print(f"'datasets' not installed — saved {len(tokenized_rows)} samples as JSONL -> {jsonl_path} "
                  f"(no split applied)")

    print_length_report(lengths, dropped_count)
    max_fit_tracker.report()

def run_target_pipeline(dataset_name, dataset_config, split, target_samples,
                         hf_token, tokenized_out_prefix, max_context_turns=4,
                         max_length=8192, tokenizer_name=TOKENIZER_NAME,
                         chars_per_token=4.15, safety_margin=0.9,
                         max_entries_cap=None, untokenized_out=None, eval_ratio=0.05):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    rows = []
    raw_samples = [] if untokenized_out else None

    for row, raw_sample in stream_approved_samples_until_target(
        dataset_name, dataset_config, split, target_samples, tokenizer,
        max_length, hf_token, max_context_turns,
        chars_per_token, safety_margin, max_entries_cap,
    ):
        rows.append(row)
        if raw_samples is not None:
            raw_samples.append(raw_sample)

    actual_n = len(rows)
    # size-in-filename, e.g. "tokenized_ds" -> "tokenized_ds_10000samples_ctx8192"
    suffix = f"_{actual_n}samples_ctx{max_length}"
    out_path = f"{tokenized_out_prefix}{suffix}"

    if untokenized_out:
        readable_path = f"{untokenized_out}{suffix}.json"
        with open(readable_path, "w", encoding="utf-8") as f:
            json.dump(raw_samples, f, indent=2, ensure_ascii=False)
        print(f"Saved {actual_n} untokenized samples -> {readable_path}")

    try:
        split_and_save(rows, out_path, eval_ratio=eval_ratio)
    except ImportError:
        jsonl_path = out_path + ".jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"'datasets' not installed — saved {actual_n} samples as JSONL -> {jsonl_path} "
              f"(no split applied)")
        
    if actual_n < target_samples:
        print(f"[warn] only reached {actual_n}/{target_samples} — dataset or max_entries_cap ran out first.")

    return out_path, actual_n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="nvidia/Nemotron-Cascade-2-SFT-Data")
    parser.add_argument("--dataset_config", default="swe")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num_samples", type=int, default=None,
                         help="Number of raw entries to pull from the HF stream. "
                              "Required unless --target_samples is used.")
    parser.add_argument("--hf_token", default=None,
                         help="HF token. If omitted, falls back to HF_TOKEN env var, "
                              "then to your local huggingface-cli login cache.")
    parser.add_argument("--untokenized_out", default=None, help="Path to save readable JSON (optional)")
    parser.add_argument("--tokenized_out", default=None, help="Path to save tokenized dataset (optional)")
    parser.add_argument("--max_context_turns", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=8192,
                         help="Training context window in tokens.")

    parser.add_argument("--target_samples", type=int, default=None,
                         help="Stream until exactly this many approved training samples collected. "
                              "Mutually exclusive with --num_samples.")
    parser.add_argument("--tokenized_out_prefix", default="tokenized_ds",
                         help="Base name for output; actual count + ctx window get appended.")
    parser.add_argument("--chars_per_token", type=float, default=4.15,
                         help="From your length report's max-fit ratio; used for the crude char prefilter.")
    parser.add_argument("--safety_margin", type=float, default=0.9,
                         help="Shrinks char budget so the crude gate stays conservative.")
    parser.add_argument("--max_entries_cap", type=int, default=None,
                         help="Safety valve for target_samples mode.")
    parser.add_argument("--tokenizer_name", default=TOKENIZER_NAME)
    parser.add_argument("--eval_ratio", type=float, default=0.05,
                         help="Fraction of samples to reserve for eval split (0-1).")   
    args = parser.parse_args()
    untokenized_out = args.untokenized_out

    if args.target_samples is None and args.num_samples is None:
        parser.error("Provide either --num_samples or --target_samples.")
    if args.target_samples is not None and args.num_samples is not None:
        parser.error("--num_samples and --target_samples are mutually exclusive — use one or the other.")

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    if args.target_samples is not None:
        if args.max_length is None:
            parser.error("--target_samples requires --max_length to be set.")
        if untokenized_out and args.target_samples > 1000:
            print(f"[warn] --untokenized_out with target_samples={args.target_samples} "
                  f"will hold both raw and tokenized samples in memory — fine for small runs, "
                  f"but consider omitting it for large target_samples.")
        run_target_pipeline(
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            split=args.split,
            target_samples=args.target_samples,
            hf_token=hf_token,
            tokenized_out_prefix=args.tokenized_out_prefix,
            max_context_turns=args.max_context_turns,
            max_length=args.max_length,
            tokenizer_name=args.tokenizer_name,
            chars_per_token=args.chars_per_token,
            safety_margin=args.safety_margin,
            max_entries_cap=args.max_entries_cap,
            untokenized_out=untokenized_out,
            eval_ratio=args.eval_ratio
        )
    else:
        run_pipeline(
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            split=args.split,
            num_samples=args.num_samples,
            hf_token=hf_token,
            tokenized_out=args.tokenized_out,
            max_context_turns=args.max_context_turns,
            max_length=args.max_length,
            tokenizer_name=args.tokenizer_name,
            untokenized_out=untokenized_out,
            eval_ratio=args.eval_ratio
        )


if __name__ == "__main__":
    main()


'''
usage:

# put your token in the env instead of hardcoding it in source:
export HF_TOKEN=hf_xxx....

# max context turns set to 2
python3 unified_loader.py --num_samples 200 --max_context_turns 2

# first pass — no max_length, just see the true distribution
python3 unified_loader.py --num_samples 200

# custom dataset/config/split
python3 unified_loader.py --dataset_name nvidia/Nemotron-Cascade-2-SFT-Data --dataset_config swe --split train --num_samples 500

# once you've picked a window from the report, run for real
python3 unified_loader.py --num_samples 2000 --tokenized_out tokenized_ds --max_length 8192

# also dump readable pre-tokenization samples for sanity-checking (num_samples mode)
python3 unified_loader.py --num_samples 200 --untokenized_out readable.json --tokenized_out tokenized_ds --max_length 8192

# target-count mode: stream until exactly 10k approved samples
python3 unified_loader.py --target_samples 10000 --max_length 8192 --max_context_turns 2

# target-count mode with untokenized dump — only use for small target_samples
python3 unified_loader.py --target_samples 200 --max_length 8192 --untokenized_out readable
'''