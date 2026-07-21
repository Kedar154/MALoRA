# unified_loader.py — CLI Reference

Streams agentic trajectories from a Hugging Face dataset, converts them to
Hermes tool-calling format, builds ramp/slide training samples, tokenizes
them, filters anything that doesn't fit your context window, and saves the
result — either as a fixed pull of raw entries or as an exact count of
approved training samples.

## Setup

```bash
export HF_TOKEN=hf_xxx....
```

Token can also be passed via `--hf_token`, or left unset entirely if you're
already logged in via `huggingface-cli login`.

## Two modes (pick exactly one)

### Mode A — `--num_samples`: pull N raw entries

Pulls `N` raw entries from the dataset, converts and samples all of them,
tokenizes with a hard drop for anything over `--max_length`, and prints a
full token-length distribution report (mean/median/percentiles + suggested
context window sizes) plus a max-fit calibration block.

Use this first, before you know what `--max_length` should be — it's your
diagnostic pass.

```bash
python unified_loader.py --num_samples 200
```

### Mode B — `--target_samples`: stream until exactly N approved samples

Streams entries indefinitely (not capped by entry count) until exactly `N`
training samples have passed both the char-length prefilter and the real
token-length check. Saves with the actual count and context window baked
into the output name, e.g. `tokenized_ds_10000samples_ctx8192`.

Use this once you've picked a `--max_length` from Mode A's report and want
a clean, uniformly-sized training set.

```bash
python unified_loader.py --target_samples 10000 --max_length 8192
```

`--num_samples` and `--target_samples` are mutually exclusive — the script
will error if you pass both or neither.

## All arguments

| Flag | Default | Mode | Description |
|---|---|---|---|
| `--dataset_name` | `nvidia/Nemotron-Cascade-2-SFT-Data` | both | HF dataset to stream from |
| `--dataset_config` | `swe` | both | Dataset config name |
| `--split` | `train` | both | Dataset split |
| `--num_samples` | — | A | Number of raw entries to pull |
| `--target_samples` | — | B | Exact number of approved training samples to collect |
| `--hf_token` | — | both | Falls back to `HF_TOKEN` env var, then local HF cache |
| `--max_context_turns` | `4` | both | How many previous turns each sample's context window reaches back — this is turns, not sample count. Every target turn gets exactly one sample regardless of this value |
| `--max_length` | `8192` | both | Token budget per sample. Mode A: drop-if-exceeds. Mode B: required, used for both the char prefilter and the real token check |
| `--tokenizer_name` | `Qwen/Qwen2.5-coder-3B-Instruct` | both | Tokenizer used for length checks |
| `--tokenized_out` | — | A | Path to save tokenized dataset (Arrow via `save_to_disk`, or `.jsonl` fallback) |
| `--untokenized_out` | — | both | Path to dump readable pre-tokenization samples as JSON, for sanity-checking. In Mode B, only use this for small `--target_samples` — it holds both raw and tokenized samples in memory simultaneously |
| `--tokenized_out_prefix` | `tokenized_ds` | B | Base name for output; actual sample count and context window get appended automatically |
| `--chars_per_token` | `4.15` | B | Crude chars-per-token ratio used for the cheap prefilter gate, before any tokenizer call. Pull this from Mode A's "max-fit sample" report line rather than guessing |
| `--safety_margin` | `0.9` | B | Shrinks the char budget so the prefilter stays conservative — better to occasionally tokenize-and-reject than let an oversized sample slip past the cheap gate |
| `--max_entries_cap` | — | B | Safety valve: stop pulling entries after this many, even if `--target_samples` hasn't been reached yet (otherwise the stream could hang on a dataset too small to satisfy the target) |

## Recommended workflow

1. **Measure the distribution** — run Mode A with no `--max_length` filter to see the true token-length spread and get a suggested context window:
```bash
   python unified_loader.py --num_samples 200
```

2. **Sanity-check a small batch** — dump readable samples to eyeball the Hermes conversion and sample boundaries:
```bash
   python unified_loader.py --num_samples 50 --untokenized_out readable.json
```

3. **Real run** — once you've picked `--max_length` from step 1's report (and noted the `chars_per_token` ratio from its max-fit line), run Mode B for your actual training set:
```bash
   python unified_loader.py --target_samples 10000 --max_length 8192 --chars_per_token 4.15
```

## Notes

- `--max_context_turns` controls context *reach*, not sample *count*. Every
  turn in every pulled entry is supervised exactly once — lowering this
  value does not give you more samples, it only shortens each sample's
  context.
- Mode A **drops** (never truncates) any sample whose full prefix+target
  exceeds `--max_length` — no partial-context sample sneaks into the
  training set.
- Mode B applies a two-gate filter: a cheap character-count check first
  (no tokenizer call), then a real token check only for samples that pass
  the first gate. This keeps large runs fast by avoiding a tokenizer call
  on every single candidate sample.