#!/usr/bin/env python3
"""
AIMO3: Prepare SFT Dataset v4 — ALL SOURCES
=============================================
Filters nvidia/Nemotron-Math-v2 for maximum training volume:

  - Splits: high_part00, high_part01, high_part02 (ALL high-reasoning trajectories)
  - Source: ALL (aops + stackflow) — hard math is hard regardless of source
  - Difficulty: reason_high_with_tool accuracy < 0.5
    (problems the model fails even at max reasoning + tools)
  - Trajectories: with-tool only (matches competition inference)
  - Per problem: shortest trajectory (less noise)

Changes from v3:
  - Removed AoPS-only filter → includes StackExchange hard problems too
  - Added high_part02 → more StackExchange hard problems
  - Reports data_source breakdown in stats

Saves locally to ./data/nemotron-sft-v4/ and uploads to Kaggle.

Usage:
    python3.12 prepare_dataset_v4.py
    python3.12 prepare_dataset_v4.py --splits high_part00        # single split test
    python3.12 prepare_dataset_v4.py --no-upload                 # skip Kaggle upload
    python3.12 prepare_dataset_v4.py --output-dir ./data/my-run

Requires .env: HF_TOKEN, KAGGLE_USERNAME, KAGGLE_API_TOKEN
"""

import os, time, sys, shutil, gc, json, re, argparse
from collections import Counter
from dotenv import load_dotenv
load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--splits", nargs="+",
                    default=["high_part00", "high_part01", "high_part02"],
                    help="Splits to process (default: all 3 high splits)")
parser.add_argument("--no-upload", action="store_true",
                    help="Skip Kaggle upload")
parser.add_argument("--output-dir", default="./data/nemotron-sft-v4",
                    help="Local output directory")
parser.add_argument("--dataset-name", default="nemotron-math-v2-sft-v4",
                    help="Kaggle dataset name for upload")
args = parser.parse_args()

# ============================================================
# CONFIG
# ============================================================
HF_DATASET      = "nvidia/Nemotron-Math-v2"
OUTPUT_DIR      = args.output_dir
ACC_KEY         = "reason_high_with_tool"   # Competition: HIGH reasoning + tools
MAX_ACC         = 0.5                        # Hard problems only (majority vote failing)
SEED            = 42

HF_TOKEN        = os.environ.get("HF_TOKEN")
KAGGLE_TOKEN    = os.environ.get("KAGGLE_API_TOKEN")
KAGGLE_USER     = os.environ.get("KAGGLE_USERNAME")

if not HF_TOKEN:
    print("ERROR: Set HF_TOKEN in .env")
    sys.exit(1)

# ============================================================
# HELPERS
# ============================================================

def disk_free_gb():
    st = os.statvfs('/')
    return (st.f_bavail * st.f_frsize) / 1e9

def clear_hf_cache():
    for d in ["datasets", "hub"]:
        p = os.path.expanduser(f"~/.cache/huggingface/{d}")
        if os.path.exists(p):
            shutil.rmtree(p)
    gc.collect()

def has_tools(example):
    t = example.get("tools")
    return t is not None and (not isinstance(t, list) or len(t) > 0)

def get_acc_high_with_tool(example):
    """Extract reason_high_with_tool accuracy. Returns None if not hard."""
    meta = example.get("metadata") or {}
    entry = meta.get(ACC_KEY, {})
    if not isinstance(entry, dict):
        return None
    acc = entry.get("accuracy")
    if acc is not None and 0 < acc < MAX_ACC:
        return acc
    return None

def is_hard(example):
    return get_acc_high_with_tool(example) is not None

def compress_messages(messages):
    """Collapse multi-turn tool conversation into 2-turn (user, assistant).
    Handles HuggingFace Arrow-backed arrays, numpy strings, and plain dicts.
    """
    if hasattr(messages, "tolist"):
        messages = messages.tolist()
    if not isinstance(messages, list) or len(messages) == 0:
        return None

    user_content = None
    assistant_parts = []

    for m in messages:
        if hasattr(m, "as_py"):
            m = m.as_py()
        if not isinstance(m, dict):
            try:
                m = dict(m)
            except Exception:
                continue

        role    = str(m.get("role") or "").strip()
        content = str(m.get("content") or "").strip()

        if not role:
            continue

        if role == "user" and user_content is None:
            user_content = content
        elif role == "assistant":
            if content:
                assistant_parts.append(content)
        elif role == "tool":
            if content:
                assistant_parts.append(f"\nOutput:\n```\n{content}\n```\n")
        elif role == "user" and user_content is not None:
            if content:
                assistant_parts.append(f"\n[Context: {content}]\n")

    if not user_content or not assistant_parts:
        return None

    return {"user": user_content, "assistant": "\n".join(assistant_parts)}


# ============================================================
# MAIN PROCESSING
# ============================================================
print("=" * 70)
print("  AIMO3 Dataset Builder v4 — ALL SOURCES")
print(f"  Splits:  {args.splits}")
print(f"  Source:  ALL (aops + stackflow)")
print(f"  Filter:  {ACC_KEY} < {MAX_ACC}")
print(f"  Output:  {OUTPUT_DIR}")
print(f"  Disk:    {disk_free_gb():.1f}GB free")
print("=" * 70)

from datasets import load_dataset
import pandas as pd

os.makedirs(OUTPUT_DIR, exist_ok=True)
chunk_paths = []

for split_name in args.splits:
    print(f"\n{'─' * 60}")
    print(f"  [{split_name}] | Free disk: {disk_free_gb():.1f}GB")
    print(f"{'─' * 60}")

    print("  Downloading from HuggingFace...")
    t0 = time.time()
    ds = load_dataset(HF_DATASET, token=HF_TOKEN, split=split_name)
    print(f"  Loaded {len(ds):,} rows in {time.time()-t0:.0f}s")

    # Filter 1: with-tool only
    before = len(ds)
    ds = ds.filter(has_tools, num_proc=4)
    print(f"  Tool filter:  {before:,} → {len(ds):,}")

    # Filter 2: hard (reason_high_with_tool < 0.5)
    before = len(ds)
    ds = ds.filter(is_hard, num_proc=4)
    print(f"  Hard filter:  {before:,} → {len(ds):,}")

    if len(ds) == 0:
        print("  No rows — skipping")
        del ds; gc.collect()
        clear_hf_cache()
        continue

    print("  Converting to pandas...")
    df = ds.to_pandas()
    del ds; gc.collect()

    # Normalize string columns (Arrow strings → Python str)
    for col in ["uuid", "problem", "expected_answer", "data_source"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: str(x) if x is not None else "")

    # Extract accuracy
    df["_acc"] = df.apply(
        lambda row: get_acc_high_with_tool(row.to_dict()), axis=1
    )

    # Compress multi-turn → 2-turn
    print("  Compressing multi-turn → 2-turn...")
    df["_compressed"] = df["messages"].apply(compress_messages)
    before = len(df)
    df = df[df["_compressed"].notna()].reset_index(drop=True)
    print(f"  Compression: {before:,} → {len(df):,}")

    df["_n_words"] = df["_compressed"].apply(
        lambda x: len(x["assistant"].split()) if x else 0
    )

    # Stats
    print(f"\n  Data sources ({split_name}):")
    for src, cnt in df["data_source"].value_counts().items():
        print(f"    {src}: {cnt:,} ({cnt/len(df)*100:.1f}%)")

    print(f"\n  Accuracy distribution ({split_name}):")
    for acc, cnt in sorted(df["_acc"].round(3).value_counts().sort_index().items()):
        print(f"    {acc:.3f}: {cnt:,}")

    # Save chunk
    chunk_path = os.path.join(OUTPUT_DIR, f"{split_name}.pkl")
    df.to_pickle(chunk_path)
    print(f"\n  Saved {chunk_path} ({os.path.getsize(chunk_path)/1e6:.0f}MB, {len(df):,} rows)")
    chunk_paths.append(chunk_path)

    del df; gc.collect()
    clear_hf_cache()


# ============================================================
# COMBINE + DEDUPLICATE (1 best trajectory per problem)
# ============================================================
print(f"\n{'=' * 70}")
print("  Combining + deduplicating")
print(f"{'=' * 70}")

if not chunk_paths:
    print("  No data! All splits filtered to empty.")
    sys.exit(1)

dfs = [pd.read_pickle(p) for p in chunk_paths]
for p, d in zip(chunk_paths, dfs):
    print(f"  {os.path.basename(p)}: {len(d):,} rows")
combined = pd.concat(dfs, ignore_index=True)

# Use problem text as dedup key — uuid has Arrow type issues with hashing
combined["_problem_key"] = combined["problem"].apply(
    lambda x: str(x).strip()[:500]
)
print(f"\n  Combined: {len(combined):,} rows | Unique problems: {combined['_problem_key'].nunique():,}")

# Deduplicate: keep shortest trajectory per problem (less noise)
combined = combined.sort_values("_n_words", ascending=True)
best = combined.drop_duplicates(subset="_problem_key", keep="first").copy()
best = best.drop(columns=["_problem_key"])
print(f"  Deduped: {len(combined):,} → {len(best):,} (1 per problem)")


# ============================================================
# BUILD FINAL DATAFRAME
# ============================================================
print(f"\n{'=' * 70}")
print("  Building final dataset")
print(f"{'=' * 70}")

records = []
for _, row in best.iterrows():
    comp = row["_compressed"]
    records.append({
        "uuid": row["uuid"],
        "messages": [
            {"role": "user",      "content": comp["user"]},
            {"role": "assistant", "content": comp["assistant"]},
        ],
        "expected_answer": str(row.get("expected_answer", "")),
        "data_source":     row.get("data_source", ""),
        "accuracy":        float(row["_acc"]),
        "problem":         str(row.get("problem", "")),
    })

final_df = pd.DataFrame(records).sample(frac=1, random_state=SEED).reset_index(drop=True)

# Stats
print(f"\n  Total examples: {len(final_df):,}")

print(f"\n  Data source breakdown:")
for src, cnt in final_df["data_source"].value_counts().items():
    print(f"    {src}: {cnt:,} ({cnt/len(final_df)*100:.1f}%)")

print(f"\n  Accuracy breakdown:")
for acc, cnt in final_df["accuracy"].round(3).value_counts().sort_index().items():
    print(f"    {acc:.3f}: {cnt:,}")

wc = final_df["messages"].apply(lambda m: len(m[1]["content"].split()))
print(f"\n  Word counts:")
print(f"    p10={int(wc.quantile(.1))} p50={int(wc.median())} p90={int(wc.quantile(.9))} max={int(wc.max())}")

code_pct  = final_df["messages"].apply(lambda m: "```" in m[1]["content"]).mean() * 100
boxed_pct = final_df["messages"].apply(lambda m: "\\boxed" in m[1]["content"]).mean() * 100
print(f"  Code blocks: {code_pct:.1f}% | \\boxed: {boxed_pct:.1f}%")


# ============================================================
# SAVE LOCALLY
# ============================================================
print(f"\n{'=' * 70}")
print(f"  Saving locally → {OUTPUT_DIR}")
print(f"{'=' * 70}")

# Parquet
final_parquet = os.path.join(OUTPUT_DIR, "data.parquet")
parquet_df = final_df.copy()
parquet_df["messages"] = parquet_df["messages"].apply(json.dumps)
parquet_df.to_parquet(final_parquet, index=False)
print(f"  data.parquet: {os.path.getsize(final_parquet)/1e6:.0f}MB")

# HF dataset
from datasets import Dataset
final_ds = Dataset.from_pandas(final_df)
hf_path = os.path.join(OUTPUT_DIR, "hf_dataset")
final_ds.save_to_disk(hf_path)
print(f"  hf_dataset/: saved")

# Clean temp chunk files
for p in chunk_paths:
    if os.path.exists(p):
        os.remove(p)

# README
src_table = ""
for src, cnt in final_df["data_source"].value_counts().items():
    src_table += f"| {src} | {cnt:,} | {cnt/len(final_df)*100:.1f}% |\n"

acc_table = ""
for acc in sorted(final_df["accuracy"].unique()):
    cnt = (final_df["accuracy"] == acc).sum()
    acc_table += f"| {acc:.3f} | {cnt:,} |\n"

readme = f"""# Nemotron-Math-v2 SFT v4 — All Sources, Hard Problems

## What This Is
SFT dataset for AIMO3 fine-tuning. All hard math problems from `nvidia/Nemotron-Math-v2` 
where the model fails at high reasoning with tools.

## Changes from v3
- ✅ Includes **all data sources** (AoPS + StackExchange) — hard is hard regardless of source
- ✅ Includes **high_part02** — more StackExchange hard problems
- ✅ Much larger dataset for better training signal

## Filters Applied
| Filter | Value | Rationale |
|--------|-------|-----------|
| Splits | `high_part00`, `high_part01`, `high_part02` | All high reasoning depth trajectories |
| Source | ALL (aops + stackflow) | Hard problems from any source improve reasoning |
| Accuracy key | `reason_high_with_tool` | Matches competition: HIGH reasoning + tools |
| Difficulty | acc < {MAX_ACC} | Problems where majority vote fails |
| Trajectories | with-tool only | Matches competition inference |
| Dedup | shortest per problem | Less noise |

## Stats

### By Data Source
| Source | Count | Percentage |
|--------|-------|-----------|
{src_table}

### By Accuracy
| Accuracy | Count |
|----------|-------|
{acc_table}
**Total**: {len(final_df):,} examples

| Metric | Value |
|--------|-------|
| Code blocks | {code_pct:.1f}% |
| \\boxed answer | {boxed_pct:.1f}% |
| Median words | {int(wc.median())} |
| p90 words | {int(wc.quantile(.9))} |

## Columns
| Column | Description |
|--------|-------------|
| `uuid` | Problem ID |
| `messages` | 2-turn: [user, assistant] |
| `expected_answer` | Verified answer |
| `data_source` | `aops` or `stackflow` |
| `accuracy` | high_with_tool accuracy (0.125/0.25/0.375) |
| `problem` | Raw problem statement |

## Usage
```python
import pandas as pd, json
df = pd.read_parquet("data.parquet")
df["messages"] = df["messages"].apply(json.loads)

# Curriculum: hardest first
stage1 = df[df["accuracy"] == 0.125]  # Model almost never solves
stage2 = df[df["accuracy"] == 0.25]
stage3 = df[df["accuracy"] == 0.375]

# AoPS only subset
aops = df[df["data_source"] == "aops"]
```

## Source
`nvidia/Nemotron-Math-v2` (CC BY 4.0 / CC BY-SA 4.0)
"""

readme_path = os.path.join(OUTPUT_DIR, "README.md")
with open(readme_path, "w") as f:
    f.write(readme)
print(f"  README.md: saved")


# ============================================================
# UPLOAD TO KAGGLE
# ============================================================
if not args.no_upload:
    print(f"\n{'=' * 70}")
    print(f"  Uploading to Kaggle: {KAGGLE_USER}/{args.dataset_name}")
    print(f"{'=' * 70}")

    if not (KAGGLE_TOKEN and KAGGLE_USER):
        print("  ⚠️  KAGGLE_API_TOKEN or KAGGLE_USERNAME not set — skipping upload")
    else:
        try:
            import kagglehub
            handle = f"{KAGGLE_USER}/{args.dataset_name}"

            upload_dir = os.path.join(OUTPUT_DIR, "_upload")
            os.makedirs(upload_dir, exist_ok=True)
            shutil.copy(final_parquet, os.path.join(upload_dir, "data.parquet"))
            shutil.copy(readme_path,   os.path.join(upload_dir, "README.md"))

            print(f"  Uploading {len(final_df):,} rows...")
            t0 = time.time()
            kagglehub.dataset_upload(
                handle, upload_dir,
                version_notes=(
                    f"v4: all sources, reason_high_with_tool<{MAX_ACC}, "
                    f"all high splits, {len(final_df):,} rows"
                ),
            )
            print(f"  ✅ Uploaded in {(time.time()-t0)/60:.1f} mins")
            print(f"  → https://www.kaggle.com/datasets/{handle}")
            shutil.rmtree(upload_dir)
        except Exception as e:
            print(f"  ⚠️  Upload failed: {e}")
else:
    print("\n  Skipping Kaggle upload (--no-upload)")


# ============================================================
# SAMPLES
# ============================================================
print(f"\n{'=' * 70}")
print("  Samples")
print(f"{'=' * 70}")

sample = final_df.sample(min(3, len(final_df)), random_state=SEED)
for i, (_, row) in enumerate(sample.iterrows()):
    asst = row["messages"][1]["content"]
    print(f"\n  ── Sample {i+1} (acc={row['accuracy']:.3f}, src={row['data_source']}) ──")
    print(f"  Answer: {row['expected_answer'][:80]}")
    print(f"  Q: {row['messages'][0]['content'][:120]}...")
    print(f"  A: {asst[:200]}...")
    print(f"  Words: {len(asst.split())} | Code: {'```' in asst} | Boxed: {'boxed' in asst}")


# ============================================================
# DONE
# ============================================================
print(f"\n{'=' * 70}")
print(f"  DONE ✅")
print(f"{'=' * 70}")
print(f"  Local:   {OUTPUT_DIR}/")
print(f"  Parquet: {final_parquet}")
print(f"  Rows:    {len(final_df):,}")
print(f"  Disk:    {disk_free_gb():.1f}GB free")
print(f"{'=' * 70}")
