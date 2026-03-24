"""
AIMO3: Prepare SFT Dataset — HIGH splits only, with AND without tools
=====================================================================
One split at a time → filter acc<0.5 → compress → save → clear cache

NO medium split. High reasoning depth only (better signal for hard problems).
Both with-tool and without-tool trajectories included (tagged).

Usage: python3 prepare_dataset.py
Env vars from .env: HF_TOKEN, KAGGLE_USERNAME, KAGGLE_API_TOKEN
"""

import os, time, sys, shutil, gc
from collections import Counter
from dotenv import load_dotenv
load_dotenv()

# ============================================================
# CONFIG
# ============================================================
HF_DATASET = "nvidia/Nemotron-Math-v2"
OUTPUT_DIR = "./data/nemotron-sft-high-medium-tools"
MAX_ACC = 0.5
SEED = 42

HF_TOKEN = os.environ.get("HF_TOKEN")
KAGGLE_API_TOKEN = os.environ.get("KAGGLE_API_TOKEN")
KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME")

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
            print(f"    Cleared cache: {d}")
    gc.collect()

def has_tools(example):
    tools = example.get('tools')
    return tools is not None and (not isinstance(tools, list) or len(tools) > 0)

def get_acc_high(example, with_tool):
    """Get accuracy from reason_high_with_tool or reason_high_no_tool."""
    meta = example.get('metadata') or {}
    key = 'reason_high_with_tool' if with_tool else 'reason_high_no_tool'
    entry = meta.get(key, {})
    return entry.get('accuracy', 1.0) if isinstance(entry, dict) else 1.0

def compress_messages(messages):
    if hasattr(messages, 'tolist'):
        messages = messages.tolist()
    if not isinstance(messages, list) or len(messages) == 0:
        return None

    user_content = None
    assistant_parts = []

    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get('role', '')
        content = str(m.get('content', ''))

        if role == 'user' and user_content is None:
            user_content = content
        elif role == 'assistant':
            assistant_parts.append(content)
        elif role == 'tool':
            assistant_parts.append(f"\nOutput:\n```\n{content}\n```\n")
        elif role == 'user' and user_content is not None:
            assistant_parts.append(f"\n[Additional context: {content}]\n")

    if user_content is None or not assistant_parts:
        return None

    return {
        "user": user_content,
        "assistant": "\n".join(assistant_parts),
    }

# ============================================================
# PROCESS HIGH SPLITS ONLY
# ============================================================
print("=" * 60)
print("Processing Nemotron-Math-v2 — HIGH splits only")
print(f"Free disk: {disk_free_gb():.1f}GB")
print("=" * 60)

from datasets import load_dataset
import pandas as pd

splits = ["high_part00", "high_part01", "high_part02", "medium"]
chunk_paths = []
total_loaded = 0
acc_counts_all = Counter()

os.makedirs(OUTPUT_DIR, exist_ok=True)

for split_name in splits:
    print(f"\n{'─'*50}")
    print(f"  Split: {split_name} | Free disk: {disk_free_gb():.1f}GB")
    print(f"{'─'*50}")

    print(f"  Downloading...")
    t0 = time.time()
    ds = load_dataset(HF_DATASET, token=HF_TOKEN, split=split_name)
    print(f"  Loaded {len(ds)} rows in {time.time()-t0:.0f}s")
    total_loaded += len(ds)

    # Filter: with-tool only
    before = len(ds)
    ds = ds.filter(has_tools, num_proc=4)
    print(f"  With-tool: {before} → {len(ds)}")

    # Filter: acc < 0.5 (use correct key per split)
    acc_key = 'reason_medium_with_tool' if split_name == 'medium' else 'reason_high_with_tool'
    before = len(ds)
    ds = ds.filter(
        lambda x: 0 < (((x.get('metadata') or {}).get(acc_key) or {}).get('accuracy', 1.0)) < MAX_ACC,
        num_proc=4
    )
    print(f"  Hard (acc<{MAX_ACC}): {before} → {len(ds)}")

    if len(ds) == 0:
        print(f"  No rows — skipping")
        del ds; gc.collect()
        clear_hf_cache()
        continue

    # Convert to pandas
    print(f"  Converting to pandas...")
    df = ds.to_pandas()
    del ds; gc.collect()

    # Add accuracy (use correct key per split)
    def _get_acc(row):
        meta = row.get('metadata') or {}
        key = 'reason_medium_with_tool' if split_name == 'medium' else 'reason_high_with_tool'
        entry = meta.get(key, {})
        return entry.get('accuracy', 1.0) if isinstance(entry, dict) else 1.0
    df['_acc'] = df.apply(lambda row: _get_acc(row.to_dict()), axis=1)
    
    # Tag reasoning depth
    df['_reasoning_depth'] = 'medium' if split_name == 'medium' else 'high'

    # Compress messages
    print(f"  Compressing multi-turn → 2-turn...")
    df['_compressed'] = df['messages'].apply(compress_messages)

    before = len(df)
    df = df[df['_compressed'].notna()].reset_index(drop=True)
    print(f"  Compression: {before} → {len(df)} (dropped {before - len(df)} broken)")

    df['_assistant_words'] = df['_compressed'].apply(
        lambda x: len(x['assistant'].split()) if x else 0
    )

    # Log
    for acc_val in sorted(df['_acc'].unique()):
        n = (df['_acc'] == acc_val).sum()
        acc_counts_all[acc_val] += n
        print(f"    acc={acc_val:.3f}: {n:>7d} rows")

    # Save chunk (pickle avoids pyarrow nested data issues)
    chunk_path = os.path.join(OUTPUT_DIR, f"{split_name}.pkl")
    df.to_pickle(chunk_path)
    chunk_gb = os.path.getsize(chunk_path) / 1e9
    print(f"  Saved: {chunk_path} ({chunk_gb:.2f}GB, {len(df)} rows)")
    chunk_paths.append(chunk_path)

    del df; gc.collect()
    clear_hf_cache()
    print(f"  Free disk: {disk_free_gb():.1f}GB")

# ============================================================
# COMBINE + SELECT BEST PER PROBLEM
# ============================================================
print(f"\n{'='*60}")
print("Combining + selecting best trajectory per problem")
print(f"{'='*60}")

dfs = []
for p in chunk_paths:
    chunk = pd.read_pickle(p)
    print(f"  {os.path.basename(p)}: {len(chunk)} rows")
    dfs.append(chunk)

combined = pd.concat(dfs, ignore_index=True)
print(f"\n  Raw combined: {len(combined)} rows")
print(f"  Unique problems: {combined['uuid'].nunique()}")

# Per problem: prefer HIGH over MEDIUM, then shortest within same depth
# Sort by: reasoning_depth (high first), then word count (shortest first)
print(f"\n  Selecting best trajectory per problem (prefer high > medium, then shortest)...")
depth_priority = {'high': 0, 'medium': 1}
combined['_depth_sort'] = combined['_reasoning_depth'].map(depth_priority).fillna(1)
combined = combined.sort_values(['_depth_sort', '_assistant_words'], ascending=[True, True])
best = combined.drop_duplicates(subset='uuid', keep='first').copy()
best = best.drop(columns=['_depth_sort'])
print(f"  {len(combined)} trajectories → {len(best)} (1 per problem)")

# Stats after dedup
print(f"\n  After selecting best per problem:")
for acc_val in sorted(best['_acc'].unique()):
    subset = best[best['_acc'] == acc_val]
    n = len(subset)
    med_w = subset['_assistant_words'].median()
    print(f"    acc={acc_val:.3f}: {n:>7d} problems (median {med_w:.0f} words)")

# Build final dataset
print(f"\n  Building final dataset...")
records = []
for _, row in best.iterrows():
    comp = row['_compressed']
    records.append({
        'uuid': row['uuid'],
        'messages': [
            {"role": "user", "content": comp['user']},
            {"role": "assistant", "content": comp['assistant']},
        ],
        'expected_answer': row.get('expected_answer', ''),
        'data_source': row.get('data_source', ''),
        'accuracy': row['_acc'],
        'reasoning_depth': row.get('_reasoning_depth', 'high'),
    })

final_df = pd.DataFrame(records)
final_df = final_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
print(f"  Final: {len(final_df)} rows")

# Stats
word_counts = final_df['messages'].apply(lambda m: len(m[1]['content'].split()))
print(f"\n  Assistant word counts:")
print(f"    p10={word_counts.quantile(0.1):.0f} p25={word_counts.quantile(0.25):.0f} "
      f"median={word_counts.median():.0f} p75={word_counts.quantile(0.75):.0f} "
      f"p90={word_counts.quantile(0.9):.0f} max={word_counts.max():.0f}")

print(f"\n  Accuracy breakdown:")
for acc_val in sorted(final_df['accuracy'].unique()):
    n = (final_df['accuracy'] == acc_val).sum()
    n_high = ((final_df['accuracy'] == acc_val) & (final_df['reasoning_depth'] == 'high')).sum()
    n_med = ((final_df['accuracy'] == acc_val) & (final_df['reasoning_depth'] == 'medium')).sum()
    print(f"    acc={acc_val:.3f}: {n:>7d} ({n_high} high, {n_med} medium)")

# ============================================================
# SAVE
# ============================================================
print(f"\n{'='*60}")
print("Saving")
print(f"{'='*60}")

import json as json_lib

# Save as parquet (convert messages to JSON strings for parquet compatibility)
final_parquet = os.path.join(OUTPUT_DIR, "data.parquet")
parquet_df = final_df.copy()
parquet_df['messages'] = parquet_df['messages'].apply(json_lib.dumps)
parquet_df.to_parquet(final_parquet, index=False)
print(f"  Parquet: {final_parquet} ({os.path.getsize(final_parquet)/1e9:.2f}GB)")

# Save as HF dataset (native format, messages stay as list of dicts)
from datasets import Dataset
final_ds = Dataset.from_pandas(final_df)
hf_path = os.path.join(OUTPUT_DIR, "hf_dataset")
final_ds.save_to_disk(hf_path)
print(f"  HF dataset: {hf_path}")

for p in chunk_paths:
    if os.path.exists(p):
        os.remove(p)

# ============================================================
# SAMPLES
# ============================================================
print(f"\n{'='*60}")
print("Samples")
print(f"{'='*60}")

for i in range(min(3, len(final_df))):
    row = final_df.iloc[i]
    asst = row['messages'][1]['content']
    print(f"\n  ── Sample {i+1} (acc={row['accuracy']}) ──")
    print(f"  USER: {row['messages'][0]['content'][:150]}...")
    print(f"  ASST: {asst[:400]}...")
    print(f"  Words: {len(asst.split())} | Code: {'```python' in asst} | Boxed: {'boxed' in asst}")

# ============================================================
# README
# ============================================================
acc_stats = ""
for acc_val in sorted(final_df['accuracy'].unique()):
    n = (final_df['accuracy'] == acc_val).sum()
    n_high = ((final_df['accuracy'] == acc_val) & (final_df['reasoning_depth'] == 'high')).sum()
    n_med = ((final_df['accuracy'] == acc_val) & (final_df['reasoning_depth'] == 'medium')).sum()
    acc_stats += f"| {acc_val:.3f} | {n} | {n_high} | {n_med} |\n"

readme = f"""# Nemotron-Math-v2 SFT — HIGH + MEDIUM Reasoning, With Tools Only

## Selection
- **Splits**: high_part00, high_part01, high_part02, medium
- **With-tool trajectories only**
- **Difficulty**: accuracy strictly < {MAX_ACC}
- **Per problem**: shortest trajectory kept
- **Answers**: ALL types

## Columns
- `uuid`: unique problem ID
- `messages`: 2-turn compressed [user, assistant]
- `expected_answer`: verified answer
- `data_source`: problem source
- `accuracy`: 0.125, 0.25, or 0.375
- `reasoning_depth`: "high" or "medium"

## Stats
| Accuracy | Total | High | Medium |
|----------|-------|------|--------|
{acc_stats}
- **Total**: {len(final_df)} examples
- **Source**: nvidia/Nemotron-Math-v2 (CC BY 4.0)

## Curriculum Training
```python
# H200 curriculum (high only):
stage1 = ds.filter(lambda x: x['accuracy'] == 0.125 and x['reasoning_depth'] == 'high')
stage2 = ds.filter(lambda x: x['accuracy'] == 0.25 and x['reasoning_depth'] == 'high')
stage3 = ds.filter(lambda x: x['accuracy'] == 0.375 and x['reasoning_depth'] == 'high')

# AWS 8xH100 (everything):
all_data = ds  # use all 147K+
```
"""

with open(os.path.join(OUTPUT_DIR, "README.md"), "w") as f:
    f.write(readme)
print(f"\n  README saved")

# ============================================================
# UPLOAD TO KAGGLE
# ============================================================
print(f"\n{'='*60}")
print("Upload to Kaggle")
print(f"{'='*60}")

if KAGGLE_API_TOKEN and KAGGLE_USERNAME:
    try:
        import kagglehub
        handle = f"{KAGGLE_USERNAME}/nemotron-math-v2-sft-high-medium-tools"
        print(f"  Uploading {handle}...")
        t0 = time.time()

        upload_dir = os.path.join(OUTPUT_DIR, "_upload")
        os.makedirs(upload_dir, exist_ok=True)
        shutil.copy(final_parquet, os.path.join(upload_dir, "data.parquet"))
        shutil.copy(os.path.join(OUTPUT_DIR, "README.md"), os.path.join(upload_dir, "README.md"))

        kagglehub.dataset_upload(
            handle, upload_dir,
            version_notes=f"HIGH only, with+without tools, acc<{MAX_ACC}. {len(final_df)} rows.",
        )
        print(f"  ✅ Uploaded in {(time.time()-t0)/60:.1f} mins")
        shutil.rmtree(upload_dir)
    except Exception as e:
        print(f"  ⚠️  Upload failed: {e}")
else:
    print("  Skipping — env vars not set")

# ============================================================
# DONE
# ============================================================
print(f"\n{'='*60}")
print("DONE")
print(f"{'='*60}")
print(f"  HF dataset: {hf_path}")
print(f"  Parquet:    {final_parquet}")
print(f"  Rows:       {len(final_df)}")
print(f"  Free disk:  {disk_free_gb():.1f}GB")
print(f"{'='*60}")
