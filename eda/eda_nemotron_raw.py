#!/usr/bin/env python3
"""
Deep EDA on RAW nvidia/Nemotron-Math-v2 from HuggingFace.
Analyzes what's actually in each split BEFORE filtering.

Setup:
    pip install python-dotenv datasets pandas

Usage:
    python3 eda_nemotron_raw.py                      # EDA on all splits
    python3 eda_nemotron_raw.py --split high_part00   # EDA on one split
    python3 eda_nemotron_raw.py --max-rows 10000      # Quick EDA on subset

Requires HF_TOKEN in .env or environment.
"""
import os
import sys
import json
import argparse
import time
from collections import Counter

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("ERROR: Set HF_TOKEN in .env or environment")
    sys.exit(1)

parser = argparse.ArgumentParser(description="EDA on raw Nemotron-Math-v2")
parser.add_argument("--split", default=None,
                    help="Specific split to analyze (e.g. high_part00)")
parser.add_argument("--max-rows", type=int, default=None,
                    help="Max rows to load per split (for quick testing)")
args = parser.parse_args()

HF_DATASET = "nvidia/Nemotron-Math-v2"

# Splits to analyze
if args.split:
    SPLITS = [args.split]
else:
    SPLITS = ["high_part00", "high_part01", "high_part02", "medium"]

from datasets import load_dataset
import pandas as pd

print("=" * 70)
print(f"  RAW EDA: {HF_DATASET}")
print("=" * 70)

for split_name in SPLITS:
    print(f"\n{'━' * 70}")
    print(f"  SPLIT: {split_name}")
    print(f"{'━' * 70}")

    t0 = time.time()
    if args.max_rows:
        ds = load_dataset(HF_DATASET, token=HF_TOKEN, split=f"{split_name}[:{args.max_rows}]")
    else:
        ds = load_dataset(HF_DATASET, token=HF_TOKEN, split=split_name)
    print(f"  Loaded {len(ds):,} rows in {time.time()-t0:.0f}s")
    print(f"  Columns: {ds.column_names}")

    # Convert a sample to explore structure
    df = ds.to_pandas()

    # ─── Basic stats ────────────────────────────────────────
    print(f"\n  --- Column Types ---")
    for col in df.columns:
        print(f"    {col}: {df[col].dtype}")

    # ─── Data source distribution ───────────────────────────
    if 'data_source' in df.columns:
        print(f"\n  --- Data Sources ---")
        src = df['data_source'].value_counts()
        for s, c in src.head(20).items():
            pct = c / len(df) * 100
            print(f"    {s}: {c:,} ({pct:.1f}%)")

    # ─── Tools presence ─────────────────────────────────────
    if 'tools' in df.columns:
        has_tools = df['tools'].apply(
            lambda x: x is not None and (not isinstance(x, list) or len(x) > 0)
        )
        print(f"\n  --- Tool Usage ---")
        print(f"    With tools: {has_tools.sum():,} ({has_tools.mean()*100:.1f}%)")
        print(f"    Without tools: {(~has_tools).sum():,} ({(~has_tools).mean()*100:.1f}%)")

    # ─── Metadata / accuracy fields ─────────────────────────
    if 'metadata' in df.columns:
        print(f"\n  --- Metadata Keys (first row) ---")
        meta0 = df['metadata'].iloc[0]
        if isinstance(meta0, dict):
            for k, v in meta0.items():
                if isinstance(v, dict):
                    print(f"    {k}: {list(v.keys())}")
                else:
                    print(f"    {k}: {type(v).__name__} = {str(v)[:80]}")

        # Extract all accuracy fields
        acc_keys = [k for k in (meta0 or {}).keys() if 'reason' in k.lower()]
        print(f"\n  --- Accuracy Fields ---")
        for acc_key in acc_keys:
            accs = df['metadata'].apply(
                lambda m: (m or {}).get(acc_key, {}).get('accuracy', None)
                if isinstance((m or {}).get(acc_key), dict) else None
            )
            valid = accs.dropna()
            if len(valid) > 0:
                print(f"\n    {acc_key}:")
                print(f"      Non-null: {len(valid):,} / {len(df):,} ({len(valid)/len(df)*100:.1f}%)")
                print(f"      Min: {valid.min():.3f}, Max: {valid.max():.3f}")
                print(f"      Mean: {valid.mean():.3f}, Median: {valid.median():.3f}")
                # Distribution
                hard = (valid < 0.5).sum()
                easy = (valid >= 0.5).sum()
                print(f"      Hard (acc<0.5): {hard:,} ({hard/len(valid)*100:.1f}%)")
                print(f"      Easy (acc>=0.5): {easy:,} ({easy/len(valid)*100:.1f}%)")

                # Accuracy value distribution
                vc = valid.round(3).value_counts().sort_index()
                print(f"      Value distribution:")
                for a, c in vc.items():
                    print(f"        {a:.3f}: {c:,}")

    # ─── Message structure ──────────────────────────────────
    if 'messages' in df.columns:
        print(f"\n  --- Message Structure ---")
        msg0 = df['messages'].iloc[0]
        if hasattr(msg0, 'tolist'):
            msg0 = msg0.tolist()

        if isinstance(msg0, list):
            print(f"    Turns in first example: {len(msg0)}")
            roles = [m.get('role', '?') for m in msg0 if isinstance(m, dict)]
            print(f"    Roles: {' → '.join(roles[:20])}{'...' if len(roles) > 20 else ''}")

            # Role distribution across all rows
            turn_counts = df['messages'].apply(
                lambda m: len(m.tolist() if hasattr(m, 'tolist') else m)
            )
            print(f"\n    Turns per example:")
            for p in [10, 25, 50, 75, 90]:
                print(f"      p{p}: {int(turn_counts.quantile(p/100))}")
            print(f"      max: {int(turn_counts.max())}")

    # ─── Expected answers ───────────────────────────────────
    if 'expected_answer' in df.columns:
        print(f"\n  --- Expected Answers ---")
        ans = df['expected_answer'].astype(str)

        # Numeric check
        numeric = ans.apply(lambda x: x.strip().lstrip('-').isdigit())
        print(f"    Purely numeric: {numeric.sum():,} ({numeric.mean()*100:.1f}%)")

        # Integer 0-999 (AIMO format)
        def is_aimo_format(x):
            s = str(x).strip()
            if s.lstrip('-').isdigit():
                n = int(s)
                return 0 <= n <= 999
            return False

        aimo = ans.apply(is_aimo_format)
        print(f"    AIMO format (int 0-999): {aimo.sum():,} ({aimo.mean()*100:.1f}%)")

        # Sample non-numeric
        non_numeric = ans[~numeric]
        if len(non_numeric) > 0:
            print(f"\n    Sample non-numeric answers:")
            for i, a in enumerate(non_numeric.head(5)):
                print(f"      [{i}] {a[:120]}")

    # ─── Cross-tabulation: data_source × accuracy ───────────
    if 'data_source' in df.columns and 'metadata' in df.columns:
        print(f"\n  --- Data Source × Difficulty ---")
        # Use the first available accuracy key
        acc_key = acc_keys[0] if acc_keys else None
        if acc_key:
            df['_tmp_acc'] = df['metadata'].apply(
                lambda m: (m or {}).get(acc_key, {}).get('accuracy', None)
                if isinstance((m or {}).get(acc_key), dict) else None
            )
            df['_tmp_hard'] = df['_tmp_acc'].apply(
                lambda x: 'hard' if x is not None and x < 0.5 else ('easy' if x is not None else 'unknown')
            )
            cross = pd.crosstab(df['data_source'], df['_tmp_hard'])
            print(cross.to_string())
            df.drop(columns=['_tmp_acc', '_tmp_hard'], inplace=True)

    # ─── Summary ────────────────────────────────────────────
    print(f"\n  {'─' * 50}")
    print(f"  SPLIT {split_name} SUMMARY:")
    print(f"    Rows: {len(df):,}")
    if 'data_source' in df.columns:
        print(f"    Sources: {df['data_source'].nunique()}")
    if 'uuid' in df.columns:
        print(f"    Unique problems: {df['uuid'].nunique():,}")

    del df, ds
    import gc; gc.collect()

print(f"\n{'=' * 70}")
print(f"  RAW EDA COMPLETE ✅")
print(f"{'=' * 70}")
