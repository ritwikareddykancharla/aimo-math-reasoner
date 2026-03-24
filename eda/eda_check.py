#!/usr/bin/env python3
"""
Comprehensive Kaggle dataset EDA with full diagnostics.

Setup:
    pip install python-dotenv kagglehub pandas pyarrow

Usage:
    # Default dataset:
    python3 eda_check.py

    # Specify a dataset:
    python3 eda_check.py --dataset ritwikakancharla/nemotron-math-v2-sft-hard-tools

    # Use a local parquet file directly (skip download):
    python3 eda_check.py --local /path/to/data.parquet

Env vars (set in .env or export):
    KAGGLE_USERNAME, KAGGLE_KEY, KAGGLE_API_TOKEN
"""
import sys
import os
import argparse

# ─── CLI args ────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Kaggle dataset EDA")
parser.add_argument("--dataset", default="ritwikakancharla/nemotron-math-v2-sft-high-medium-tools",
                    help="Kaggle dataset slug (owner/name)")
parser.add_argument("--local", default=None,
                    help="Path to local parquet file (skips download)")
args = parser.parse_args()

DATASET = args.dataset

print("=" * 60)
print("  ENVIRONMENT DIAGNOSTICS")
print("=" * 60)
print(f"  Python: {sys.version}")
print(f"  Python path: {sys.executable}")
print(f"  CWD: {os.getcwd()}")

# ─── 1. Load .env ───────────────────────────────────────────
try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"  ✅ Loaded .env from: {env_path}")
    else:
        # Also check CWD
        env_path2 = os.path.join(os.getcwd(), '.env')
        if os.path.exists(env_path2):
            load_dotenv(env_path2)
            print(f"  ✅ Loaded .env from: {env_path2}")
        else:
            print(f"  ⚠️  No .env found (checked {env_path} and {env_path2})")
except ImportError:
    print("  ⚠️  python-dotenv not installed, using env vars only")

# ─── 2. Check all Kaggle env vars ──────────────────────────
print(f"\n--- Kaggle Environment Variables ---")
kaggle_vars = [
    "KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN",
    "KAGGLE_API_KEY", "KAGGLE_CONFIG_DIR"
]
for var in kaggle_vars:
    val = os.environ.get(var)
    if val:
        masked = val[:8] + "..." + val[-4:] if len(val) > 12 else val
        print(f"  {var} = {masked}")
    else:
        print(f"  {var} = (not set)")

# ─── 3. Check kaggle.json ──────────────────────────────────
print(f"\n--- Kaggle Config Files ---")
kaggle_json_paths = [
    os.path.expanduser("~/.kaggle/kaggle.json"),
]
config_dir = os.environ.get("KAGGLE_CONFIG_DIR")
if config_dir:
    kaggle_json_paths.append(os.path.join(config_dir, "kaggle.json"))

for p in kaggle_json_paths:
    if p and os.path.exists(p):
        import json
        with open(p) as f:
            cfg = json.load(f)
        print(f"  ✅ Found: {p}")
        for k, v in cfg.items():
            masked = str(v)[:8] + "..." if len(str(v)) > 8 else v
            print(f"     {k} = {masked}")
    else:
        print(f"  ❌ Not found: {p}")

# ─── 4. Check kagglehub version ────────────────────────────
print(f"\n--- KaggleHub ---")
try:
    import kagglehub
    print(f"  Version: {kagglehub.__version__}")
    print(f"  Location: {kagglehub.__file__}")
except ImportError:
    print("  ❌ kagglehub not installed!")
    if not args.local:
        sys.exit(1)

# ─── 5. Credentials check ──────────────────────────────────
print(f"\n--- KaggleHub Credentials Check ---")
try:
    from kagglehub.config import get_kaggle_credentials
    creds = get_kaggle_credentials()
    print(f"  Credentials: {creds}")
except Exception as e:
    print(f"  ⚠️  Could not get credentials: {e}")

# ─── 6. Download or use local path ─────────────────────────
if args.local:
    parquet_file = args.local
    path = os.path.dirname(parquet_file)
    print(f"\n--- Using local file: {parquet_file} ---")
else:
    print(f"\n--- Attempting Download: {DATASET} ---")
    try:
        path = kagglehub.dataset_download(DATASET)
        print(f"  ✅ Downloaded to: {path}")
    except Exception as e:
        print(f"  ❌ Download failed: {e}")
        print(f"\n  Trying alternative: kaggle CLI download...")

        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, "-m", "kaggle", "datasets", "download",
                 DATASET, "-p", "/tmp/dataset", "--unzip"],
                capture_output=True, text=True, timeout=300
            )
            if result.returncode == 0:
                path = "/tmp/dataset"
                print(f"  ✅ CLI download succeeded: {path}")
            else:
                print(f"  ❌ CLI also failed: {result.stderr}")
                print(f"\n  ⚠️  AUTH FAILED. Options:")
                print(f"     1. Make dataset public on Kaggle")
                print(f"     2. Download manually, run with: --local /path/to/data.parquet")
                sys.exit(1)
        except (FileNotFoundError, Exception) as e2:
            print(f"  ❌ kaggle CLI failed: {e2}")
            sys.exit(1)

    # Find parquet in downloaded path
    parquet_file = None
    print(f"\n--- Files in dataset ---")
    for f in sorted(os.listdir(path)):
        fpath = os.path.join(path, f)
        size = os.path.getsize(fpath) / 1e6
        print(f"  {f} ({size:.1f} MB)")
        if f.endswith('.parquet') and parquet_file is None:
            parquet_file = fpath

if not parquet_file or not os.path.exists(parquet_file):
    print("  ❌ No parquet file found!")
    sys.exit(1)

# ─── 7. EDA ─────────────────────────────────────────────────
import pandas as pd
import json

df = pd.read_parquet(parquet_file)

print(f"\n{'=' * 60}")
print(f"  DATASET EDA: {DATASET}")
print(f"{'=' * 60}")
print(f"  Total rows: {len(df):,}")
print(f"  Columns: {list(df.columns)}")
print(f"\n--- Column Types ---")
for col in df.columns:
    print(f"  {col}: {df[col].dtype}")

print(f"\n--- Accuracy Distribution ---")
acc = df['accuracy'].value_counts().sort_index()
for a, c in acc.items():
    print(f"  {a:.3f} → {c:,} problems")
print(f"  Total: {len(df):,}")

if 'reasoning_depth' in df.columns:
    print(f"\n--- Reasoning Depth ---")
    rd = df['reasoning_depth'].value_counts()
    for r, c in rd.items():
        print(f"  {r}: {c:,}")

    print(f"\n--- Accuracy × Reasoning Depth ---")
    cross = df.groupby(['accuracy', 'reasoning_depth']).size().unstack(fill_value=0)
    print(cross.to_string())

if 'data_source' in df.columns:
    print(f"\n--- Top Data Sources ---")
    ds_counts = df['data_source'].value_counts().head(15)
    for s, c in ds_counts.items():
        print(f"  {s}: {c:,}")

if 'has_tools' in df.columns:
    print(f"\n--- Has Tools ---")
    print(df['has_tools'].value_counts().to_string())

# ─── Message analysis ───────────────────────────────────────
print(f"\n--- Message Analysis ---")
if 'messages' in df.columns:
    def parse_msgs(m):
        if isinstance(m, str):
            return json.loads(m)
        return m

    first = df['messages'].iloc[0]
    print(f"  Type: {type(first).__name__}")

    sample = parse_msgs(first)
    print(f"  Turns per example: {len(sample)}")
    for m in sample:
        role = m.get('role', '?')
        content_len = len(m.get('content', ''))
        print(f"    {role}: {content_len:,} chars")

    # Word counts
    def assistant_words(row):
        msgs = parse_msgs(row)
        for m in msgs:
            if m.get('role') == 'assistant':
                return len(m.get('content', '').split())
        return 0

    print(f"\n  Computing word count stats...")
    wc = df['messages'].apply(assistant_words)
    print(f"  Assistant response word counts:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"    p{p}: {int(wc.quantile(p/100)):,}")
    print(f"    max: {int(wc.max()):,}")
    print(f"    mean: {wc.mean():.0f}")

    # Code blocks
    def has_code(row):
        msgs = parse_msgs(row)
        for m in msgs:
            if m.get('role') == 'assistant' and '```' in m.get('content', ''):
                return True
        return False

    code_pct = df['messages'].apply(has_code).mean() * 100
    print(f"\n  Has code blocks: {code_pct:.1f}%")

    # Boxed answers
    def has_boxed(row):
        msgs = parse_msgs(row)
        for m in msgs:
            if m.get('role') == 'assistant' and '\\boxed' in m.get('content', ''):
                return True
        return False

    boxed_pct = df['messages'].apply(has_boxed).mean() * 100
    print(f"  Has \\boxed{{}} answer: {boxed_pct:.1f}%")

# ─── Expected answers ────────────────────────────────────────
if 'expected_answer' in df.columns:
    print(f"\n--- Expected Answers ---")
    ans = df['expected_answer']
    print(f"  Type: {ans.dtype}")
    print(f"  Sample: {ans.head(5).tolist()}")

    numeric = ans.apply(lambda x: str(x).strip().lstrip('-').isdigit())
    print(f"  Purely numeric: {numeric.mean()*100:.1f}%")

    if numeric.mean() > 0.1:
        nums = ans[numeric].astype(int)
        print(f"  Numeric range: [{nums.min()}, {nums.max()}]")
        print(f"  Numeric count: {len(nums):,}")

print(f"\n{'=' * 60}")
print(f"  EDA COMPLETE ✅")
print(f"{'=' * 60}")
