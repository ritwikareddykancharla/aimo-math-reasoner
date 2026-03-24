#!/bin/bash
set -e

# Load .env
if [ -f .env ]; then
    echo "Loading .env..."
    set -a; source .env; set +a
else
    echo "ERROR: .env not found! Run: cp .env.template .env"
    exit 1
fi

echo ""
echo "============================================"
echo "  STEP 1: Install dependencies"
echo "============================================"

pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install --upgrade \
    unsloth unsloth_zoo trl datasets transformers accelerate \
    peft bitsandbytes huggingface_hub hf_transfer \
    kagglehub google-genai python-dotenv wandb vllm

echo ""
echo "============================================"
echo "  STEP 2: Setup Kaggle credentials"
echo "============================================"

# kagglehub uses ~/.kaggle/kaggle.json (legacy format)
mkdir -p ~/.kaggle
echo "{\"username\":\"${KAGGLE_USERNAME}\",\"key\":\"${KAGGLE_API_TOKEN}\"}" > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
echo "  Wrote ~/.kaggle/kaggle.json for user: ${KAGGLE_USERNAME}"

echo ""
echo "============================================"
echo "  STEP 3: Download Qwen3-32B"
echo "============================================"

export HF_HUB_ENABLE_HF_TRANSFER=1
python3 -c "
import os
from huggingface_hub import snapshot_download
model_dir = os.environ.get('MODEL_DIR', './models/Qwen3-32B')
model_name = os.environ.get('MODEL_NAME', 'Qwen/Qwen3-32B')
print(f'Downloading {model_name} to {model_dir}...')
path = snapshot_download(repo_id=model_name, local_dir=model_dir, token=os.environ.get('HF_TOKEN'))
print(f'Done: {path}')
"

echo ""
echo "============================================"
echo "  STEP 4: Download, Filter & Upload Dataset"
echo "============================================"

python3 << 'EOF'
import os, kagglehub, pandas as pd

data_dir = os.environ.get('DATA_DIR', './data/nemotron-math-v2')
username = os.environ['KAGGLE_USERNAME']

# Download the existing filtered-high dataset
print("Downloading ritwikakancharla/nemotron-math-v2-filtered-high...")
path = kagglehub.dataset_download('ritwikakancharla/nemotron-math-v2-filtered-high')
print(f"  Downloaded to: {path}")

# Find data files
data_files = []
for root, dirs, files in os.walk(path):
    for f in files:
        if f.endswith(('.parquet', '.csv', '.jsonl')):
            data_files.append(os.path.join(root, f))
            print(f"  Found: {f}")

assert data_files, f"No data files found in {path}"

# Load
ext = data_files[0].split('.')[-1]
if ext == 'parquet':
    df = pd.concat([pd.read_parquet(f) for f in data_files if f.endswith('.parquet')])
elif ext == 'csv':
    df = pd.concat([pd.read_csv(f) for f in data_files if f.endswith('.csv')])
elif ext == 'jsonl':
    df = pd.concat([pd.read_json(f, lines=True) for f in data_files if f.endswith('.jsonl')])

print(f"\n  Raw: {len(df)} rows")
print(f"  Columns: {list(df.columns)}")

# Find answer column
answer_col = None
for col in df.columns:
    if col.lower() in ('answer', 'expected_answer', 'final_answer', 'target'):
        answer_col = col
        break
assert answer_col, f"No answer column found in {list(df.columns)}"
print(f"  Answer column: {answer_col}")
print(f"  Sample answers: {df[answer_col].head(5).tolist()}")

# Filter: integer answers in [0, 99999]
def is_valid_aimo3(val):
    try:
        v = float(val)
        return v == int(v) and 0 <= int(v) <= 99999
    except (ValueError, TypeError):
        return False

mask = df[answer_col].apply(is_valid_aimo3)
df_filtered = df[mask].copy()
df_filtered[answer_col] = df_filtered[answer_col].apply(lambda x: str(int(float(x))))

print(f"  After AIMO3 filter (int 0-99999): {len(df_filtered)} rows")
print(f"  Removed: {len(df) - len(df_filtered)} rows")

# Save locally
os.makedirs(data_dir, exist_ok=True)
out_path = os.path.join(data_dir, 'data.parquet')
df_filtered.to_parquet(out_path, index=False)
print(f"  Saved: {out_path}")

# Write README
readme = f"""# Nemotron Math v2 — Filtered High (AIMO3 Ready)

## What is this?
A filtered version of [nvidia/Nemotron-Math-v2](https://huggingface.co/datasets/nvidia/Nemotron-Math-v2)
optimized for the [AIMO3 competition](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3).

## Filtering Pipeline
1. **Source**: nvidia/Nemotron-Math-v2 (full dataset)
2. **V1 filter**: Keep only high-quality AoPS competition math problems (high reasoning accuracy)
3. **V2 filter (this version)**: Keep only problems with **integer answers between 0 and 99999**
   - This matches the AIMO3 competition format exactly
   - Removed: fractions, decimals, negative numbers, and answers > 99999

## Stats
- **V1 (filtered-high)**: {len(df)} rows
- **V2 (AIMO3-ready)**: {len(df_filtered)} rows
- **Removed**: {len(df) - len(df_filtered)} rows ({(len(df) - len(df_filtered))/len(df)*100:.1f}%)

## Columns
{chr(10).join(f'- `{col}`' for col in df_filtered.columns)}

## Usage
```python
import kagglehub
import pandas as pd

path = kagglehub.dataset_download('{username}/nemotron-math-v2-filtered-high')
df = pd.read_parquet(path + '/data.parquet')
```

## License
Apache 2.0 (same as original Nemotron dataset)
"""

readme_path = os.path.join(data_dir, 'README.md')
with open(readme_path, 'w') as f:
    f.write(readme)
print(f"  Wrote: {readme_path}")

# Upload as V2 to the same dataset
print(f"\n  Uploading as v2 to {username}/nemotron-math-v2-filtered-high...")
try:
    kagglehub.dataset_upload(
        f'{username}/nemotron-math-v2-filtered-high',
        data_dir,
        version_notes='V2: Filtered to AIMO3 format (integer answers 0-99999 only)',
    )
    print("  Uploaded!")
except Exception as e:
    print(f"  Upload failed: {e}")
    print(f"  Data saved locally at: {data_dir}/")

# Also save as HuggingFace dataset for training script
from datasets import Dataset
ds = Dataset.from_pandas(df_filtered, preserve_index=False)
ds.save_to_disk(data_dir + '_hf')
print(f"  HF dataset saved to: {data_dir}_hf")
print(f"  Final: {len(ds)} rows")
EOF

echo ""
echo "============================================"
echo "  STEP 5: Verify everything"
echo "============================================"

python3 << 'EOF'
import os, torch

print("Credentials:")
for key in ['KAGGLE_USERNAME', 'KAGGLE_API_TOKEN', 'HF_TOKEN', 'GEMINI_API_KEY', 'WANDB_API_KEY']:
    val = os.environ.get(key, '')
    if not val:
        print(f"  ○ {key}: (not set)")
    else:
        print(f"  ✓ {key}: {val[:6]}...")

print(f"\nGPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_mem/1e9:.0f}GB)")

model_dir = os.environ.get('MODEL_DIR', './models/Qwen3-32B')
data_dir = os.environ.get('DATA_DIR', './data/nemotron-math-v2')
print(f"Model: {'✓' if os.path.exists(model_dir) else '✗'} {model_dir}")
print(f"Data:  {'✓' if os.path.exists(data_dir) else '✗'} {data_dir}")

# Test kagglehub
try:
    import kagglehub
    print(f"Kaggle: ✓ kagglehub ready")
except Exception as e:
    print(f"Kaggle: ✗ {e}")

# Test Gemini
try:
    from google import genai
    client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
    r = client.models.generate_content(model="gemini-2.5-flash", contents="Say OK")
    print(f"Gemini: ✓ {r.text.strip()}")
except Exception as e:
    print(f"Gemini: ✗ {e}")
EOF

echo ""
echo "============================================"
echo "  SETUP COMPLETE"
echo "============================================"
echo "  Now run: python3 train_dpo.py"
echo "============================================"
