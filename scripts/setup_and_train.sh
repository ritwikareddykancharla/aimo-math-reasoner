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
import os, kagglehub, math, pandas as pd
from datasets import load_from_disk, Dataset

data_dir = os.environ.get('DATA_DIR', './data/nemotron-math-v2')
username = os.environ['KAGGLE_USERNAME']

# Download
print("Downloading ritwikakancharla/nemotron-math-v2-filtered-high...")
path = kagglehub.dataset_download('ritwikakancharla/nemotron-math-v2-filtered-high')
print(f"  Downloaded to: {path}")

# Load as HuggingFace dataset (Arrow format)
ds = load_from_disk(path)
print(f"\n  Raw: {len(ds)} rows")
print(f"  Columns: {ds.column_names}")
print(f"  Sample: { {k: str(v)[:80] for k, v in ds[0].items()} }")

# Find answer column
answer_col = None
for col in ds.column_names:
    if col.lower() in ('answer', 'expected_answer', 'final_answer', 'target'):
        answer_col = col
        break
assert answer_col, f"No answer column found in {ds.column_names}"
print(f"  Answer column: {answer_col}")

# Filter: integer answers in [0, 99999]
def is_valid_aimo3(example):
    try:
        v = float(example[answer_col])
        if not math.isfinite(v):
            return False
        return v == int(v) and 0 <= int(v) <= 99999
    except (ValueError, TypeError, OverflowError):
        return False

ds_filtered = ds.filter(is_valid_aimo3, num_proc=4)

# Convert answers to clean int strings
def clean_answer(example):
    example[answer_col] = str(int(float(example[answer_col])))
    return example

ds_filtered = ds_filtered.map(clean_answer, num_proc=4)

print(f"  After AIMO3 filter (int 0-99999): {len(ds_filtered)} rows")
print(f"  Removed: {len(ds) - len(ds_filtered)} rows")

# Save locally as HF dataset for training
os.makedirs(data_dir, exist_ok=True)
ds_filtered.save_to_disk(data_dir)
print(f"  Saved HF dataset to: {data_dir}")

# Also save as parquet for Kaggle upload
upload_dir = data_dir + '_upload'
os.makedirs(upload_dir, exist_ok=True)
ds_filtered.to_parquet(os.path.join(upload_dir, 'data.parquet'))

# Write README
readme = f"""# Nemotron Math v2 — Filtered High (AIMO3 Ready)

## What is this?
A filtered version of [nvidia/Nemotron-Math-v2](https://huggingface.co/datasets/nvidia/Nemotron-Math-v2)
optimized for the [AIMO3 competition](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3).

## Filtering Pipeline
1. **Source**: nvidia/Nemotron-Math-v2 (full dataset)
2. **V1 filter**: Keep only high-quality AoPS competition math problems
3. **V2 filter (this version)**: Keep only problems with **integer answers between 0 and 99999**
   - Matches AIMO3 competition format exactly
   - Removed fractions, decimals, negatives, and answers > 99999

## Stats
- **V1 (filtered-high)**: {len(ds)} rows
- **V2 (AIMO3-ready)**: {len(ds_filtered)} rows
- **Removed**: {len(ds) - len(ds_filtered)} rows ({(len(ds) - len(ds_filtered))/len(ds)*100:.1f}%)

## Columns
{chr(10).join(f'- `{col}`' for col in ds_filtered.column_names)}

## Usage
```python
from datasets import load_dataset
ds = load_dataset('ritwikakancharla/nemotron-math-v2-filtered-high')
```

## License
Apache 2.0
"""
with open(os.path.join(upload_dir, 'README.md'), 'w') as f:
    f.write(readme)
print(f"  Wrote README.md")

# Upload as V2
print(f"\n  Uploading as v2 to {username}/nemotron-math-v2-filtered-high...")
try:
    kagglehub.dataset_upload(
        f'{username}/nemotron-math-v2-filtered-high',
        upload_dir,
        version_notes='V2: Filtered to AIMO3 format (integer answers 0-99999 only)',
    )
    print("  Uploaded!")
except Exception as e:
    print(f"  Upload failed: {e}")
    print(f"  Data saved locally at: {upload_dir}/")

print(f"\n  Final: {len(ds_filtered)} rows ready for training")
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
