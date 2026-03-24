#!/usr/bin/env python3
"""
Quantize trained model to 4-bit and upload to Kaggle.
Run AFTER training completes.

Usage:
    python3.12 quantize_and_upload.py --model-dir model-full-ft-final
    python3.12 quantize_and_upload.py --model-dir model-full-ft-final --no-upload
"""

import os, sys, time, shutil, argparse, json
from dotenv import load_dotenv
load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--model-dir", required=True, help="Path to full bf16 model")
parser.add_argument("--output-dir", default=None, help="Output dir for quantized model")
parser.add_argument("--no-upload", action="store_true", help="Skip Kaggle upload")
parser.add_argument("--bits", type=int, default=4, choices=[4, 8], help="Quantization bits")
parser.add_argument("--dataset-name", default="gpt-oss-120b-aimo3-sft-v4",
                    help="Kaggle model name")
args = parser.parse_args()

KAGGLE_USER = os.environ.get("KAGGLE_USERNAME", "")
HF_TOKEN    = os.environ.get("HF_TOKEN", "")

output_dir = args.output_dir or f"{args.model_dir}-{args.bits}bit"

print("=" * 60)
print(f"  Quantize + Upload")
print(f"  Source:  {args.model_dir}")
print(f"  Output:  {output_dir}")
print(f"  Bits:    {args.bits}")
print("=" * 60)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── Load in quantized mode ──
print(f"\n  Loading model in {args.bits}-bit...")
t0 = time.time()

if args.bits == 4:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
else:
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(
    args.model_dir,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    token=HF_TOKEN,
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    args.model_dir,
    token=HF_TOKEN,
    trust_remote_code=True,
)

load_time = time.time() - t0
print(f"  Loaded in {load_time:.0f}s")
print(f"  GPU memory: {torch.cuda.memory_allocated()/1e9:.1f}GB")

# ── Save quantized ──
print(f"\n  Saving quantized model to {output_dir}...")
t0 = time.time()
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Calculate size
total_size = sum(
    os.path.getsize(os.path.join(dp, f))
    for dp, _, filenames in os.walk(output_dir)
    for f in filenames
) / 1e9

print(f"  Saved in {time.time()-t0:.0f}s | Size: {total_size:.1f}GB")

# ── Upload to Kaggle ──
if not args.no_upload:
    if not KAGGLE_USER:
        print("\n  ⚠️  KAGGLE_USERNAME not set — skipping upload")
    else:
        print(f"\n  Uploading to Kaggle: {KAGGLE_USER}/{args.dataset_name}")
        try:
            import kagglehub
            handle = f"{KAGGLE_USER}/{args.dataset_name}/transformers/default"
            t0 = time.time()
            kagglehub.model_upload(
                handle, output_dir,
                version_notes=f"SFT v4 | {args.bits}-bit quantized | {total_size:.1f}GB",
                license_name="Apache 2.0",
            )
            print(f"  ✅ Uploaded in {(time.time()-t0)/60:.1f} mins")
            print(f"  → https://www.kaggle.com/models/{KAGGLE_USER}/{args.dataset_name}")
        except Exception as e:
            print(f"  ⚠️  Upload failed: {e}")
            print(f"  Quantized model at: {output_dir}")
else:
    print("\n  Skipping Kaggle upload (--no-upload)")

print(f"\n{'=' * 60}")
print(f"  DONE")
print(f"  Quantized model: {output_dir}/ ({total_size:.1f}GB)")
print(f"{'=' * 60}")
