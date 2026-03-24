"""
AIMO3: Full Fine-Tuning on 8× H100 (DeepSpeed ZeRO-3)
======================================================
NO LoRA. NO quantization. Full parameter training on all 120B params.
DeepSpeed ZeRO Stage 3 shards model across 8 GPUs.

Dataset: v4 (79K hard problems, all sources, reason_high_with_tool < 0.5)
Upsampling: acc=0.125 → 4x, acc=0.25 → 2x, acc=0.375 → 1x (~193K effective)

Setup (run once on the instance):
    pip install deepspeed transformers trl datasets accelerate torch
    pip install flash-attn --no-build-isolation

Launch:
    accelerate launch --config_file accelerate_config.yaml train_full_8xh100.py

    # Or with deepspeed directly:
    deepspeed --num_gpus=8 train_full_8xh100.py

Env vars from .env: HF_TOKEN, KAGGLE_USERNAME, KAGGLE_API_TOKEN, WANDB_API_KEY
"""

import os, sys, time, gc, json, shutil, glob
from dotenv import load_dotenv
load_dotenv()

KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME", "")
DATA_DIR = os.environ.get("DATA_DIR", "./data/nemotron-sft-v4/hf_dataset")

# ============================================================
# W&B
# ============================================================
WANDB_KEY = os.environ.get("WANDB_API_KEY", "")
if WANDB_KEY:
    try:
        import wandb
        wandb.login(key=WANDB_KEY)
        os.environ["WANDB_PROJECT"] = "aimo3-full-ft-8xh100"
        REPORT_TO = "wandb"
        print("W&B: enabled")
    except Exception:
        REPORT_TO = "none"
else:
    REPORT_TO = "none"

# ============================================================
# IMPORTS
# ============================================================
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    TrainerCallback,
)
from trl import SFTTrainer, SFTConfig
import torch

set_seed(42)

MAX_SEQ_LEN = 4096
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
IS_MAIN = LOCAL_RANK == 0

def log(msg):
    if IS_MAIN:
        print(msg)

# ============================================================
# DEEPSPEED CONFIG (ZeRO Stage 3)
# ============================================================
# This gets passed to the trainer — no separate file needed

DS_CONFIG = {
    "bf16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": True,
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 10,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False,
}

# Save DS config to file (some launchers need it)
ds_config_path = "ds_config_zero3.json"
if IS_MAIN:
    with open(ds_config_path, "w") as f:
        json.dump(DS_CONFIG, f, indent=2)

# ============================================================
# LOAD MODEL — full precision, no quantization
# ============================================================
log("\n" + "=" * 60)
log("  Loading model (full bf16, no quantization)")
log("=" * 60)

MODEL_NAME = "openai/gpt-oss-120b"  # 117B params, 5.1B active (MoE), native MXFP4

# Try loading — the model will be sharded by DeepSpeed ZeRO-3
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",  # saves memory + faster
    token=os.environ.get("HF_TOKEN"),
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    token=os.environ.get("HF_TOKEN"),
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

log(f"  Model loaded: {MODEL_NAME}")
log(f"  Parameters: {sum(p.numel() for p in model.parameters())/1e9:.1f}B")

# ============================================================
# LOAD & FORMAT DATASET
# ============================================================
log("\n" + "=" * 60)
log("  Loading dataset")
log("=" * 60)

ds = load_from_disk(DATA_DIR)
log(f"  Total: {len(ds)} rows")

# Upsample hardest problems so model sees them more
# acc=0.125 → 4x, acc=0.25 → 2x, acc=0.375+ → 1x
log("  Upsampling hardest problems...")
import pandas as pd
df = ds.to_pandas()

acc_weights = {0.125: 4, 0.25: 2, 0.375: 1}
df['_repeat'] = df['accuracy'].map(lambda a: acc_weights.get(a, 1))

log(f"  Upsample plan:")
for acc_val in sorted(df['accuracy'].unique()):
    n = (df['accuracy'] == acc_val).sum()
    w = acc_weights.get(acc_val, 1)
    log(f"    acc={acc_val:.3f}: {n:>7d} × {w}x = {n*w:>7d}")

df = df.loc[df.index.repeat(df['_repeat'])].reset_index(drop=True)
df = df.drop(columns=['_repeat'])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

from datasets import Dataset as HFDataset
ds = HFDataset.from_pandas(df)
log(f"  After upsample: {len(ds)} rows")

def format_for_sft(example):
    msgs = example['messages']
    text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False,
    )
    return {"text": text}

log("  Formatting...")
t0 = time.time()
sft_data = ds.map(format_for_sft, num_proc=8, remove_columns=ds.column_names)
sft_data = sft_data.filter(lambda x: len(x['text']) > 100)
log(f"  Ready: {len(sft_data)} samples in {time.time()-t0:.0f}s")

# Token length check
if IS_MAIN:
    sample_lens = [len(tokenizer.encode(sft_data[i]['text'])) for i in range(min(500, len(sft_data)))]
    sample_lens.sort()
    n = len(sample_lens)
    log(f"  Tokens: median={sample_lens[n//2]} p90={sample_lens[9*n//10]} max={sample_lens[-1]}")

train_data = sft_data

# ============================================================
# TRAIN
# ============================================================
log("\n" + "=" * 60)
log("  TRAINING — Full fine-tune, 8× H100, DeepSpeed ZeRO-3")
log(f"  {len(train_data)} samples | lr=2e-5 | 1 epoch")
log("=" * 60)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    args=SFTConfig(
        output_dir="sft-full-8xh100",
        deepspeed=ds_config_path,
        per_device_train_batch_size=2,       # per GPU — 8 GPUs × 2 = 16
        gradient_accumulation_steps=2,        # effective batch = 32
        num_train_epochs=1,
        learning_rate=2e-5,
        bf16=True,
        max_seq_length=MAX_SEQ_LEN,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,                    # small warmup for stability with full FT
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        logging_steps=1,
        eval_strategy="no",
        max_grad_norm=1.0,
        report_to=REPORT_TO,
        run_name="full-ft-8xh100",
        dataset_text_field="text",
        packing=True,
        ddp_timeout=7200,                     # 2h timeout for large model syncs
    ),
    processing_class=tokenizer,
)

log("\n  Starting training...")
t0 = time.time()
checkpoints = sorted(glob.glob("sft-full-8xh100/checkpoint-*"))
if checkpoints:
    log(f"  Resuming from {checkpoints[-1]}...")
    trainer.train(resume_from_checkpoint=checkpoints[-1])
else:
    log("  Starting fresh...")
    trainer.train()

train_time = time.time() - t0
final_step = trainer.state.global_step
final_loss = trainer.state.log_history[-1].get('loss', '?') if trainer.state.log_history else '?'
log(f"\n  Done in {train_time/3600:.1f}h | step={final_step} | loss={final_loss}")

# ============================================================
# SAVE FINAL MODEL
# ============================================================
log("\n  Saving final model...")
save_dir = "model-full-ft-final"
trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)
log(f"  Saved to {save_dir}")

# ============================================================
# UPLOAD TO KAGGLE (from main process only)
# ============================================================
if IS_MAIN and KAGGLE_USERNAME:
    log("\n  Uploading to Kaggle...")
    try:
        import kagglehub
        handle = f"{KAGGLE_USERNAME}/gpt-oss-120b-aimo3-sft-v4/transformers/default"
        t0 = time.time()
        kagglehub.model_upload(
            handle, save_dir,
            version_notes=f"Full FT 8xH100 | step={final_step} | loss={final_loss} | {len(train_data)} samples",
            license_name="Apache 2.0",
        )
        log(f"  ✅ Upload done in {(time.time()-t0)/60:.1f} mins")
    except Exception as e:
        log(f"  ⚠️  Upload failed: {e}")
        log(f"  Model saved locally at {save_dir}")

log("\n" + "=" * 60)
log("  COMPLETE")
log(f"  Time: {train_time/3600:.1f}h")
log(f"  Loss: {final_loss}")
log("=" * 60)

if WANDB_KEY:
    try:
        wandb.finish()
    except Exception:
        pass
