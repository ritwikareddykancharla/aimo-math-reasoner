"""
AIMO3: Full Fine-Tuning on 8× H100 (DeepSpeed ZeRO-3)
======================================================
Full parameter training. DeepSpeed ZeRO-3 shards across 8 GPUs.

Dataset: v4 (79K hard problems, upsampled to ~193K effective)
Checkpoints: Every 200 steps → uploaded to Kaggle (~57GB each)
             Only last 3 Kaggle versions. 1 local checkpoint on disk.

Launch:
    export PATH=$PATH:/home/ssm-user/.local/bin
    deepspeed --num_gpus=8 training/sft/train_full_8xh100.py

Env vars from .env: HF_TOKEN, KAGGLE_USERNAME, KAGGLE_API_TOKEN, WANDB_API_KEY
"""

import os, sys, time, gc, json, shutil, glob, threading, inspect
from dotenv import load_dotenv
load_dotenv()

KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME", "")
KAGGLE_MODEL    = os.environ.get("KAGGLE_MODEL_NAME", "gpt-oss-120b-aimo3-sft-v4")
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

ds_config_path = "ds_config_zero3.json"
if IS_MAIN:
    with open(ds_config_path, "w") as f:
        json.dump(DS_CONFIG, f, indent=2)

# ============================================================
# KAGGLE UPLOAD CALLBACK (background thread, no quantization)
# ============================================================

class KaggleUploadCallback(TrainerCallback):
    """After each checkpoint save, upload to Kaggle in a background thread.
    Model is ~57GB on disk, so 3 versions fit Kaggle's 200GB limit.
    """

    def __init__(self, kaggle_user, model_name):
        self.kaggle_user = kaggle_user
        self.model_name = model_name
        self.bg_thread = None
        self.version_count = 0

    def on_save(self, args, state, control, **kwargs):
        if not IS_MAIN or not self.kaggle_user:
            return

        step = state.global_step
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
        if not os.path.isdir(checkpoint_dir):
            return

        # Wait for previous upload
        if self.bg_thread and self.bg_thread.is_alive():
            log(f"  [Kaggle] Waiting for previous upload...")
            self.bg_thread.join(timeout=1800)

        self.bg_thread = threading.Thread(
            target=self._upload,
            args=(checkpoint_dir, step, state),
            daemon=True,
        )
        self.bg_thread.start()
        log(f"  [Kaggle] Upload started for step {step}")

    def _upload(self, checkpoint_dir, step, state):
        try:
            import kagglehub
            loss = state.log_history[-1].get("loss", "?") if state.log_history else "?"

            # Calculate dir size
            total_gb = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, _, fns in os.walk(checkpoint_dir)
                for f in fns
            ) / 1e9

            log(f"  [Kaggle] Uploading step-{step} ({total_gb:.1f}GB)...")
            t0 = time.time()

            handle = f"{self.kaggle_user}/{self.model_name}/transformers/default"
            kagglehub.model_upload(
                handle, checkpoint_dir,
                version_notes=f"step-{step} | loss={loss} | {total_gb:.1f}GB",
                license_name="Apache 2.0",
            )

            self.version_count += 1
            mins = (time.time() - t0) / 60
            log(f"  [Kaggle] ✅ step-{step} uploaded in {mins:.1f}min (v{self.version_count})")

        except Exception as e:
            log(f"  [Kaggle] ⚠️ Upload failed for step {step}: {e}")

    def on_train_end(self, args, state, control, **kwargs):
        if self.bg_thread and self.bg_thread.is_alive():
            log("  [Kaggle] Waiting for final upload...")
            self.bg_thread.join(timeout=3600)


# ============================================================
# LOAD MODEL
# ============================================================
log("\n" + "=" * 60)
log("  Loading model (full bf16)")
log("=" * 60)

MODEL_NAME = "openai/gpt-oss-120b"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    attn_implementation="eager",
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

# Free HF cache to reclaim disk (~57GB)
for d in ["datasets", "hub"]:
    p = os.path.expanduser(f"~/.cache/huggingface/{d}")
    if os.path.exists(p):
        shutil.rmtree(p)
        log(f"  Cleared cache: {p}")
gc.collect()

# ============================================================
# LOAD & FORMAT DATASET
# ============================================================
log("\n" + "=" * 60)
log("  Loading dataset")
log("=" * 60)

ds = load_from_disk(DATA_DIR)
log(f"  Total: {len(ds)} rows")

# Upsample: acc=0.125 → 4x, acc=0.25 → 2x, acc=0.375 → 1x
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

if IS_MAIN:
    sample_lens = [len(tokenizer.encode(sft_data[i]['text'])) for i in range(min(500, len(sft_data)))]
    sample_lens.sort()
    n = len(sample_lens)
    log(f"  Tokens: median={sample_lens[n//2]} p90={sample_lens[9*n//10]} max={sample_lens[-1]}")

train_data = sft_data

# ============================================================
# CALLBACKS
# ============================================================
callbacks = []
if IS_MAIN and KAGGLE_USERNAME:
    callbacks.append(KaggleUploadCallback(KAGGLE_USERNAME, KAGGLE_MODEL))
    log(f"  Kaggle upload: every checkpoint → {KAGGLE_USERNAME}/{KAGGLE_MODEL}")

# ============================================================
# TRAIN
# ============================================================
log("\n" + "=" * 60)
log("  TRAINING — Full fine-tune, 8× H100, DeepSpeed ZeRO-3")
log(f"  {len(train_data)} samples | lr=2e-5 | 1 epoch")
log("=" * 60)

# Build config — handle TRL version differences
sft_kwargs = dict(
    output_dir="sft-full-8xh100",
    deepspeed=ds_config_path,
    per_device_train_batch_size=2,       # per GPU: 8 × 2 = 16
    gradient_accumulation_steps=2,        # effective batch = 32
    num_train_epochs=1,
    learning_rate=2e-5,
    bf16=True,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=1,
    logging_steps=1,
    eval_strategy="no",
    max_grad_norm=1.0,
    report_to=REPORT_TO,
    run_name="full-ft-8xh100-v4",
    dataset_text_field="text",
    packing=True,
    ddp_timeout=7200,
)

# Version-safe: add params only if supported
sig = inspect.signature(SFTConfig.__init__)
for param, val in [("save_only_model", True), ("max_seq_length", MAX_SEQ_LEN)]:
    if param in sig.parameters:
        sft_kwargs[param] = val

trainer_extra = {}
if "max_seq_length" not in sft_kwargs:
    trainer_extra["max_seq_length"] = MAX_SEQ_LEN

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    args=SFTConfig(**sft_kwargs),
    processing_class=tokenizer,
    callbacks=callbacks,
    **trainer_extra,
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

# Upload final to Kaggle
if IS_MAIN and KAGGLE_USERNAME:
    log("\n  Uploading final model to Kaggle...")
    try:
        import kagglehub
        handle = f"{KAGGLE_USERNAME}/{KAGGLE_MODEL}/transformers/default"
        kagglehub.model_upload(
            handle, save_dir,
            version_notes=f"FINAL | step={final_step} | loss={final_loss} | {len(train_data)} samples",
            license_name="Apache 2.0",
        )
        log(f"  ✅ Final model uploaded")
    except Exception as e:
        log(f"  ⚠️ Final upload failed: {e}")

log("\n" + "=" * 60)
log("  COMPLETE")
log(f"  Time: {train_time/3600:.1f}h | Loss: {final_loss}")
log(f"  Model: {save_dir}/")
log("=" * 60)

if WANDB_KEY:
    try:
        wandb.finish()
    except Exception:
        pass
