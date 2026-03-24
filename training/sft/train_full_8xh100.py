"""
AIMO3: Full Fine-Tuning on 7× H100 (DeepSpeed ZeRO-3)
======================================================
Full parameter training. DeepSpeed ZeRO-3 shards across 7 GPUs.
GPU 7 reserved for 4-bit quantization + Kaggle upload after each checkpoint.

Dataset: v4 (79K hard problems, upsampled to ~193K effective)
Checkpoints: Every 200 steps → quantized to 4-bit (~60GB) → uploaded to Kaggle
             Only last 3 Kaggle versions kept. 1 local bf16 checkpoint.

Launch:
    export PATH=$PATH:/home/ssm-user/.local/bin
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 deepspeed --num_gpus=7 training/sft/train_full_8xh100.py

    # GPU 7 is automatically used for quantization by the callback.

Env vars from .env: HF_TOKEN, KAGGLE_USERNAME, KAGGLE_API_TOKEN, WANDB_API_KEY
"""

import os, sys, time, gc, json, shutil, glob, threading
from dotenv import load_dotenv
load_dotenv()

KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME", "")
KAGGLE_MODEL    = os.environ.get("KAGGLE_MODEL_NAME", "gpt-oss-120b-aimo3-sft-v4")
DATA_DIR = os.environ.get("DATA_DIR", "./data/nemotron-sft-v4/hf_dataset")
QUANT_GPU = 7   # Reserved GPU for quantization (not used by DeepSpeed)

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
    BitsAndBytesConfig,
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
# KAGGLE CHECKPOINT CALLBACK (4-bit quantize on GPU 7 + upload)
# ============================================================

class QuantizeAndUploadCallback(TrainerCallback):
    """After each checkpoint:
    1. Load saved model in 4-bit on reserved GPU (not used by training)
    2. Save 4-bit model to temp dir (~60GB)
    3. Upload to Kaggle in background
    4. Clean temp dir
    """

    def __init__(self, kaggle_user, model_name, quant_gpu=7, keep_versions=3):
        self.kaggle_user = kaggle_user
        self.model_name = model_name
        self.quant_gpu = quant_gpu
        self.keep_versions = keep_versions
        self.bg_thread = None
        self.version_count = 0

    def on_save(self, args, state, control, **kwargs):
        if not IS_MAIN or not self.kaggle_user:
            return

        step = state.global_step
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{step}")

        if not os.path.isdir(checkpoint_dir):
            return

        # Wait for any previous quantize+upload to finish
        if self.bg_thread and self.bg_thread.is_alive():
            log(f"  [Q+U] Waiting for previous upload to finish...")
            self.bg_thread.join(timeout=1800)

        # Run quantize + upload in background (uses reserved GPU, not training GPUs)
        self.bg_thread = threading.Thread(
            target=self._quantize_and_upload,
            args=(checkpoint_dir, step, state),
            daemon=True,
        )
        self.bg_thread.start()
        log(f"  [Q+U] Started quantize+upload for step {step} on GPU {self.quant_gpu}")

    def _quantize_and_upload(self, checkpoint_dir, step, state):
        temp_dir = f"_temp_4bit_step{step}"
        try:
            loss = state.log_history[-1].get("loss", "?") if state.log_history else "?"
            hf_token = os.environ.get("HF_TOKEN", "")

            # ── Step 1: Load on reserved GPU in 4-bit ──
            log(f"  [Q+U] Loading checkpoint in 4-bit on cuda:{self.quant_gpu}...")
            t0 = time.time()

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

            quant_model = AutoModelForCausalLM.from_pretrained(
                checkpoint_dir,
                quantization_config=bnb_config,
                device_map={"": f"cuda:{self.quant_gpu}"},
                torch_dtype=torch.bfloat16,
                token=hf_token,
                trust_remote_code=True,
            )

            quant_tokenizer = AutoTokenizer.from_pretrained(
                checkpoint_dir,
                token=hf_token,
                trust_remote_code=True,
            )

            load_time = time.time() - t0
            log(f"  [Q+U] Loaded 4-bit in {load_time:.0f}s")

            # ── Step 2: Save 4-bit model ──
            log(f"  [Q+U] Saving 4-bit to {temp_dir}...")
            quant_model.save_pretrained(temp_dir)
            quant_tokenizer.save_pretrained(temp_dir)

            # Calculate size
            total_size = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, _, fns in os.walk(temp_dir)
                for f in fns
            ) / 1e9
            log(f"  [Q+U] Saved: {total_size:.1f}GB")

            # Free GPU memory
            del quant_model
            gc.collect()
            torch.cuda.empty_cache()

            # ── Step 3: Upload to Kaggle ──
            log(f"  [Q+U] Uploading step-{step} to Kaggle ({total_size:.1f}GB)...")
            t0 = time.time()

            import kagglehub
            handle = f"{self.kaggle_user}/{self.model_name}/transformers/default"
            kagglehub.model_upload(
                handle, temp_dir,
                version_notes=f"step-{step} | loss={loss} | 4-bit NF4 | {total_size:.1f}GB",
                license_name="Apache 2.0",
            )

            self.version_count += 1
            upload_mins = (time.time() - t0) / 60
            log(f"  [Q+U] ✅ Uploaded step-{step} in {upload_mins:.1f}min "
                f"(v{self.version_count}, {total_size:.1f}GB)")

        except Exception as e:
            log(f"  [Q+U] ⚠️ Failed for step {step}: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Always clean temp dir
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                log(f"  [Q+U] Cleaned {temp_dir}")

    def on_train_end(self, args, state, control, **kwargs):
        if self.bg_thread and self.bg_thread.is_alive():
            log("  [Q+U] Waiting for final upload to complete...")
            self.bg_thread.join(timeout=3600)


# ============================================================
# LOAD MODEL
# ============================================================
log("\n" + "=" * 60)
log("  Loading model (full bf16, no quantization)")
log(f"  Training on GPUs 0-6 | GPU {QUANT_GPU} reserved for quantization")
log("=" * 60)

MODEL_NAME = "openai/gpt-oss-120b"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
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
    callbacks.append(QuantizeAndUploadCallback(
        kaggle_user=KAGGLE_USERNAME,
        model_name=KAGGLE_MODEL,
        quant_gpu=QUANT_GPU,
        keep_versions=3,
    ))
    log(f"  Kaggle: every checkpoint → 4-bit on GPU {QUANT_GPU} → upload")
    log(f"  Target: ~60GB per version, 3 versions max on Kaggle")

# ============================================================
# TRAIN
# ============================================================
log("\n" + "=" * 60)
log("  TRAINING — Full fine-tune, 7× H100, DeepSpeed ZeRO-3")
log(f"  {len(train_data)} samples | lr=2e-5 | 1 epoch")
log(f"  Checkpoints: every 200 steps → quantize → Kaggle")
log("=" * 60)

# Build SFTConfig — handle version differences
sft_kwargs = dict(
    output_dir="sft-full-8xh100",
    deepspeed=ds_config_path,
    per_device_train_batch_size=2,       # per GPU: 7 GPUs × 2 = 14
    gradient_accumulation_steps=2,        # effective batch = 28
    num_train_epochs=1,
    learning_rate=2e-5,
    bf16=True,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=1,                   # 1 local checkpoint (disk-safe)
    logging_steps=1,
    eval_strategy="no",
    max_grad_norm=1.0,
    report_to=REPORT_TO,
    run_name="full-ft-7xh100-v4",
    dataset_text_field="text",
    packing=True,
    ddp_timeout=7200,
)

# save_only_model: skip optimizer states (~1TB saved)
import inspect
if "save_only_model" in inspect.signature(SFTConfig.__init__).parameters:
    sft_kwargs["save_only_model"] = True

# max_seq_length: some TRL versions have it in SFTConfig, others in SFTTrainer
if "max_seq_length" in inspect.signature(SFTConfig.__init__).parameters:
    sft_kwargs["max_seq_length"] = MAX_SEQ_LEN

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
# SAVE FINAL MODEL + QUANTIZE + UPLOAD
# ============================================================
log("\n  Saving final model (full bf16)...")
save_dir = "model-full-ft-final"
trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)
log(f"  Saved to {save_dir}")

# Final 4-bit quantize + upload
if IS_MAIN and KAGGLE_USERNAME:
    log(f"\n  Final quantize (4-bit on GPU {QUANT_GPU}) + upload...")
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        final_4bit = AutoModelForCausalLM.from_pretrained(
            save_dir,
            quantization_config=bnb_config,
            device_map={"": f"cuda:{QUANT_GPU}"},
            torch_dtype=torch.bfloat16,
            token=os.environ.get("HF_TOKEN"),
            trust_remote_code=True,
        )
        final_tok = AutoTokenizer.from_pretrained(save_dir, trust_remote_code=True)

        final_4bit_dir = "model-full-ft-final-4bit"
        final_4bit.save_pretrained(final_4bit_dir)
        final_tok.save_pretrained(final_4bit_dir)

        del final_4bit
        gc.collect()
        torch.cuda.empty_cache()

        import kagglehub
        handle = f"{KAGGLE_USERNAME}/{KAGGLE_MODEL}/transformers/default"
        kagglehub.model_upload(
            handle, final_4bit_dir,
            version_notes=f"FINAL | step={final_step} | loss={final_loss} | 4-bit NF4",
            license_name="Apache 2.0",
        )
        log(f"  ✅ Final 4-bit model uploaded to Kaggle")
        shutil.rmtree(final_4bit_dir)
    except Exception as e:
        log(f"  ⚠️  Final upload failed: {e}")

log("\n" + "=" * 60)
log("  COMPLETE")
log(f"  Time: {train_time/3600:.1f}h")
log(f"  Loss: {final_loss}")
log(f"  Model: {save_dir}/")
log("=" * 60)

if WANDB_KEY:
    try:
        wandb.finish()
    except Exception:
        pass
