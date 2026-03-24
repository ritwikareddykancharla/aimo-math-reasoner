"""
AIMO3: SFT on GPT-OSS-120B (MoE, 4-bit QLoRA)
================================================
Hardware: 1× NVIDIA H200 (141GB)
Model: GPT-OSS-120B via unsloth/gpt-oss-120b-unsloth-bnb-4bit
Dataset: 130K Nemotron Math v2 gold solutions
GPU: 62GB model + 88GB free for training

Usage: python3 train_sft_120b.py
"""

# ============================================================
# IMPORTS — unsloth MUST be first
# ============================================================
import unsloth  # noqa: F401  — must be before transformers

import os, sys, re, json, time, gc, glob, shutil
from dotenv import load_dotenv
load_dotenv()

KAGGLE_USERNAME = os.environ["KAGGLE_USERNAME"]
DATA_DIR = os.environ.get("DATA_DIR", "./data/nemotron-math-v2")
MAX_LOCAL_CHECKPOINTS = 3   # keep only last N checkpoint dirs on disk

# ============================================================
# W&B + WEAVE
# ============================================================
WANDB_KEY = os.environ.get("WANDB_API_KEY", "")
wandb = None
weave = None

if WANDB_KEY:
    try:
        import wandb
        wandb.login(key=WANDB_KEY)
        os.environ["WANDB_PROJECT"] = "aimo3-sft-120b"
        REPORT_TO = "wandb"
        print("W&B: enabled")

        # Enable Weave for LLM call tracing
        try:
            import weave as _weave
            _weave.init("aimo3-sft-120b")
            weave = _weave
            print("Weave: enabled (tracing active)")
        except ImportError:
            print("Weave: not installed — run `pip install weave` to enable tracing")
        except Exception as e:
            print(f"Weave: init failed ({e})")

    except Exception as e:
        REPORT_TO = "none"
        print(f"W&B: failed ({e})")
else:
    REPORT_TO = "none"
    print("W&B: disabled (set WANDB_API_KEY to enable)")

# ============================================================
# MODEL IMPORTS
# ============================================================
from datasets import load_from_disk
from transformers import set_seed, TrainerCallback
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
import torch

set_seed(42)

# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 60)
print("  Loading GPT-OSS-120B (4-bit QLoRA)")
print("=" * 60)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gpt-oss-120b-unsloth-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
    fast_inference=False,   # No vLLM — just training
    max_lora_rank=32,
)

print(f"  GPU used: {torch.cuda.memory_allocated()/1e9:.1f}GB")

model = FastLanguageModel.get_peft_model(
    model, r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    use_gradient_checkpointing="unsloth",
)

free_gb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  GPU free:   {free_gb:.1f}GB")
print(f"  Trainable:  {trainable/1e6:.0f}M params")
print("  Model ready.\n")

# ============================================================
# LOAD & FORMAT DATASET
# ============================================================

print("=" * 60)
print("  Loading dataset")
print("=" * 60)

ds = load_from_disk(DATA_DIR)
ds = ds.shuffle(seed=42)
print(f"  Total: {len(ds)} rows")

def format_for_sft(example):
    """Format gold solution with GPT-OSS chat template."""
    msgs = example['messages']
    clean_msgs = [
        {"role": "user",      "content": msgs[0]['content']},
        {"role": "assistant", "content": msgs[1]['content']},
    ]
    text = tokenizer.apply_chat_template(
        clean_msgs, tokenize=False, add_generation_prompt=False,
    )
    return {"text": text}

print("  Formatting...")
t0 = time.time()
sft_data = ds.map(format_for_sft, num_proc=8, remove_columns=ds.column_names)
print(f"  Formatted {len(sft_data)} samples in {time.time()-t0:.0f}s")

# Sanity checks
print(f"  Sample (first 300): {sft_data[0]['text'][:300]}")
print(f"  Sample (last  200): {sft_data[0]['text'][-200:]}")
print(f"  Has boxed: {'boxed' in sft_data[0]['text']}")

# Split: first 2000 = eval, rest = train
eval_data  = sft_data.select(range(0, min(2000, len(sft_data))))
train_data = sft_data.select(range(min(2000, len(sft_data)), len(sft_data)))
print(f"  Train: {len(train_data)} | Eval: {len(eval_data)}\n")


# ============================================================
# CHECKPOINT CALLBACK — upload every 100 steps, keep last 3
# ============================================================

class CheckpointCallback(TrainerCallback):
    def __init__(self, upload_every=100):
        self.upload_every = upload_every
        self.last_uploaded_step = 0
        self.local_checkpoints = []   # ordered list of saved dirs

    # ----------------------------------------------------------
    def _cleanup_old_checkpoints(self):
        """Delete oldest dirs, keeping only MAX_LOCAL_CHECKPOINTS."""
        while len(self.local_checkpoints) > MAX_LOCAL_CHECKPOINTS:
            oldest = self.local_checkpoints.pop(0)
            if os.path.exists(oldest):
                shutil.rmtree(oldest)
                print(f"    🗑  Deleted old checkpoint: {oldest}")

    # ----------------------------------------------------------
    def _upload_to_kaggle(self, ckpt_dir, step, loss):
        """Upload LoRA adapter to Kaggle model registry."""
        try:
            import kagglehub
            readme = (
                f"# GPT-OSS-120B AIMO3 SFT LoRA\n\n"
                f"| Key | Value |\n"
                f"|-----|-------|\n"
                f"| Step | {step} |\n"
                f"| Loss | {loss} |\n"
                f"| Base model | unsloth/gpt-oss-120b-unsloth-bnb-4bit |\n"
                f"| LoRA rank | r=32 |\n"
                f"| Dataset | Nemotron Math v2 ({len(train_data)} samples) |\n"
                f"| LR | 2e-5 cosine, warmup 3% |\n"
            )
            with open(os.path.join(ckpt_dir, "README.md"), "w") as f:
                f.write(readme)

            handle = f"{KAGGLE_USERNAME}/gpt-oss-120b-aimo3-sft-lora/transformers/default"
            kagglehub.model_upload(
                handle, ckpt_dir,
                version_notes=f"LoRA step {step} | loss={loss}",
                license_name="Apache 2.0",
            )
            print(f"    ✅ Kaggle upload done: {handle}  (step={step})")
        except Exception as e:
            print(f"    ⚠️  Kaggle upload failed: {e}")

    # ----------------------------------------------------------
    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step <= 0 or step % self.upload_every != 0 or step == self.last_uploaded_step:
            return

        self.last_uploaded_step = step
        loss = state.log_history[-1].get('loss', '?') if state.log_history else '?'
        print(f"\n  [Step {step}] loss={loss} — saving & uploading...")

        ckpt = f"sft-output-120b/checkpoint-{step}"
        os.makedirs(ckpt, exist_ok=True)
        model.save_pretrained(ckpt)
        state.save_to_json(os.path.join(ckpt, "trainer_state.json"))
        print(f"    💾 Saved locally: {ckpt}")

        self.local_checkpoints.append(ckpt)
        self._cleanup_old_checkpoints()
        self._upload_to_kaggle(ckpt, step, loss)

    # ----------------------------------------------------------
    def on_train_end(self, args, state, control, **kwargs):
        step = state.global_step
        loss = state.log_history[-1].get('loss', '?') if state.log_history else '?'
        print(f"\n  [Final] Step {step} | loss={loss} — saving final LoRA...")

        ckpt = "checkpoint-120b-sft-final"
        os.makedirs(ckpt, exist_ok=True)
        model.save_pretrained(ckpt)
        state.save_to_json(os.path.join(ckpt, "trainer_state.json"))
        print(f"    💾 Final LoRA saved: {ckpt}")

        self._upload_to_kaggle(ckpt, f"{step} (FINAL)", loss)


# ============================================================
# TRAIN
# ============================================================

print("=" * 60)
print(f"  TRAINING SFT on GPT-OSS-120B")
print(f"  {len(train_data)} samples | 1 epoch | lr=2e-5")
print("=" * 60)

FastLanguageModel.for_training(model)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    args=SFTConfig(
        output_dir="sft-output-120b",
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,        # effective batch = 32
        num_train_epochs=1,
        learning_rate=2e-5,                   # ← was 1e-4 (too high for 120B)
        bf16=True,
        max_seq_length=2048,
        lr_scheduler_type="constant",           # ← was constant
        warmup_ratio=0,                    # ← was 0
        save_steps=100,                       # trainer auto-saves every 100 steps
        save_total_limit=3,                   # trainer keeps last 3 auto-saves
        logging_steps=1,
        eval_strategy="no",
        max_grad_norm=0.5,                    # ← was 1.0, slightly tighter
        report_to=REPORT_TO,
        run_name="sft-120b",
        dataset_text_field="text",
        packing=True,
    ),
    processing_class=tokenizer,
    callbacks=[CheckpointCallback(upload_every=100)],
)

print("\n  Starting training...")
t0 = time.time()
checkpoints = sorted(glob.glob("sft-output-120b/checkpoint-*"))
if checkpoints:
    print(f"  Resuming from {checkpoints[-1]}...")
    trainer.train(resume_from_checkpoint=checkpoints[-1])
else:
    print("  Starting fresh...")
    trainer.train()

train_time = time.time() - t0

# Capture final state before freeing trainer
final_step = trainer.state.global_step
final_loss = trainer.state.log_history[-1].get('loss', '?') if trainer.state.log_history else '?'
print(f"\n  SFT done in {train_time/3600:.1f}h  |  final step={final_step}  |  final loss={final_loss}")

del trainer
gc.collect()
torch.cuda.empty_cache()

# ============================================================
# MERGE + UPLOAD  (after trainer freed memory)
# ============================================================

print("\n  Merging LoRA into full model (16-bit)...")
try:
    merged_dir = "merged-120b-sft-final"
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")

    readme = f"""# GPT-OSS-120B AIMO3 — SFT Fine-tuned

## Training summary

| Key | Value |
|-----|-------|
| Base model | GPT-OSS-120B (MoE, 128 experts) |
| Final step | {final_step} |
| Final loss | {final_loss} |
| Dataset | Nemotron Math v2 — {len(train_data)} gold solutions |
| LoRA rank | r=32 (attention + MoE expert layers) |
| Learning rate | 2e-5 cosine, 3% warmup |
| Batch size | 2 × 8 grad accum = 16 effective |
| Sequence length | 2048 |
| Training time | {train_time/3600:.1f}h on H200 |

## Dataset
- [ritwikakancharla/nemotron-math-v2-filtered-high](https://www.kaggle.com/datasets/ritwikakancharla/nemotron-math-v2-filtered-high)
- 130K competition math problems (AIMO3 format, int answers 0–999999)

## Usage
```python
model_path = '/kaggle/input/models/{KAGGLE_USERNAME}/gpt-oss-120b-aimo3-sft/transformers/default'
```
"""
    with open(os.path.join(merged_dir, "README.md"), "w") as f:
        f.write(readme)

    import kagglehub
    handle = f"{KAGGLE_USERNAME}/gpt-oss-120b-aimo3-sft/transformers/default"
    print(f"  Uploading merged model → {handle}")
    kagglehub.model_upload(
        handle, merged_dir,
        version_notes=f"SFT final | step={final_step} | loss={final_loss} | {train_time/3600:.1f}h",
        license_name="Apache 2.0",
    )
    print("  ✅ Merged model uploaded!")

except Exception as e:
    print(f"  ⚠️  Merge/upload failed: {e}")
    print("  LoRA still available at: checkpoint-120b-sft-final/")

# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 60)
print("  COMPLETE")
print("=" * 60)
print(f"  Time:       {train_time/3600:.1f}h")
print(f"  Final step: {final_step}")
print(f"  Final loss: {final_loss}")
print(f"  LoRA:       {KAGGLE_USERNAME}/gpt-oss-120b-aimo3-sft-lora")
print(f"  Merged:     {KAGGLE_USERNAME}/gpt-oss-120b-aimo3-sft")
print("=" * 60)

if wandb:
    try:
        wandb.finish()
    except Exception:
        pass

if weave:
    try:
        weave.finish()
    except Exception:
        pass
