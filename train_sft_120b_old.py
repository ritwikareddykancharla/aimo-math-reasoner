"""
AIMO3: SFT on GPT-OSS-120B (MoE, 4-bit QLoRA)
================================================
Hardware: 1× NVIDIA H200 (141GB)
Model: GPT-OSS-120B via unsloth/gpt-oss-120b-unsloth-bnb-4bit
Dataset: 130K Nemotron Math v2 gold solutions
GPU: 62GB model + 88GB free for training

Usage: python3 train_sft_120b.py
"""

import os, sys, re, json, time, gc, glob
from dotenv import load_dotenv
load_dotenv()

KAGGLE_USERNAME = os.environ["KAGGLE_USERNAME"]
DATA_DIR = os.environ.get("DATA_DIR", "./data/nemotron-math-v2")

# W&B
WANDB_KEY = os.environ.get("WANDB_API_KEY", "")
wandb = None
if WANDB_KEY:
    try:
        import wandb
        wandb.login(key=WANDB_KEY)
        os.environ["WANDB_PROJECT"] = "aimo3-sft-120b"
        REPORT_TO = "wandb"
        print("W&B: enabled")
    except:
        REPORT_TO = "none"
else:
    REPORT_TO = "none"
    print("W&B: disabled")

from datasets import load_from_disk
from transformers import set_seed, TrainerCallback
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

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

import torch
print(f"  GPU used: {torch.cuda.memory_allocated()/1e9:.1f}GB")

model = FastLanguageModel.get_peft_model(
    model, r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32, lora_dropout=0,
    use_gradient_checkpointing="unsloth",
)

free_gb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  GPU free: {free_gb:.1f}GB")
print(f"  Trainable: {trainable/1e6:.0f}M params")
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
        {"role": "user", "content": msgs[0]['content']},
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

# Verify
print(f"  Sample (first 300): {sft_data[0]['text'][:300]}")
print(f"  Sample (last 200): {sft_data[0]['text'][-200:]}")
print(f"  Has boxed: {'boxed' in sft_data[0]['text']}")

# Split
eval_data = sft_data.select(range(0, min(2000, len(sft_data))))
train_data = sft_data.select(range(min(2000, len(sft_data)), len(sft_data)))
print(f"  Train: {len(train_data)} | Eval: {len(eval_data)}\n")


# ============================================================
# CHECKPOINT CALLBACK
# ============================================================

class CheckpointCallback(TrainerCallback):
    def __init__(self, upload_every=300):
        self.upload_every = upload_every
        self.last_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step > 0 and step % self.upload_every == 0 and step != self.last_step:
            self.last_step = step
            loss = state.log_history[-1].get('loss', '?') if state.log_history else '?'
            print(f"\n  [Step {step}] loss={loss}")

            # Save LoRA + trainer state
            ckpt = f"sft-output-120b/checkpoint-{step}"
            os.makedirs(ckpt, exist_ok=True)
            model.save_pretrained(ckpt)
            state.save_to_json(os.path.join(ckpt, "trainer_state.json"))
            print(f"    Saved: {ckpt}")

            # Upload LoRA to Kaggle
            try:
                import kagglehub
                readme = f"# GPT-OSS-120B AIMO3 SFT LoRA\n\nStep: {step} | Loss: {loss}\n"
                with open(os.path.join(ckpt, "README.md"), "w") as f:
                    f.write(readme)
                handle = f"{KAGGLE_USERNAME}/gpt-oss-120b-aimo3-sft-lora/transformers/default"
                kagglehub.model_upload(handle, ckpt,
                    version_notes=f"LoRA step {step} | loss={loss}",
                    license_name="Apache 2.0")
                print(f"    Uploaded: {handle}")
            except Exception as e:
                print(f"    Upload failed: {e}")

    def on_train_end(self, args, state, control, **kwargs):
        step = state.global_step
        loss = state.log_history[-1].get('loss', '?') if state.log_history else '?'
        print(f"\n  [Final] Step {step}, saving...")

        # Save final LoRA
        ckpt = "checkpoint-120b-sft-final"
        model.save_pretrained(ckpt)
        state.save_to_json(os.path.join(ckpt, "trainer_state.json"))

        # Upload final LoRA
        try:
            import kagglehub
            readme = f"# GPT-OSS-120B AIMO3 SFT — Final\n\nStep: {step} | Loss: {loss}\n\nApply this LoRA on top of gpt-oss-120b for AIMO3 inference.\n"
            with open(os.path.join(ckpt, "README.md"), "w") as f:
                f.write(readme)
            handle = f"{KAGGLE_USERNAME}/gpt-oss-120b-aimo3-sft-lora/transformers/default"
            kagglehub.model_upload(handle, ckpt,
                version_notes=f"FINAL step {step} | loss={loss}",
                license_name="Apache 2.0")
            print(f"    Final LoRA uploaded: {handle}")
        except Exception as e:
            print(f"    Upload failed: {e}")


# ============================================================
# TRAIN
# ============================================================

print("=" * 60)
print(f"  TRAINING SFT on GPT-OSS-120B")
print(f"  {len(train_data)} samples, 1 epoch")
print("=" * 60)

FastLanguageModel.for_training(model)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    args=SFTConfig(
        output_dir="sft-output-120b",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=1e-4,
        bf16=True,
        max_seq_length=2048,
        lr_scheduler_type="constant",
        warmup_ratio=0,
        save_steps=300,
        logging_steps=1,
        eval_strategy="no",
        max_grad_norm=1.0,
        report_to=REPORT_TO,
        run_name="sft-120b",
        dataset_text_field="text",
        packing=True,
    ),
    processing_class=tokenizer,
    callbacks=[CheckpointCallback(upload_every=300)],
)

# Resume if checkpoint exists
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
print(f"\n  SFT done in {train_time/3600:.1f}h")

# Cleanup
del trainer
gc.collect()
torch.cuda.empty_cache()

# ============================================================
# MERGE + UPLOAD (after trainer freed memory)
# ============================================================

print("\n  Merging LoRA into full model...")
try:
    merged_dir = "merged-120b-sft-final"
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")

    readme = f"""# GPT-OSS-120B AIMO3 — SFT Fine-tuned

## Model
- **Base**: GPT-OSS-120B (MoE, 128 experts)
- **Training**: SFT on {len(train_data)} Nemotron Math v2 gold solutions
- **LoRA**: r=32, targeting attention + MoE expert layers
- **Training time**: {train_time/3600:.1f}h on H200

## Dataset
- [ritwikakancharla/nemotron-math-v2-filtered-high](https://www.kaggle.com/datasets/ritwikakancharla/nemotron-math-v2-filtered-high)
- 130K competition math problems (AIMO3 format, int answers 0-99999)

## Usage
Replace model_path in the AIMO3 submission notebook:
```python
model_path = '/kaggle/input/models/ritwikakancharla/gpt-oss-120b-aimo3-sft/transformers/default'
```
"""
    with open(os.path.join(merged_dir, "README.md"), "w") as f:
        f.write(readme)

    import kagglehub
    handle = f"{KAGGLE_USERNAME}/gpt-oss-120b-aimo3-sft/transformers/default"
    print(f"  Uploading merged model: {handle}")
    kagglehub.model_upload(handle, merged_dir,
        version_notes=f"SFT final | {train_time/3600:.1f}h",
        license_name="Apache 2.0")
    print("  Merged model uploaded!")
except Exception as e:
    print(f"  Merge/upload failed: {e}")
    print("  LoRA saved at: checkpoint-120b-sft-final/")

print("\n" + "=" * 60)
print("  COMPLETE")
print("=" * 60)
print(f"  Time: {train_time/3600:.1f}h")
print(f"  LoRA: {KAGGLE_USERNAME}/gpt-oss-120b-aimo3-sft-lora")
print(f"  Merged: {KAGGLE_USERNAME}/gpt-oss-120b-aimo3-sft")
print("=" * 60)

if wandb:
    try:
        wandb.finish()
    except:
        pass
