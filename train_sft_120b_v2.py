"""
AIMO3: SFT on GPT-OSS-120B (MoE, 4-bit QLoRA)
================================================
Hardware: 1× NVIDIA H200 (141GB)
Model: GPT-OSS-120B via unsloth/gpt-oss-120b-unsloth-bnb-4bit
Dataset: 130K Nemotron Math v2 gold solutions

Checkpointing strategy:
- Every 500 steps: full merge → 4bit (~65GB) → upload to Kaggle → delete local
- No LoRA uploads (useless without merge for MoE models)

Usage: python3 train_sft_120b.py
"""

# ============================================================
# IMPORTS — unsloth MUST be first
# ============================================================
import unsloth  # noqa: F401  — must be before transformers

import os, time, gc, glob, shutil
from dotenv import load_dotenv
load_dotenv()

KAGGLE_USERNAME      = os.environ["KAGGLE_USERNAME"]
DATA_DIR             = os.environ.get("DATA_DIR", "./data/nemotron-math-v2")
MERGE_UPLOAD_EVERY   = 500    # full merged 4bit upload every N steps

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

        try:
            import weave as _weave
            _weave.init("aimo3-sft-120b")
            weave = _weave
            print("Weave: enabled")
        except ImportError:
            print("Weave: not installed — pip install weave")
        except Exception as e:
            print(f"Weave: init failed ({e})")

    except Exception as e:
        REPORT_TO = "none"
        print(f"W&B: failed ({e})")
else:
    REPORT_TO = "none"
    print("W&B: disabled")

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
    fast_inference=False,
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

print(f"  Sample (first 300): {sft_data[0]['text'][:300]}")
print(f"  Sample (last  200): {sft_data[0]['text'][-200:]}")
print(f"  Has boxed: {'boxed' in sft_data[0]['text']}")

eval_data  = sft_data.select(range(0, min(2000, len(sft_data))))
train_data = sft_data.select(range(min(2000, len(sft_data)), len(sft_data)))
print(f"  Train: {len(train_data)} | Eval: {len(eval_data)}\n")


# ============================================================
# CHECKPOINT CALLBACK — merged 4bit only
# ============================================================

class CheckpointCallback(TrainerCallback):
    def __init__(self):
        self.last_merge_step = 0

    def _get_loss(self, state):
        return state.log_history[-1].get('loss', '?') if state.log_history else '?'

    def _merge_and_upload(self, state):
        """Merge LoRA into base, save as 4bit (~65GB), upload, delete local."""
        step = state.global_step
        loss = self._get_loss(state)

        print(f"\n  [Step {step}] loss={loss} — starting full 4bit merge...")
        t_merge = time.time()

        merged_dir = f"merged-4bit-step-{step}"
        try:
            model.save_pretrained_merged(
                merged_dir,
                tokenizer,
                save_method="merged_4bit",
            )
            merge_mins = (time.time() - t_merge) / 60
            print(f"  ✅ Merge done in {merge_mins:.1f} mins")

            # write readme
            readme = (
                f"# GPT-OSS-120B AIMO3 SFT — Merged 4bit Checkpoint\n\n"
                f"| Key | Value |\n"
                f"|-----|-------|\n"
                f"| Step | {step} |\n"
                f"| Loss | {loss} |\n"
                f"| Format | Merged 4bit (~65GB) |\n"
                f"| Base | unsloth/gpt-oss-120b-unsloth-bnb-4bit |\n"
                f"| LoRA rank | r=32 |\n"
                f"| Dataset | Nemotron Math v2 ({len(train_data)} samples) |\n"
                f"| LR | 2e-5 cosine |\n"
                f"| Merge time | {merge_mins:.1f} mins |\n"
            )
            with open(os.path.join(merged_dir, "README.md"), "w") as f:
                f.write(readme)

            # upload
            import kagglehub
            handle = f"{KAGGLE_USERNAME}/gpt-oss-120b-aimo3-sft-merged/transformers/default"
            print(f"  Uploading ~65GB → {handle} ...")
            t_upload = time.time()
            kagglehub.model_upload(
                handle, merged_dir,
                version_notes=f"Merged 4bit | step={step} | loss={loss} | merge={merge_mins:.1f}min",
                license_name="Apache 2.0",
            )
            upload_mins = (time.time() - t_upload) / 60
            print(f"  ✅ Upload done in {upload_mins:.1f} mins")

        except Exception as e:
            print(f"  ⚠️  Merge/upload failed: {e}")

        finally:
            # always free disk regardless of success/failure
            if os.path.exists(merged_dir):
                shutil.rmtree(merged_dir)
                print(f"  🗑  Freed disk: deleted {merged_dir}")

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step > 0 and step % MERGE_UPLOAD_EVERY == 0 and step != self.last_merge_step:
            self.last_merge_step = step
            self._merge_and_upload(state)

    def on_train_end(self, args, state, control, **kwargs):
        # always do a final merge at end of training
        self._merge_and_upload(state)


# ============================================================
# TRAIN
# ============================================================

print("=" * 60)
print(f"  TRAINING SFT on GPT-OSS-120B")
print(f"  {len(train_data)} samples | 1 epoch | lr=2e-5")
print(f"  Merged 4bit upload every {MERGE_UPLOAD_EVERY} steps (~65GB)")
print("=" * 60)

FastLanguageModel.for_training(model)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    args=SFTConfig(
        output_dir="sft-output-120b",
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        learning_rate=2e-5,
        bf16=True,
        max_seq_length=2048,
        lr_scheduler_type="cosine",
        warmup_ratio=0,
        save_steps=500,           # trainer saves every 500 to match merge cadence
        save_total_limit=2,       # keep last 2 trainer checkpoints for resume only
        logging_steps=1,
        eval_strategy="no",
        max_grad_norm=0.5,
        report_to=REPORT_TO,
        run_name="sft-120b",
        dataset_text_field="text",
        packing=True,
    ),
    processing_class=tokenizer,
    callbacks=[CheckpointCallback()],
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
final_step = trainer.state.global_step
final_loss = trainer.state.log_history[-1].get('loss', '?') if trainer.state.log_history else '?'
print(f"\n  SFT done in {train_time/3600:.1f}h | step={final_step} | loss={final_loss}")

del trainer
gc.collect()
torch.cuda.empty_cache()

# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 60)
print("  COMPLETE")
print("=" * 60)
print(f"  Time:        {train_time/3600:.1f}h")
print(f"  Final step:  {final_step}")
print(f"  Final loss:  {final_loss}")
print(f"  Merged 4bit: {KAGGLE_USERNAME}/gpt-oss-120b-aimo3-sft-merged")
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
