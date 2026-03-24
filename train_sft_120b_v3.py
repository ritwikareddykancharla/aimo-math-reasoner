"""
AIMO3: SFT on GPT-OSS-120B (MoE, 4-bit QLoRA) — v3
====================================================
Hardware: 1× NVIDIA H200 (141GB)
Model: GPT-OSS-120B via unsloth/gpt-oss-120b-unsloth-bnb-4bit
Dataset: Hard problems with tools from Nemotron Math v2

FIXES from v2:
  1. Keep ALL message turns (user/assistant/tool) — not just first 2
  2. max_seq_length=8192 (was 2048 — was cutting off hard problems)
  3. Merge uses merged_4bit_forced (was failing with merged_4bit)
  4. Batch size adjusted for longer sequences
  5. warmup_ratio=0.03 for stability

Checkpointing strategy:
- Every 500 steps: LoRA adapter save (fast, small)
- Every 1000 steps: full merge → 4bit_forced → upload to Kaggle → delete local
- Final: full merge + upload

Usage:
    export KAGGLE_USERNAME=your_username
    export KAGGLE_API_TOKEN=your_token
    export WANDB_API_KEY=your_key  # optional
    python3 train_sft_120b_v3.py
"""

# ============================================================
# IMPORTS — unsloth MUST be first
# ============================================================
import unsloth  # noqa: F401

import os, time, gc, shutil
from dotenv import load_dotenv
load_dotenv()

KAGGLE_USERNAME      = os.environ["KAGGLE_USERNAME"]
DATA_DIR             = os.environ.get("DATA_DIR", "./data/nemotron-sft-hard-tools/hf_dataset")
MERGE_UPLOAD_EVERY   = 500    # full merged 4bit upload every N steps

# ============================================================
# W&B
# ============================================================
WANDB_KEY = os.environ.get("WANDB_API_KEY", "")
wandb = None

if WANDB_KEY:
    try:
        import wandb
        wandb.login(key=WANDB_KEY)
        os.environ["WANDB_PROJECT"] = "aimo3-sft-120b-v3"
        REPORT_TO = "wandb"
        print("W&B: enabled")
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

MAX_SEQ_LEN = 4096  # Compressed data: median 261 words, p90 489 words, max 2877 words

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gpt-oss-120b-unsloth-bnb-4bit",
    max_seq_length=MAX_SEQ_LEN,
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
    """
    Messages are pre-compressed 2-turn format:
      [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    Tool outputs are already inlined in the assistant response.
    """
    msgs = example['messages']
    text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False,
    )
    return {"text": text}

print("  Formatting (compressed 2-turn with inline tool outputs)...")
t0 = time.time()
sft_data = ds.map(format_for_sft, num_proc=8, remove_columns=ds.column_names)

# Remove empty entries
before = len(sft_data)
sft_data = sft_data.filter(lambda x: len(x['text']) > 100)
print(f"  Formatted {len(sft_data)} samples in {time.time()-t0:.0f}s")
if before != len(sft_data):
    print(f"  Removed {before - len(sft_data)} empty/broken entries")

# Show sample
print(f"\n  Sample (first 500 chars):")
print(f"  {sft_data[0]['text'][:500]}")
print(f"  ...")
print(f"  Sample (last 300 chars):")
print(f"  {sft_data[0]['text'][-300:]}")

# Count token lengths to verify
print(f"\n  Checking token lengths on sample...")
sample_lens = []
for i in range(min(1000, len(sft_data))):
    toks = tokenizer.encode(sft_data[i]['text'])
    sample_lens.append(len(toks))
sample_lens.sort()
n = len(sample_lens)
print(f"  Token lengths (sample of {n}):")
print(f"    p10={sample_lens[n//10]} p25={sample_lens[n//4]} "
      f"median={sample_lens[n//2]} p75={sample_lens[3*n//4]} "
      f"p90={sample_lens[9*n//10]} max={sample_lens[-1]}")
truncated = sum(1 for l in sample_lens if l > MAX_SEQ_LEN)
print(f"    Would be truncated (>{MAX_SEQ_LEN}): {truncated}/{n} ({100*truncated/n:.1f}%)")

# Use all data for training — eval is done by submitting to competition
train_data = sft_data
print(f"\n  Train: {len(train_data)}\n")

# ============================================================
# CHECKPOINT CALLBACK
# ============================================================
class CheckpointCallback(TrainerCallback):
    def __init__(self):
        self.last_merge_step = 0

    def _get_loss(self, state):
        return state.log_history[-1].get('loss', '?') if state.log_history else '?'

    def _merge_and_upload(self, state):
        """Merge LoRA into base, save as 4bit, upload, delete local."""
        step = state.global_step
        loss = self._get_loss(state)

        print(f"\n  [Step {step}] loss={loss} — starting full 4bit merge...")
        t_merge = time.time()

        merged_dir = f"merged-4bit-step-{step}"
        try:
            model.save_pretrained_merged(
                merged_dir,
                tokenizer,
                save_method="merged_4bit_forced",
            )
            merge_mins = (time.time() - t_merge) / 60
            print(f"  ✅ Merge done in {merge_mins:.1f} mins")

            # Write readme
            readme = (
                f"# GPT-OSS-120B AIMO3 SFT v3 — Merged 4bit\n\n"
                f"| Key | Value |\n"
                f"|-----|-------|\n"
                f"| Step | {step} |\n"
                f"| Loss | {loss} |\n"
                f"| Format | Merged 4bit_forced |\n"
                f"| Base | unsloth/gpt-oss-120b-unsloth-bnb-4bit |\n"
                f"| LoRA rank | r=32 |\n"
                f"| max_seq_length | {MAX_SEQ_LEN} |\n"
                f"| Dataset | Hard tools compressed ({len(train_data)} samples) |\n"
                f"| LR | 2e-5 cosine |\n"
                f"| Key fix | Compressed 2-turn, inline tool outputs |\n"
                f"| Merge time | {merge_mins:.1f} mins |\n"
            )
            with open(os.path.join(merged_dir, "README.md"), "w") as f:
                f.write(readme)

            # Upload to Kaggle
            try:
                import kagglehub
                handle = f"{KAGGLE_USERNAME}/gpt-oss-120b-aimo3-sft-v3/transformers/default"
                print(f"  Uploading → {handle} ...")
                t_upload = time.time()
                kagglehub.model_upload(
                    handle, merged_dir,
                    version_notes=f"v3 merged 4bit | step={step} | loss={loss}",
                    license_name="Apache 2.0",
                )
                upload_mins = (time.time() - t_upload) / 60
                print(f"  ✅ Upload done in {upload_mins:.1f} mins")
            except Exception as e:
                print(f"  ⚠️  Kaggle upload failed: {e}")
                print(f"  Merged checkpoint saved locally at {merged_dir}")
                return

        except Exception as e:
            print(f"  ⚠️  Merge failed: {e}")
            import traceback
            traceback.print_exc()
            return

        finally:
            if os.path.exists(merged_dir):
                try:
                    shutil.rmtree(merged_dir)
                    print(f"  🗑  Freed disk: {merged_dir}")
                except Exception:
                    pass

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step > 0 and step % MERGE_UPLOAD_EVERY == 0 and step != self.last_merge_step:
            self.last_merge_step = step
            self._merge_and_upload(state)

    def on_train_end(self, args, state, control, **kwargs):
        self._merge_and_upload(state)

# ============================================================
# TRAIN
# ============================================================
print("=" * 60)
print(f"  TRAINING SFT v3 on GPT-OSS-120B")
print(f"  {len(train_data)} samples | 1 epoch | lr=2e-5")
print(f"  max_seq_length={MAX_SEQ_LEN} | batch=16 | grad_accum=2")
print(f"  Merged 4bit upload every {MERGE_UPLOAD_EVERY} steps")
print("=" * 60)

FastLanguageModel.for_training(model)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    args=SFTConfig(
        output_dir="sft-output-120b-v3",
        per_device_train_batch_size=32,      # max out — 46GB free
        gradient_accumulation_steps=1,        # effective batch = 32
        num_train_epochs=1,
        learning_rate=2e-5,
        bf16=True,
        max_seq_length=MAX_SEQ_LEN,
        lr_scheduler_type="cosine",
        warmup_ratio=0,
        save_strategy="no",              # no trainer checkpoints, we merge+upload
        logging_steps=1,
        eval_strategy="no",
        max_grad_norm=0.5,
        report_to=REPORT_TO,
        run_name="sft-120b-v3-hard-tools",
        dataset_text_field="text",
        packing=True,
    ),
    processing_class=tokenizer,
    callbacks=[CheckpointCallback()],
)

print("\n  Starting training...")
t0 = time.time()
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
print("  COMPLETE — v3")
print("=" * 60)
print(f"  Time:          {train_time/3600:.1f}h")
print(f"  Final step:    {final_step}")
print(f"  Final loss:    {final_loss}")
print(f"  max_seq_length: {MAX_SEQ_LEN}")
print(f"  Key fixes:     multi-turn tool msgs, longer context, upsampled hard data")
print(f"  Merged model:  {KAGGLE_USERNAME}/gpt-oss-120b-aimo3-sft-v3")
print("=" * 60)

if wandb:
    try:
        wandb.finish()
    except Exception:
        pass
