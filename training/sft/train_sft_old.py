"""
AIMO3 Phase 1: SFT on Nemotron Gold Solutions
===============================================
Hardware: 1× NVIDIA H200 (141GB)
Model: Qwen3-32B (4-bit via Unsloth)
Dataset: 130K Nemotron Math v2 (AIMO3 filtered)

Gold solutions are reformatted as:
  <think>
  [step-by-step reasoning]
  </think>
  \boxed{answer}

This teaches the model to reason inside <think> and output \boxed{} after.
Checkpoints + uploads to Kaggle every 2000 steps.

Usage: python3 train_sft.py
"""

import os, sys, re, json, time, gc
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
        os.environ["WANDB_PROJECT"] = os.environ.get("WANDB_PROJECT", "aimo3-sft")
        REPORT_TO = "wandb"
        print("W&B: enabled")
    except Exception as e:
        print(f"W&B: failed ({e})")
        REPORT_TO = "none"
        wandb = None
else:
    REPORT_TO = "none"
    print("W&B: disabled")

os.environ["UNSLOTH_VLLM_STANDBY"] = "1"

from datasets import load_from_disk
from transformers import set_seed, TrainerCallback
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from vllm import SamplingParams

set_seed(42)


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 60)
print("  Loading Qwen3-32B")
print("=" * 60)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-32B",
    max_seq_length=4096,
    load_in_4bit=True,
    fast_inference=True,
    max_lora_rank=64,
    gpu_memory_utilization=0.60,   # Lower — leave room for training gradients
)

model = FastLanguageModel.get_peft_model(
    model, r=64,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=64, lora_dropout=0,
    use_gradient_checkpointing="unsloth",
)
print("Model loaded.\n")


# ============================================================
# LOAD & FORMAT DATASET
# ============================================================

print("=" * 60)
print("  Loading and formatting dataset")
print("=" * 60)

ds = load_from_disk(DATA_DIR)
ds = ds.shuffle(seed=42)
print(f"  Raw: {len(ds)} rows")

def reformat_solution(solution, expected_answer):
    """
    Move reasoning inside <think> tags, put only \boxed{} after </think>.
    
    Input:  "Since ABC... The sum is 260. \boxed{260}"
    Output: "<think>\nSince ABC... The sum is 260.\n</think>\n\n\\boxed{260}"
    """
    # Find the LAST \boxed{...} in the solution
    boxed_match = re.search(r'(\\\[?\s*\\boxed\{[^}]+\}\s*\\\]?\s*)$', solution.strip())
    
    if boxed_match:
        reasoning = solution[:boxed_match.start()].strip()
        # Extract just \boxed{N} without surrounding \[ \]
        inner = re.search(r'\\boxed\{([^}]+)\}', boxed_match.group(1))
        boxed_str = f"\\boxed{{{inner.group(1)}}}" if inner else boxed_match.group(1).strip()
    else:
        # No \boxed at end — add it from expected_answer
        reasoning = solution.strip()
        boxed_str = f"\\boxed{{{expected_answer}}}"
    
    return f"<think>\n{reasoning}\n</think>\n\n{boxed_str}"


def format_for_sft(example):
    """Convert each sample to chat format with <think> reasoning."""
    msgs = example['messages']
    user_content = msgs[0]['content']
    assistant_content = msgs[1]['content']
    expected = example['expected_answer']
    
    # Reformat: reasoning inside <think>, \boxed{} outside
    new_assistant = reformat_solution(assistant_content, expected)
    
    # Build clean messages
    clean_msgs = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": new_assistant},
    ]
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        clean_msgs,
        tokenize=False,
        add_generation_prompt=False,
    )
    
    return {"text": text}


# Format all samples
print("  Formatting with <think> tags...")
t0 = time.time()
sft_data = ds.map(format_for_sft, num_proc=8, remove_columns=ds.column_names)
print(f"  Formatted {len(sft_data)} samples in {time.time()-t0:.0f}s")

# Verify
print(f"\n  Sample preview (first 600 chars):")
print(f"  {sft_data[0]['text'][:600]}")
print(f"\n  Sample preview (last 300 chars):")
print(f"  {sft_data[0]['text'][-300:]}")
print(f"  Has <think>: {'<think>' in sft_data[0]['text']}")
print(f"  Has </think>: {'</think>' in sft_data[0]['text']}")
print(f"  Has boxed: {'boxed' in sft_data[0]['text']}")

# Split for eval
eval_data = sft_data.select(range(0, min(2000, len(sft_data))))
train_data = sft_data.select(range(min(2000, len(sft_data)), len(sft_data)))
print(f"\n  Train: {len(train_data)} | Eval: {len(eval_data)}")

# Also keep raw dataset for eval answer checking
raw_eval = ds.select(range(0, min(2000, len(ds))))


# ============================================================
# CHECKPOINT + UPLOAD CALLBACK
# ============================================================

class KaggleCheckpointCallback(TrainerCallback):
    """Merge LoRA + eval + upload to Kaggle periodically."""
    
    def __init__(self, upload_every=300):
        self.upload_every = upload_every
        self.last_step = 0
        self.scores = []
    
    def _write_readme(self, path, step, loss, score=None):
        """Write README for Kaggle model page."""
        readme = f"""# Qwen3-32B AIMO3 — SFT Checkpoint

## Model
- **Base**: Qwen3-32B (4-bit LoRA, merged to 16-bit)
- **Training**: SFT on 120K Nemotron Math v2 gold solutions
- **Format**: Reasoning inside `<think>` tags, answer in `\\boxed{{}}`

## This Checkpoint
- **Step**: {step}
- **Loss**: {loss}
{"- **Eval accuracy**: " + f"{score:.3f}" if score else ""}

## Training Data
- [ritwikakancharla/nemotron-math-v2-filtered-high](https://www.kaggle.com/datasets/ritwikakancharla/nemotron-math-v2-filtered-high)
- 120K competition math problems from AoPS
- Filtered to integer answers 0-99999 (AIMO3 format)

## Usage
Use with vLLM for inference. See the AIMO3 submission notebook.

## Expected Output Format
```
<think>
[step-by-step reasoning]
</think>

\\boxed{{answer}}
```
"""
        with open(os.path.join(path, "README.md"), "w") as f:
            f.write(readme)
    
    def _quick_eval(self, step, n=50):
        """Fast eval on 50 problems — takes ~5 min."""
        print(f"    [Eval] Running {n}-problem eval...")
        FastLanguageModel.for_inference(model)
        ok = 0
        t0 = time.time()
        for i in range(min(n, len(raw_eval))):
            try:
                msgs = raw_eval[i]['messages']
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": msgs[0]['content']}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=True,
                )
                out = model.fast_generate([prompt],
                    sampling_params=SamplingParams(temperature=0.6, top_p=0.95, max_tokens=2048))
                text = out[0].outputs[0].text
                pred = extract(text)
                gt = raw_eval[i]['expected_answer']
                if check(pred, gt):
                    ok += 1
            except:
                pass
        score = ok / n
        elapsed = time.time() - t0
        print(f"    [Eval] Step {step}: {ok}/{n} = {score:.3f} ({elapsed/60:.1f}min)")
        self.scores.append((step, score))
        
        if wandb:
            wandb.log({"eval/accuracy": score, "eval/step": step})
        
        FastLanguageModel.for_training(model)
        return score
    
    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step > 0 and step % self.upload_every == 0 and step != self.last_step:
            self.last_step = step
            loss = state.log_history[-1].get('loss', '?') if state.log_history else '?'
            print(f"\n  [Step {step}] loss={loss}")
            
            # Quick eval
            score = self._quick_eval(step)
            
            # Save LoRA + trainer state locally (so resume works)
            ckpt = f"sft-output/checkpoint-{step}"
            os.makedirs(ckpt, exist_ok=True)
            model.save_pretrained(ckpt)
            state.save_to_json(os.path.join(ckpt, "trainer_state.json"))
            print(f"    Checkpoint saved: {ckpt}")
            
            # Merge + upload with README
            try:
                import kagglehub
                merged = f"merged-sft-step{step}"
                model.save_pretrained_merged(merged, tokenizer, save_method="merged_16bit")
                self._write_readme(merged, step, loss, score)
                handle = f"{KAGGLE_USERNAME}/qwen3-32b-aimo3-sft/transformers/default"
                kagglehub.model_upload(
                    handle, merged,
                    version_notes=f"step {step} | loss={loss} | acc={score:.3f}",
                    license_name="Apache 2.0",
                )
                print(f"    Uploaded: {handle} (step {step})")
            except Exception as e:
                print(f"    Upload failed: {e}")
    
    def on_train_end(self, args, state, control, **kwargs):
        step = state.global_step
        loss = state.log_history[-1].get('loss', '?') if state.log_history else '?'
        print(f"\n  [Final] Step {step}, uploading final model...")
        
        score = self._quick_eval(step, n=100)
        
        try:
            import kagglehub
            merged = "merged-sft-final"
            model.save_pretrained_merged(merged, tokenizer, save_method="merged_16bit")
            self._write_readme(merged, step, loss, score)
            handle = f"{KAGGLE_USERNAME}/qwen3-32b-aimo3-sft/transformers/default"
            kagglehub.model_upload(handle, merged,
                version_notes=f"FINAL step {step} | acc={score:.3f}",
                license_name="Apache 2.0")
            print(f"    Final uploaded: {handle}")
        except Exception as e:
            print(f"    Final upload failed: {e}")
        
        print(f"\n  Score progression:")
        for s, acc in self.scores:
            bar = "█" * int(acc * 50)
            print(f"    Step {s:>6}: {acc:.3f} {bar}")


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def extract(text):
    if not text:
        return None
    m = re.findall(r'\\boxed\{([^}]+)\}', text)
    if m:
        return m[-1].strip().strip('$')
    return None

def check(pred, gt):
    if pred is None or gt is None:
        return False
    try:
        return abs(float(pred) - float(gt)) < 1e-6
    except (ValueError, TypeError):
        return str(pred).strip() == str(gt).strip()

def evaluate(n=200):
    """Quick eval: generate and check accuracy."""
    FastLanguageModel.for_inference(model)
    ok = 0
    total = min(n, len(raw_eval))
    t0 = time.time()
    
    for i in range(total):
        try:
            msgs = raw_eval[i]['messages']
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": msgs[0]['content']}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            out = model.fast_generate(
                [prompt],
                sampling_params=SamplingParams(
                    temperature=0.6, top_p=0.95, max_tokens=2048,
                ),
            )
            text = out[0].outputs[0].text
            pred = extract(text)
            gt = raw_eval[i]['expected_answer']
            if check(pred, gt):
                ok += 1
        except:
            pass
        if (i+1) % 50 == 0:
            print(f"    {i+1}/{total} acc={ok/(i+1):.3f} ({(time.time()-t0)/60:.1f}min)")
    
    score = ok / total
    print(f"  Eval: {ok}/{total} = {score:.3f} ({(time.time()-t0)/60:.1f}min)")
    
    if wandb:
        wandb.log({"eval/accuracy": score, "eval/correct": ok, "eval/total": total})
    
    return score


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 60)
print("  TRAINING SFT")
print(f"  {len(train_data)} samples, 1 epoch")
print(f"  Estimated steps: ~{len(train_data) // 16}")
print("=" * 60)

FastLanguageModel.for_training(model)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    args=SFTConfig(
        output_dir="sft-output",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=2e-5,
        bf16=True,
        max_seq_length=2048,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        save_steps=300,
        logging_steps=1,               # Log loss EVERY step
        eval_strategy="steps",
        eval_steps=300,
        max_grad_norm=1.0,
        report_to=REPORT_TO,
        run_name="sft-think-boxed",
        dataset_text_field="text",
        packing=True,
    ),
    processing_class=tokenizer,
    callbacks=[KaggleCheckpointCallback(upload_every=300)],
)

print("\n  Starting training...")
t0 = time.time()

# Resume from checkpoint if sft-output has one
import glob
checkpoints = sorted(glob.glob("sft-output/checkpoint-*"))
if checkpoints:
    print(f"  Resuming from {checkpoints[-1]}...")
    trainer.train(resume_from_checkpoint=checkpoints[-1])
else:
    print("  Starting fresh...")
    trainer.train()

train_time = time.time() - t0
print(f"\n  SFT done in {train_time/3600:.1f}h")

# Save final LoRA
model.save_pretrained("checkpoint-sft-final")

del trainer
gc.collect()

print("\n" + "=" * 60)
print("  SFT COMPLETE")
print("=" * 60)
print(f"  Baseline: ~12.5% (from earlier run)")
print(f"  Time:     {train_time/3600:.1f}h")
print(f"\n  Model on Kaggle: {KAGGLE_USERNAME}/qwen3-32b-aimo3-sft")
print(f"  Next: Run GRPO with train_grpo.py")
print("=" * 60)

if wandb:
    wandb.log({"final/train_hours": train_time / 3600})
    wandb.finish()
