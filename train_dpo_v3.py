"""
AIMO3 DPO Training — Qwen3-32B on H200 (16K context, W&B logging, no filtering)
==================================================================================
Loads config from .env file.
Usage: python3 train_dpo.py
"""

import os, sys, re, json, time, gc
from dotenv import load_dotenv
load_dotenv()

KAGGLE_USERNAME = os.environ["KAGGLE_USERNAME"]
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("VERTEX_API_KEY", "")
MODEL_DIR = os.environ.get("MODEL_DIR", "./models/Qwen3-32B")
DATA_DIR = os.environ.get("DATA_DIR", "./data/nemotron-math-v2")

WANDB_KEY = os.environ.get("WANDB_API_KEY", "")
if WANDB_KEY:
    try:
        import wandb
        wandb.login(key=WANDB_KEY)
        REPORT_TO = "wandb"
        # Initialize wandb run for the whole session
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "aimo3-dpo"),
            entity="ritwikareddykancharla-n-a",
            name="aimo3-dpo-full-run",
            config={
                "model": "Qwen3-32B",
                "max_seq_length": 16384,
                "load_in_4bit": True,
                "lora_rank": 64,
            }
        )
        print(f"W&B: enabled (project={os.environ.get('WANDB_PROJECT', 'aimo3-dpo')})")
    except Exception as e:
        print(f"W&B: failed ({e}), disabling")
        REPORT_TO = "none"
        wandb = None
else:
    REPORT_TO = "none"
    print("W&B: disabled")
    wandb = None

os.environ["UNSLOTH_VLLM_STANDBY"] = "1"

from datasets import load_from_disk, Dataset
from transformers import set_seed
from unsloth import FastLanguageModel
from trl import DPOTrainer, DPOConfig
from vllm import SamplingParams

set_seed(42)

# ============================================================
# LOAD MODEL (16K context)
# ============================================================

print("\n" + "=" * 60)
print("  Loading Qwen3-32B (16K context)")
print("=" * 60)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-32B",
    max_seq_length=16384,  # INCREASED from 8192
    load_in_4bit=True,
    fast_inference=True,
    max_lora_rank=64,
    gpu_memory_utilization=0.85,  # Slightly higher for bigger batches
)

model = FastLanguageModel.get_peft_model(
    model, r=64,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=64, lora_dropout=0,
    use_gradient_checkpointing="unsloth",
)
print("Model loaded.\n")

# ============================================================
# LOAD DATASET (NO FILTERING)
# ============================================================

print("=" * 60)
print("  Loading dataset")
print("=" * 60)

ds = load_from_disk(DATA_DIR)
ds = ds.shuffle(seed=42)

print(f"  Total: {len(ds)} rows")
print(f"  Columns: {ds.column_names}")

# Quick peek at first sample
sample = ds[0]
print(f"\n  Sample problem: {str(sample.get('problem', ''))[:100]}...")
print(f"  Sample answer: {repr(sample.get('expected_answer'))}")

# Detect columns
QUESTION_COL = ANSWER_COL = SOLUTION_COL = None
for col in ds.column_names:
    cl = col.lower()
    if cl in ('question', 'problem', 'prompt', 'input'):
        QUESTION_COL = col
    elif cl in ('answer', 'expected_answer', 'final_answer', 'target'):
        ANSWER_COL = col
    elif cl in ('solution', 'response', 'output', 'generation'):
        SOLUTION_COL = col

print(f"\n  Detected columns:")
print(f"    Question: {QUESTION_COL}")
print(f"    Answer: {ANSWER_COL}")
print(f"    Solution: {SOLUTION_COL}")

assert QUESTION_COL and ANSWER_COL, f"Cannot detect columns from: {ds.column_names}"

# Split dataset
eval_set = ds.select(range(0, min(5000, len(ds))))
train_set = ds.select(range(min(5000, len(ds)), len(ds)))
print(f"\n  Split: Train={len(train_set)} | Eval={len(eval_set)}")

# ============================================================
# HELPERS
# ============================================================

def fmt(q):
    return f"Please reason step by step, and put your final answer within \\boxed{{}}.\n\n{q}"

def extract(text):
    """Extract answer from \\boxed{}."""
    if not text:
        return None
    m = re.findall(r'\\boxed\{([^}]+)\}', text)
    if not m:
        return None
    ans = m[-1].strip()
    ans = ans.strip('$')
    return ans

def check(pred, gt):
    """Check if prediction matches ground truth."""
    if pred is None or gt is None:
        return False
    try:
        pred_num = float(pred)
        gt_num = float(gt)
        return abs(pred_num - gt_num) < 1e-6
    except (ValueError, TypeError):
        return str(pred).strip() == str(gt).strip()

def make_chat_prompt(question):
    """Create chat-formatted prompt for a question."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": fmt(question)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )

def gen_batch(prompts, temp=0.7, max_tokens=8192):  # INCREASED max_tokens
    """Generate for a batch of prompts using vLLM."""
    if not prompts:
        return []
    sampling_params = SamplingParams(
        temperature=temp, 
        top_p=0.95, 
        top_k=20, 
        max_tokens=max_tokens  # Now 8192 for longer reasoning
    )
    outputs = model.fast_generate(prompts, sampling_params=sampling_params)
    return [out.outputs[0].text for out in outputs]

def gen_single(prompt, temp=0.7, max_tokens=8192):
    """Generate for a single prompt."""
    return gen_batch([prompt], temp=temp, max_tokens=max_tokens)[0]

def evaluate(n=200, batch_size=32):  # INCREASED batch_size
    """
    Batched evaluation with live accuracy display and W&B logging.
    """
    FastLanguageModel.for_inference(model)
    ok = 0
    total = min(n, len(eval_set))
    
    if total == 0:
        print("  WARNING: No evaluation samples!")
        return 0.0
    
    print(f"\n  Evaluating {total} problems with batch_size={batch_size}...")
    print(f"  {'Problem':<8} {'Predicted':<15} {'Ground Truth':<15} {'Correct?':<8} {'Running Acc'}")
    print(f"  {'-'*70}")
    
    t0 = time.time()
    all_results = []
    
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_indices = list(range(batch_start, batch_end))
        
        questions = [eval_set[i][QUESTION_COL] for i in batch_indices]
        answers = [eval_set[i][ANSWER_COL] for i in batch_indices]
        prompts = [make_chat_prompt(q) for q in questions]
        
        # Generate batch
        try:
            outputs = gen_batch(prompts, temp=0.6, max_tokens=8192)
        except Exception as e:
            print(f"    Batch {batch_start}-{batch_end} failed: {e}, falling back to single")
            outputs = []
            for q in questions:
                try: 
                    outputs.append(gen_single(make_chat_prompt(q), temp=0.6, max_tokens=8192))
                except: 
                    outputs.append("")
        
        # Check answers and display live results
        for i, idx in enumerate(batch_indices):
            out = outputs[i]
            gt = answers[i]
            pred = extract(out)
            is_correct = check(pred, gt)
            
            if is_correct:
                ok += 1
            
            # Store result
            result = {
                "index": idx,
                "question": questions[i][:50] + "..." if len(questions[i]) > 50 else questions[i],
                "predicted": str(pred)[:20] if pred else "None",
                "ground_truth": str(gt)[:20],
                "correct": is_correct,
                "output_preview": out[:100] + "..." if out else ""
            }
            all_results.append(result)
            
            # Print live row (only first 5 and every 10th to avoid spam)
            global_idx = batch_start + i + 1
            if global_idx <= 5 or global_idx % 10 == 0 or global_idx == total:
                status = "✓" if is_correct else "✗"
                running_acc = ok / global_idx
                print(f"  {global_idx:<8} {str(pred)[:15]:<15} {str(gt)[:15]:<15} {status:<8} {running_acc:.3f}")
        
        # Batch progress and W&B logging
        done = batch_end
        acc = ok / done
        elapsed = time.time() - t0
        rate = done / elapsed * 3600 if elapsed > 0 else 0
        
        # Log to W&B
        if wandb and REPORT_TO == "wandb":
            wandb.log({
                "eval/completed": done,
                "eval/total": total,
                "eval/accuracy": acc,
                "eval/rate_per_hour": rate,
                "eval/elapsed_minutes": elapsed / 60,
            })
        
        print(f"  [Batch] {done}/{total} completed | Acc: {acc:.3f} | Speed: {rate:.0f}/hr | Time: {elapsed/60:.1f}min")
    
    # Final summary
    final_acc = ok / total
    total_time = time.time() - t0
    
    print(f"\n  {'='*70}")
    print(f"  FINAL EVALUATION RESULTS")
    print(f"  {'='*70}")
    print(f"  Total evaluated: {total}")
    print(f"  Correct: {ok}")
    print(f"  Accuracy: {final_acc:.3f} ({final_acc*100:.1f}%)")
    print(f"  Time taken: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"  Speed: {total/total_time*3600:.0f} problems/hour")
    print(f"  {'='*70}")
    
    # Log final metrics to W&B
    if wandb and REPORT_TO == "wandb":
        wandb.log({
            "eval/final_accuracy": final_acc,
            "eval/correct_count": ok,
            "eval/total_time_hours": total_time / 3600,
            "eval/speed_per_hour": total / total_time * 3600,
        })
        
        # Create a summary table of results
        results_table = wandb.Table(columns=["problem", "predicted", "ground_truth", "correct"])
        for r in all_results[:50]:  # Log first 50 for inspection
            results_table.add_data(r["question"], r["predicted"], r["ground_truth"], r["correct"])
        wandb.log({"eval/sample_results": results_table})
    
    return final_acc

# ============================================================
# KAGGLE UPLOAD
# ============================================================

def upload(round_num):
    import kagglehub
    out = f"qwen3-32b-aimo3-round{round_num}"
    print(f"\n  Merging LoRA → {out}...")
    model.save_pretrained_merged(out, tokenizer, save_method="merged_16bit")

    handle = f"{KAGGLE_USERNAME}/qwen3-32b-aimo3-round{round_num}/transformers/default"
    print(f"  Uploading: {handle}")
    try:
        kagglehub.model_upload(handle, out, version_notes=f"DPO round {round_num}", license_name="Apache 2.0")
        print("  Done!\n")
    except Exception as e:
        print(f"  Upload failed: {e}\n  Model saved at: {out}/\n")

# ============================================================
# GEMINI JUDGE
# ============================================================

def setup_gemini():
    from google import genai
    client = genai.Client(api_key=GEMINI_API_KEY)

    PROMPT = """Correct answer: {gt}
Student solution: {sol}
Reply ONLY with JSON: {{"answer_correct": bool, "reasoning_valid": bool, "score": 0.0-1.0}}
1.0=correct+sound, 0.5=correct+flawed, 0.3=wrong+good reasoning, 0.0=wrong+bad"""

    def judge(sol, gt):
        try:
            r = client.models.generate_content(model="gemini-2.5-flash",
                contents=PROMPT.format(gt=gt, sol=sol[:3000]))
            return json.loads(r.text)
        except: 
            return None

    print("  Testing Gemini...", end=" ")
    t = judge("2+2=4, answer is 4", "4")
    print(f"OK: {t}")
    return judge

# ============================================================
# DPO TRAINING
# ============================================================

def train_round(pairs, round_num, lr, beta):
    print(f"\n  Training round {round_num}: {len(pairs)} pairs, lr={lr}, beta={beta}")
    
    if len(pairs) == 0:
        print("  WARNING: No pairs to train on! Skipping.")
        return
    
    with open(f"pairs_round{round_num}.json", "w") as f:
        json.dump(pairs, f)

    FastLanguageModel.for_training(model)
    
    # Log training start to W&B
    if wandb and REPORT_TO == "wandb":
        wandb.log({
            f"round_{round_num}/start": True,
            f"round_{round_num}/num_pairs": len(pairs),
            f"round_{round_num}/lr": lr,
            f"round_{round_num}/beta": beta,
        })
    
    trainer = DPOTrainer(
        model=model,
        args=DPOConfig(
            output_dir=f"dpo-round{round_num}",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=1,
            learning_rate=lr,
            beta=beta,
            bf16=True,
            max_length=16384,  # Match model context
            max_prompt_length=4096,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            save_steps=5000,
            logging_steps=50,
            max_grad_norm=1.0,
            report_to=REPORT_TO,  # W&B logging during training
            run_name=f"dpo-round{round_num}",
        ),
        train_dataset=Dataset.from_list(pairs),
        processing_class=tokenizer,
    )
    trainer.train()
    model.save_pretrained(f"checkpoint-round{round_num}")
    
    # Log training end to W&B
    if wandb and REPORT_TO == "wandb":
        wandb.log({
            f"round_{round_num}/complete": True,
        })
    
    del trainer
    gc.collect()

# ============================================================
# PAIR GENERATION (BATCHED)
# ============================================================

def generate_pairs_round1(batch_size=32):  # INCREASED batch_size
    """Round 1: 1 solution per problem."""
    FastLanguageModel.for_inference(model)
    pairs, skip, err = [], 0, 0
    t0 = time.time()
    total = len(train_set)

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_indices = list(range(batch_start, batch_end))
        
        prompts = [make_chat_prompt(train_set[i][QUESTION_COL]) for i in batch_indices]
        gts = [train_set[i][ANSWER_COL] for i in batch_indices]
        
        try:
            outputs = gen_batch(prompts, temp=0.7, max_tokens=8192)
        except Exception as e:
            print(f"    Batch failed: {e}, using singles")
            outputs = []
            for p in prompts:
                try: 
                    outputs.append(gen_single(p, temp=0.7, max_tokens=8192))
                except: 
                    outputs.append("")
        
        for i, idx in enumerate(batch_indices):
            try:
                out = outputs[i]
                gt = gts[i]
                pred = extract(out)
                
                if check(pred, gt):
                    skip += 1
                else:
                    gold = train_set[idx].get(SOLUTION_COL)
                    if not gold:
                        gold = f"The answer is \\boxed{{{gt}}}."
                    pairs.append({
                        "prompt": fmt(train_set[idx][QUESTION_COL]), 
                        "chosen": gold, 
                        "rejected": out
                    })
            except:
                err += 1
        
        if batch_end % 5000 < batch_size or batch_end == total:
            elapsed = time.time() - t0
            rate = batch_end / elapsed * 3600 if elapsed > 0 else 0
            print(f"  {batch_end}/{total} | {len(pairs)} pairs | {rate:.0f}/hr")
            
            # Log progress to W&B
            if wandb and REPORT_TO == "wandb":
                wandb.log({
                    "pairs_gen/completed": batch_end,
                    "pairs_gen/total": total,
                    "pairs_gen/collected": len(pairs),
                    "pairs_gen/skipped": skip,
                    "pairs_gen/errors": err,
                    "pairs_gen/rate_per_hour": rate,
                })

    print(f"  Done: {len(pairs)} pairs, {skip} skip, {err} err, {(time.time()-t0)/3600:.1f}h")
    return pairs


def generate_pairs_gemini(judge, temp=0.8, batch_size=16):  # INCREASED (was 8)
    """Rounds 2-3: 2 solutions per problem, Gemini judges ties."""
    FastLanguageModel.for_inference(model)
    pairs, skip, err = [], 0, 0
    t0 = time.time()
    total = len(train_set)

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_indices = list(range(batch_start, batch_end))
        
        questions = [train_set[i][QUESTION_COL] for i in batch_indices]
        gts = [train_set[i][ANSWER_COL] for i in batch_indices]
        
        # Generate 2 solutions per problem
        prompts_a = [make_chat_prompt(q) for q in questions]
        prompts_b = [make_chat_prompt(q) for q in questions]
        
        try:
            outputs_a = gen_batch(prompts_a, temp=temp, max_tokens=8192)
            outputs_b = gen_batch(prompts_b, temp=temp, max_tokens=8192)
        except Exception as e:
            print(f"    Batch failed: {e}, using singles")
            outputs_a, outputs_b = [], []
            for q in questions:
                try:
                    p = make_chat_prompt(q)
                    outputs_a.append(gen_single(p, temp=temp, max_tokens=8192))
                    outputs_b.append(gen_single(p, temp=temp, max_tokens=8192))
                except:
                    outputs_a.append("")
                    outputs_b.append("")
        
        for i, idx in enumerate(batch_indices):
            try:
                gt = gts[i]
                a, b = outputs_a[i], outputs_b[i]
                pred_a, pred_b = extract(a), extract(b)
                ok_a, ok_b = check(pred_a, gt), check(pred_b, gt)

                if ok_a and not ok_b:
                    pairs.append({"prompt": fmt(questions[i]), "chosen": a, "rejected": b})
                elif ok_b and not ok_a:
                    pairs.append({"prompt": fmt(questions[i]), "chosen": b, "rejected": a})
                elif not ok_a and not ok_b:
                    gold = train_set[idx].get(SOLUTION_COL)
                    if not gold:
                        gold = f"The answer is \\boxed{{{gt}}}."
                    pairs.append({"prompt": fmt(questions[i]), "chosen": gold, "rejected": a})
                elif ok_a and ok_b:
                    ja, jb = judge(a, gt), judge(b, gt)
                    sa = ja["score"] if ja else 1.0
                    sb = jb["score"] if jb else 1.0
                    if abs(sa - sb) >= 0.2:
                        pairs.append({
                            "prompt": fmt(questions[i]),
                            "chosen": a if sa > sb else b,
                            "rejected": b if sa > sb else a
                        })
                    else: 
                        skip += 1
                else: 
                    skip += 1
            except: 
                err += 1
        
        if batch_end % 5000 < batch_size or batch_end == total:
            elapsed = time.time() - t0
            rate = batch_end / elapsed * 3600 if elapsed > 0 else 0
            print(f"  {batch_end}/{total} | {len(pairs)} pairs | {rate:.0f}/hr")
            
            if wandb and REPORT_TO == "wandb":
                wandb.log({
                    "pairs_gemini/completed": batch_end,
                    "pairs_gemini/collected": len(pairs),
                    "pairs_gemini/skipped": skip,
                    "pairs_gemini/errors": err,
                })

    print(f"  Done: {len(pairs)} pairs, {skip} skip, {err} err, {(time.time()-t0)/3600:.1f}h")
    return pairs

# ============================================================
# MAIN EXECUTION
# ============================================================

print("\n" + "=" * 60)
print("  BASELINE EVALUATION")
print("=" * 60)
baseline = evaluate(n=200, batch_size=32)

# Log baseline to W&B
if wandb and REPORT_TO == "wandb":
    wandb.log({"baseline/accuracy": baseline})

# Round 1
print("\n" + "=" * 60)
print("  ROUND 1: Direct DPO")
print("=" * 60)
pairs = generate_pairs_round1(batch_size=32)
train_round(pairs, 1, lr=5e-7, beta=0.10)
score_r1 = evaluate(n=200, batch_size=32)
upload(1)
print(f"  Round 1: {baseline:.3f} → {score_r1:.3f}")

if wandb and REPORT_TO == "wandb":
    wandb.log({"round1/accuracy": score_r1, "round1/improvement": score_r1 - baseline})

# Round 2
print("\n" + "=" * 60)
print("  ROUND 2: Gemini-judged DPO")
print("=" * 60)
judge = setup_gemini()
pairs = generate_pairs_gemini(judge, temp=0.8, batch_size=16)
train_round(pairs, 2, lr=3e-7, beta=0.12)
score_r2 = evaluate(n=200, batch_size=32)
upload(2)
print(f"  Round 2: {score_r1:.3f} → {score_r2:.3f}")

if wandb and REPORT_TO == "wandb":
    wandb.log({"round2/accuracy": score_r2, "round2/improvement": score_r2 - score_r1})

# Round 3
print("\n" + "=" * 60)
print("  ROUND 3: Gemini-judged DPO (high temp)")
print("=" * 60)
pairs = generate_pairs_gemini(judge, temp=0.9, batch_size=16)
train_round(pairs, 3, lr=2e-7, beta=0.15)
score_r3 = evaluate(n=200, batch_size=32)
upload(3)
print(f"  Round 3: {score_r2:.3f} → {score_r3:.3f}")

if wandb and REPORT_TO == "wandb":
    wandb.log({"round3/accuracy": score_r3, "round3/improvement": score_r3 - score_r2})

# Final summary
print("\n" + "=" * 60)
print("  TRAINING COMPLETE")
print("=" * 60)
print(f"  Baseline: {baseline:.3f}")
print(f"  Round 1:  {score_r1:.3f}")
print(f"  Round 2:  {score_r2:.3f}")
print(f"  Round 3:  {score_r3:.3f}")

if wandb and REPORT_TO == "wandb":
    wandb.log({
        "final/baseline": baseline,
        "final/round1": score_r1,
        "final/round2": score_r2,
        "final/round3": score_r3,
        "final/best": max(baseline, score_r1, score_r2, score_r3),
    })
    wandb.finish()

print(f"\n  Models on Kaggle:")
print(f"    {KAGGLE_USERNAME}/qwen3-32b-aimo3-round1")
print(f"    {KAGGLE_USERNAME}/qwen3-32b-aimo3-round2")
print(f"    {KAGGLE_USERNAME}/qwen3-32b-aimo3-round3")
print("=" * 60)
