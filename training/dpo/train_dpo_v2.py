"""
AIMO3 DPO Training — Qwen3-32B on H200 (Batched Version with Debug)
====================================================================
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
        print(f"W&B: enabled (project={os.environ.get('WANDB_PROJECT', 'aimo3-dpo')})")
    except Exception as e:
        print(f"W&B: failed ({e}), disabling")
        REPORT_TO = "none"
else:
    REPORT_TO = "none"
    print("W&B: disabled")

os.environ["UNSLOTH_VLLM_STANDBY"] = "1"

from datasets import load_from_disk, Dataset
from transformers import set_seed
from unsloth import FastLanguageModel
from trl import DPOTrainer, DPOConfig
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
    max_seq_length=8192,
    load_in_4bit=True,
    fast_inference=True,
    max_lora_rank=64,
    gpu_memory_utilization=0.80,
)

model = FastLanguageModel.get_peft_model(
    model, r=64,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=64, lora_dropout=0,
    use_gradient_checkpointing="unsloth",
)
print("Model loaded.\n")

# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("  Loading dataset")
print("=" * 60)

ds = load_from_disk(DATA_DIR)
ds = ds.shuffle(seed=42)

print(f"  Total: {len(ds)} rows")
print(f"  Columns: {ds.column_names}")

# ============================================================
# DEBUG: Print multiple samples to verify answer formats
# ============================================================

print("\n" + "=" * 60)
print("  DEBUG: Checking 10 random samples")
print("=" * 60)

for i in range(min(10, len(ds))):
    sample = ds[i]
    print(f"\n--- Sample {i} ---")
    print(f"  uuid: {sample.get('uuid', 'N/A')}")
    print(f"  expected_answer: {repr(sample.get('expected_answer'))} (type: {type(sample.get('expected_answer')).__name__})")
    print(f"  original_expected_answer: {repr(sample.get('original_expected_answer'))} (type: {type(sample.get('original_expected_answer')).__name__})")
    print(f"  changed_answer_to_majority: {sample.get('changed_answer_to_majority')}")
    print(f"  problem: {str(sample.get('problem', ''))[:150]}...")
    
    # Try to parse expected_answer as number
    ans = sample.get('expected_answer')
    try:
        num_ans = float(ans) if ans is not None else None
        print(f"  parsed as float: {num_ans}")
        print(f"  in range [0, 99999]: {0 <= num_ans <= 99999 if num_ans is not None else False}")
    except:
        print(f"  parsed as float: FAILED (not a simple number)")
    
    # Check messages for answer extraction
    messages = sample.get('messages', [])
    if messages and len(messages) >= 2:
        assistant_msg = messages[-1].get('content', '') if isinstance(messages[-1], dict) else str(messages[-1])
        # Try to find boxed answer in assistant message
        boxed = re.findall(r'\\boxed\{([^}]+)\}', assistant_msg)
        if boxed:
            print(f"  boxed answer in messages: {boxed[-1]}")
        else:
            print(f"  boxed answer in messages: NOT FOUND")
    else:
        print(f"  messages: empty or invalid")

print("\n" + "=" * 60)
print("  END DEBUG")
print("=" * 60)

# ============================================================
# FILTER DATASET: Keep only answers in [0, 99999]
# ============================================================

print("\n" + "=" * 60)
print("  FILTERING: Keeping only answers in [0, 99999]")
print("=" * 60)

def is_valid_answer(ans):
    """Check if answer is a valid number in range [0, 99999]."""
    if ans is None:
        return False
    try:
        num = float(ans)
        return 0 <= num <= 99999
    except (ValueError, TypeError):
        return False

# Filter dataset
original_count = len(ds)
ds = ds.filter(lambda x: is_valid_answer(x.get('expected_answer')), num_proc=4)
filtered_count = len(ds)

print(f"  Original: {original_count} rows")
print(f"  After filtering: {filtered_count} rows")
print(f"  Removed: {original_count - filtered_count} rows ({100*(original_count-filtered_count)/original_count:.1f}%)")

if filtered_count == 0:
    raise ValueError("No valid samples after filtering! Check your data.")

# ============================================================
# DETECT COLUMNS
# ============================================================

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

# ============================================================
# SPLIT DATASET
# ============================================================

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
    # Clean up the extracted answer
    ans = m[-1].strip()
    # Remove $ wrappers if present
    ans = ans.strip('$')
    return ans

def check(pred, gt):
    """Check if prediction matches ground truth."""
    if pred is None or gt is None:
        return False
    # Try numeric comparison first
    try:
        pred_num = float(pred)
        gt_num = float(gt)
        return abs(pred_num - gt_num) < 1e-6
    except (ValueError, TypeError):
        # Fall back to string comparison
        return str(pred).strip() == str(gt).strip()

def make_chat_prompt(question):
    """Create chat-formatted prompt for a question."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": fmt(question)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )

def gen_batch(prompts, temp=0.7, max_tokens=4096):
    """Generate for a batch of prompts using vLLM."""
    if not prompts:
        return []
    sampling_params = SamplingParams(
        temperature=temp, 
        top_p=0.95, 
        top_k=20, 
        max_tokens=max_tokens
    )
    outputs = model.fast_generate(prompts, sampling_params=sampling_params)
    return [out.outputs[0].text for out in outputs]

def gen_single(prompt, temp=0.7, max_tokens=4096):
    """Generate for a single prompt."""
    return gen_batch([prompt], temp=temp, max_tokens=max_tokens)[0]

def evaluate(n=200, batch_size=16):
    """Batched evaluation."""
    FastLanguageModel.for_inference(model)
    ok = 0
    total = min(n, len(eval_set))
    
    if total == 0:
        print("  WARNING: No evaluation samples!")
        return 0.0
    
    print(f"  Evaluating {total} problems with batch_size={batch_size}...")
    t0 = time.time()
    
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_indices = range(batch_start, batch_end)
        
        questions = [eval_set[i][QUESTION_COL] for i in batch_indices]
        answers = [eval_set[i][ANSWER_COL] for i in batch_indices]
        prompts = [make_chat_prompt(q) for q in questions]
        
        try:
            outputs = gen_batch(prompts, temp=0.6)
        except Exception as e:
            print(f"    Batch {batch_start}-{batch_end} failed: {e}, falling back to single")
            outputs = []
            for q in questions:
                try: 
                    outputs.append(gen_single(make_chat_prompt(q), temp=0.6))
                except: 
                    outputs.append("")
        
        for i, (out, gt) in enumerate(zip(outputs, answers)):
            pred = extract(out)
            if check(pred, gt):
                ok += 1
        
        done = batch_end
        acc = ok / done
        elapsed = time.time() - t0
        rate = done / elapsed * 3600 if elapsed > 0 else 0
        print(f"    {done}/{total} acc={acc:.3f} ({rate:.0f}/hr)")
    
    final_acc = ok / total
    print(f"  Eval: {ok}/{total} = {final_acc:.3f} (took {elapsed/60:.1f}min)")
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
            max_length=8192,
            max_prompt_length=2048,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            save_steps=5000,
            logging_steps=50,
            max_grad_norm=1.0,
            report_to=REPORT_TO,
            run_name=f"dpo-round{round_num}",
        ),
        train_dataset=Dataset.from_list(pairs),
        processing_class=tokenizer,
    )
    trainer.train()
    model.save_pretrained(f"checkpoint-round{round_num}")
    del trainer
    gc.collect()

# ============================================================
# PAIR GENERATION (BATCHED)
# ============================================================

def generate_pairs_round1(batch_size=16):
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
            outputs = gen_batch(prompts, temp=0.7)
        except Exception as e:
            print(f"    Batch failed: {e}, using singles")
            outputs = []
            for p in prompts:
                try: 
                    outputs.append(gen_single(p, temp=0.7))
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

    print(f"  Done: {len(pairs)} pairs, {skip} skip, {err} err, {(time.time()-t0)/3600:.1f}h")
    return pairs


def generate_pairs_gemini(judge, temp=0.8, batch_size=8):
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
        
        prompts_a = [make_chat_prompt(q) for q in questions]
        prompts_b = [make_chat_prompt(q) for q in questions]
        
        try:
            outputs_a = gen_batch(prompts_a, temp=temp)
            outputs_b = gen_batch(prompts_b, temp=temp)
        except Exception as e:
            print(f"    Batch failed: {e}, using singles")
            outputs_a, outputs_b = [], []
            for q in questions:
                try:
                    p = make_chat_prompt(q)
                    outputs_a.append(gen_single(p, temp=temp))
                    outputs_b.append(gen_single(p, temp=temp))
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

    print(f"  Done: {len(pairs)} pairs, {skip} skip, {err} err, {(time.time()-t0)/3600:.1f}h")
    return pairs

# ============================================================
# MAIN EXECUTION
# ============================================================

print("\n" + "=" * 60)
print("  BASELINE EVALUATION")
print("=" * 60)
baseline = evaluate(n=200, batch_size=16)

# Round 1
print("\n" + "=" * 60)
print("  ROUND 1: Direct DPO")
print("=" * 60)
pairs = generate_pairs_round1(batch_size=16)
train_round(pairs, 1, lr=5e-7, beta=0.10)
score_r1 = evaluate(n=200, batch_size=16)
upload(1)
print(f"  Round 1: {baseline:.3f} → {score_r1:.3f}")

# Round 2
print("\n" + "=" * 60)
print("  ROUND 2: Gemini-judged DPO")
print("=" * 60)
judge = setup_gemini()
pairs = generate_pairs_gemini(judge, temp=0.8, batch_size=8)
train_round(pairs, 2, lr=3e-7, beta=0.12)
score_r2 = evaluate(n=200, batch_size=16)
upload(2)
print(f"  Round 2: {score_r1:.3f} → {score_r2:.3f}")

# Round 3
print("\n" + "=" * 60)
print("  ROUND 3: Gemini-judged DPO (high temp)")
print("=" * 60)
pairs = generate_pairs_gemini(judge, temp=0.9, batch_size=8)
train_round(pairs, 3, lr=2e-7, beta=0.15)
score_r3 = evaluate(n=200, batch_size=16)
upload(3)
print(f"  Round 3: {score_r2:.3f} → {score_r3:.3f}")

# Summary
print("\n" + "=" * 60)
print("  TRAINING COMPLETE")
print("=" * 60)
print(f"  Baseline: {baseline:.3f}")
print(f"  Round 1:  {score_r1:.3f}")
print(f"  Round 2:  {score_r2:.3f}")
print(f"  Round 3:  {score_r3:.3f}")
print(f"\n  Models on Kaggle:")
print(f"    {KAGGLE_USERNAME}/qwen3-32b-aimo3-round1")
print(f"    {KAGGLE_USERNAME}/qwen3-32b-aimo3-round2")
print(f"    {KAGGLE_USERNAME}/qwen3-32b-aimo3-round3")
print("=" * 60)
