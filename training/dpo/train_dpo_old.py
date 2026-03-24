"""
AIMO3 DPO Training — Qwen3-32B on H200
========================================
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
REPORT_TO = "wandb" if WANDB_KEY else "none"
if WANDB_KEY:
    import wandb
    wandb.login(key=WANDB_KEY)
    print(f"W&B: enabled (project={os.environ.get('WANDB_PROJECT', 'aimo3-dpo')})")
else:
    print("W&B: disabled")

os.environ["UNSLOTH_VLLM_STANDBY"] = "1"

from datasets import load_from_disk, Dataset
from transformers import set_seed
from unsloth import FastLanguageModel
from trl import DPOTrainer, DPOConfig

set_seed(42)

# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 60)
print("  Loading Qwen3-32B")
print("=" * 60)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_DIR,
    max_seq_length=8192,
    load_in_fp8=True,
    fast_inference=True,
    max_lora_rank=64,
    gpu_memory_utilization=0.92,
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

# Show structure
print(f"  Total: {len(ds)} rows")
print(f"  Columns: {ds.column_names}")
sample = ds[0]
for k, v in sample.items():
    print(f"    {k}: {str(v)[:100]}")

# Auto-detect columns
QUESTION_COL = ANSWER_COL = SOLUTION_COL = None
for col in ds.column_names:
    cl = col.lower()
    if cl in ('question', 'problem', 'prompt', 'input'):
        QUESTION_COL = col
    elif cl in ('answer', 'expected_answer', 'final_answer', 'target'):
        ANSWER_COL = col
    elif cl in ('solution', 'response', 'output', 'generation'):
        SOLUTION_COL = col

print(f"\n  Question: {QUESTION_COL} | Answer: {ANSWER_COL} | Solution: {SOLUTION_COL}")
assert QUESTION_COL and ANSWER_COL, f"Cannot detect columns from: {ds.column_names}"

eval_set = ds.select(range(0, 5000))
train_set = ds.select(range(5000, len(ds)))
print(f"  Train: {len(train_set)} | Eval: {len(eval_set)}\n")

# ============================================================
# HELPERS
# ============================================================

def fmt(q):
    return f"Please reason step by step, and put your final answer within \\boxed{{}}.\n\n{q}"

def extract(text):
    m = re.findall(r'\\boxed\{([^}]+)\}', text)
    return m[-1].strip() if m else None

def check(pred, gt):
    if pred is None: return False
    try: return abs(float(pred) - float(gt)) < 1e-6
    except: return str(pred).strip() == str(gt).strip()

def gen(prompt, temp=0.7):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    from vllm import SamplingParams
    out = model.fast_generate([text],
        sampling_params=SamplingParams(temperature=temp, top_p=0.95, top_k=20, max_tokens=4096))
    return out[0].outputs[0].text

def evaluate(n=200):
    FastLanguageModel.for_inference(model)
    ok = 0
    total = min(n, len(eval_set))
    for i in range(total):
        try:
            if check(extract(gen(fmt(eval_set[i][QUESTION_COL]), 0.6)), eval_set[i][ANSWER_COL]):
                ok += 1
        except: pass
        if (i+1) % 50 == 0: print(f"    {i+1}/{total} acc={ok/(i+1):.3f}")
    print(f"  Eval: {ok}/{total} = {ok/total:.3f}")
    return ok / total

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
        except: return None

    print("  Testing Gemini...", end=" ")
    t = judge("2+2=4, answer is 4", "4")
    print(f"OK: {t}")
    return judge

# ============================================================
# DPO TRAINING FUNCTION
# ============================================================

def train_round(pairs, round_num, lr, beta):
    print(f"\n  Training round {round_num}: {len(pairs)} pairs, lr={lr}, beta={beta}")
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
    del trainer; gc.collect()

# ============================================================
# GENERATE PAIRS FUNCTION
# ============================================================

def generate_pairs_round1():
    """Round 1: 1 solution per problem, wrong → gold vs rejected."""
    FastLanguageModel.for_inference(model)
    pairs, skip, err = [], 0, 0
    t0 = time.time()

    for i in range(len(train_set)):
        if i % 5000 == 0 and i > 0:
            print(f"  {i}/{len(train_set)} | {len(pairs)} pairs | {i/(time.time()-t0)*3600:.0f}/hr")
        ex = train_set[i]
        try:
            out = gen(fmt(ex[QUESTION_COL]), 0.7)
            if check(extract(out), ex[ANSWER_COL]):
                skip += 1
            else:
                gold = ex.get(SOLUTION_COL) or f"The answer is \\boxed{{{ex[ANSWER_COL]}}}."
                pairs.append({"prompt": fmt(ex[QUESTION_COL]), "chosen": gold, "rejected": out})
        except: err += 1

    print(f"  Done: {len(pairs)} pairs, {skip} skip, {err} err, {(time.time()-t0)/3600:.1f}h")
    return pairs


def generate_pairs_gemini(judge, temp):
    """Rounds 2-3: 2 solutions per problem, Gemini judges ties."""
    FastLanguageModel.for_inference(model)
    pairs, skip, err = [], 0, 0
    t0 = time.time()

    for i in range(len(train_set)):
        if i % 5000 == 0 and i > 0:
            print(f"  {i}/{len(train_set)} | {len(pairs)} pairs | {i/(time.time()-t0)*3600:.0f}/hr")
        ex = train_set[i]
        gt = ex[ANSWER_COL]
        prompt = fmt(ex[QUESTION_COL])

        try:
            a, b = gen(prompt, temp), gen(prompt, temp)
            ok_a, ok_b = check(extract(a), gt), check(extract(b), gt)

            if ok_a and not ok_b:
                pairs.append({"prompt": prompt, "chosen": a, "rejected": b})
            elif ok_b and not ok_a:
                pairs.append({"prompt": prompt, "chosen": b, "rejected": a})
            elif not ok_a and not ok_b:
                gold = ex.get(SOLUTION_COL) or f"The answer is \\boxed{{{gt}}}."
                pairs.append({"prompt": prompt, "chosen": gold, "rejected": a})
            elif ok_a and ok_b:
                ja, jb = judge(a, gt), judge(b, gt)
                sa = ja["score"] if ja else 1.0
                sb = jb["score"] if jb else 1.0
                if abs(sa - sb) >= 0.2:
                    pairs.append({"prompt": prompt,
                                  "chosen": a if sa > sb else b,
                                  "rejected": b if sa > sb else a})
                else: skip += 1
            else: skip += 1
        except: err += 1

    print(f"  Done: {len(pairs)} pairs, {skip} skip, {err} err, {(time.time()-t0)/3600:.1f}h")
    return pairs


# ============================================================
# RUN ALL 3 ROUNDS
# ============================================================

print("\n" + "=" * 60)
print("  BASELINE EVALUATION")
print("=" * 60)
baseline = evaluate(200)

# ── Round 1 ──
print("\n" + "=" * 60)
print("  ROUND 1: Direct DPO (no Gemini)")
print("=" * 60)
pairs = generate_pairs_round1()
train_round(pairs, 1, lr=5e-7, beta=0.10)
score_r1 = evaluate(200)
upload(1)
print(f"  Round 1: {baseline:.3f} → {score_r1:.3f}")

# ── Round 2 ──
print("\n" + "=" * 60)
print("  ROUND 2: Gemini-judged DPO")
print("=" * 60)
judge = setup_gemini()
pairs = generate_pairs_gemini(judge, temp=0.8)
train_round(pairs, 2, lr=3e-7, beta=0.12)
score_r2 = evaluate(200)
upload(2)
print(f"  Round 2: {score_r1:.3f} → {score_r2:.3f}")

# ── Round 3 ──
print("\n" + "=" * 60)
print("  ROUND 3: Gemini-judged DPO (high temp)")
print("=" * 60)
pairs = generate_pairs_gemini(judge, temp=0.9)
train_round(pairs, 3, lr=2e-7, beta=0.15)
score_r3 = evaluate(200)
upload(3)
print(f"  Round 3: {score_r2:.3f} → {score_r3:.3f}")

# ── Summary ──
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
print(f"\n  Update model_path in your submission notebook to the best round.")
print("=" * 60)
