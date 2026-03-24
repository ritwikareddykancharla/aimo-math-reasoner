"""
AIMO3: Eval script for SFT checkpoint
======================================
Runs pass@1 on held-out Nemotron Math v2 eval problems.

Usage:
    python3 eval_sft.py --model ./merged-4bit-step-500
    python3 eval_sft.py --model ritwikakancharla/gpt-oss-120b-aimo3-sft-merged
    python3 eval_sft.py --model ./merged-4bit-step-500 --n_problems 100 --n_samples 8
"""

import unsloth  # noqa — must be first

import os, re, json, argparse, time
from datasets import load_from_disk
from unsloth import FastLanguageModel
import torch

# ============================================================
# ARGS
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument("--model",      type=str, required=True,
                    help="Path or Kaggle handle to merged 4bit model")
parser.add_argument("--data_dir",   type=str, default="./data/nemotron-math-v2")
parser.add_argument("--n_problems", type=int, default=50,
                    help="Number of eval problems to test")
parser.add_argument("--n_samples",  type=int, default=1,
                    help="Samples per problem (1=pass@1, 8=majority@8)")
parser.add_argument("--temperature",type=float, default=0.7)
parser.add_argument("--max_tokens", type=int, default=2048)
parser.add_argument("--output",     type=str, default="eval_results.json")
args = parser.parse_args()

# ============================================================
# LOAD MODEL
# ============================================================

print(f"\n  Loading model: {args.model}")
t0 = time.time()

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=args.model,
    max_seq_length=args.max_tokens + 512,
    load_in_4bit=True,
    fast_inference=True,   # vLLM for faster inference
    gpu_memory_utilization=0.85,
)
FastLanguageModel.for_inference(model)
print(f"  Model loaded in {(time.time()-t0)/60:.1f} mins\n")

# ============================================================
# LOAD EVAL DATA
# ============================================================

print(f"  Loading eval data from {args.data_dir}...")
ds = load_from_disk(args.data_dir)
ds = ds.shuffle(seed=42)

# same split as training — first 2000 are eval
eval_data = ds.select(range(0, min(2000, len(ds))))
eval_data = eval_data.select(range(min(args.n_problems, len(eval_data))))
print(f"  Eval problems: {len(eval_data)}\n")

# ============================================================
# ANSWER EXTRACTION
# ============================================================

def extract_answer(text: str):
    """Extract integer answer from \\boxed{} — takes the last one."""
    matches = re.findall(r'\\boxed\{([^}]+)\}', text)
    if not matches:
        return None
    raw = matches[-1].strip()
    # clean: remove commas, spaces, dollar signs
    raw = re.sub(r'[$,\s]', '', raw)
    # handle simple fractions like \frac{4}{2}
    frac = re.match(r'\\frac\{(\d+)\}\{(\d+)\}', raw)
    if frac:
        num, den = int(frac.group(1)), int(frac.group(2))
        if den != 0 and num % den == 0:
            return num // den
        return None
    try:
        val = int(float(raw))
        if 0 <= val <= 999999:
            return val
        return None
    except:
        return None

def majority_vote(answers: list):
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    return max(set(valid), key=valid.count)

def vote_confidence(answers: list, winner) -> float:
    valid = [a for a in answers if a is not None]
    if not valid or winner is None:
        return 0.0
    return valid.count(winner) / len(valid)

# ============================================================
# INFERENCE
# ============================================================

def solve(problem: str, n_samples: int, temperature: float) -> tuple:
    """Returns (predicted_answer, all_answers, confidence)."""

    # build prompt using chat template
    msgs = [{"role": "user", "content": problem}]
    prompt = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    answers = []
    for _ in range(n_samples):
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                repetition_penalty=1.1,
            )
        # decode only the new tokens
        new_tokens = out[0][inputs['input_ids'].shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        answers.append(extract_answer(text))

    winner     = majority_vote(answers)
    confidence = vote_confidence(answers, winner)
    return winner, answers, confidence

# ============================================================
# RUN EVAL
# ============================================================

print("=" * 60)
print(f"  EVAL: {len(eval_data)} problems | "
      f"{args.n_samples} sample(s) each | "
      f"temp={args.temperature}")
print("=" * 60)

results  = []
correct  = 0
skipped  = 0   # no valid answer extracted
t_start  = time.time()

for i, example in enumerate(eval_data):
    problem  = example['messages'][0]['content']
    gold_ans = extract_answer(example['messages'][1]['content'])

    if gold_ans is None:
        skipped += 1
        continue

    t_prob = time.time()
    pred_ans, all_answers, confidence = solve(
        problem, args.n_samples, args.temperature
    )
    elapsed = time.time() - t_prob

    is_correct = (pred_ans == gold_ans)
    if is_correct:
        correct += 1

    results.append({
        "idx":        i,
        "problem":    problem[:200],   # truncate for readability
        "gold":       gold_ans,
        "pred":       pred_ans,
        "correct":    is_correct,
        "all_answers": all_answers,
        "confidence": round(confidence, 3),
        "time_s":     round(elapsed, 1),
    })

    # live progress
    total_so_far = i + 1 - skipped
    acc_so_far   = correct / total_so_far if total_so_far > 0 else 0
    eta          = (time.time() - t_start) / total_so_far * (len(eval_data) - total_so_far)
    print(f"  [{i+1:3d}/{len(eval_data)}] "
          f"gold={gold_ans:6d} pred={str(pred_ans):6s} "
          f"{'✅' if is_correct else '❌'} "
          f"conf={confidence:.0%} "
          f"acc={acc_so_far:.1%} "
          f"eta={eta/60:.0f}min")

# ============================================================
# SUMMARY
# ============================================================

total_valid = len(results)
accuracy    = correct / total_valid if total_valid > 0 else 0
total_time  = time.time() - t_start

print("\n" + "=" * 60)
print("  RESULTS")
print("=" * 60)
print(f"  Problems:  {total_valid} (skipped {skipped} with no gold answer)")
print(f"  Correct:   {correct}/{total_valid}")
print(f"  Accuracy:  {accuracy:.1%}")
print(f"  Samples:   {args.n_samples} per problem")
print(f"  Time:      {total_time/60:.1f} mins")
print(f"  Per prob:  {total_time/total_valid:.1f}s")
print("=" * 60)

# breakdown by confidence
high_conf   = [r for r in results if r['confidence'] >= 0.7]
low_conf    = [r for r in results if r['confidence'] <  0.4]
high_correct = sum(r['correct'] for r in high_conf)
low_correct  = sum(r['correct'] for r in low_conf)

print(f"\n  High confidence (≥70%): {high_correct}/{len(high_conf)} correct")
print(f"  Low  confidence (<40%): {low_correct}/{len(low_conf)} correct")
print(f"  → Low confidence problems are where majority voting helps most\n")

# save results
with open(args.output, "w") as f:
    json.dump({
        "model":      args.model,
        "n_problems": total_valid,
        "n_samples":  args.n_samples,
        "correct":    correct,
        "accuracy":   accuracy,
        "results":    results,
    }, f, indent=2)
print(f"  Results saved: {args.output}")
