# 🧠 AIMO3 Competition Strategy: Iterative DPO with Gemini Judge

## Overview

| Component | Choice |
|---|---|
| **Base Model** | Qwen3-32B (bf16 for training, FP8 for inference) |
| **Dataset** | Nemotron Math v2 Filtered High (440K AoPS competition math) |
| **Method** | Iterative Online DPO — 3 rounds, all 440K problems each round |
| **Judge** | Gemini 2.5 Flash via Vertex AI (async, 50 concurrent calls) |
| **Gold Fallback** | Nemotron gpt-oss-120B trajectories (already in dataset) |
| **Framework** | Unsloth + TRL + vLLM |
| **Hardware** | 1× NVIDIA H200 (141GB HBM3) |
| **Total Time** | ~42 hours (~1.75 days) |
| **Gemini Cost** | ~$100–150 (Flash, Rounds 2–3 only) |

---

## Pipeline: Round 1 (Fast, No Gemini)

```
440K problems
    │
    ▼
Step 1: vLLM generates 1 solution per problem (~3.5 hrs, GPU)
    │
    ▼
Step 2: Regex check — is the answer correct? (instant)
    │
    ├── Correct? → SKIP (nothing to learn)
    │
    └── Wrong? → DPO pair:
            chosen  = Nemotron 120B gold trajectory (from dataset)
            rejected = model's wrong attempt
    │
    ▼
Step 3: DPO train, 1 epoch (~7 hrs, GPU)
    │
    ▼
Step 4: Merge LoRA + upload to Kaggle (30 min)
```

Round 1 always uses Nemotron gold as chosen. No Gemini needed. Only 1 generation per problem (not 2), so generation is 2× faster.

## Pipeline: Rounds 2–3 (Gemini Judge, Model vs Model)

```
440K problems
    │
    ▼
Step 1: vLLM generates 2 solutions per problem (~7 hrs, GPU)
    │
    ▼ (overlapped with Step 1)
Step 2: Gemini Flash judges all solutions (2-3 hrs, async API)
        → "answer_correct": true/false
        → "reasoning_valid": true/false
        → "score": 0.0 to 1.0
    │
    ▼
Step 3: Build DPO pairs from scores (instant)
        → One right, one wrong   → chosen=correct, rejected=wrong
        → Both right, diff quality → Gemini picks better reasoning
        → Both wrong              → chosen=Nemotron gold, rejected=worst
        → Same score (within 0.1) → skip
    │
    ▼
Step 4: DPO train, 1 epoch (5-7 hrs, GPU)
    │
    ▼
Step 5: Merge LoRA + upload to Kaggle (30 min)
    │
    ▼
Next round (model is better, different errors)
```

---

## Timeline

| | Generate | Judge | Train | Merge + Upload | Total |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Round 1** | 3.5 hrs (1 sol each) | none (regex only) | 7 hrs | 30 min | ~11 hrs |
| **Round 2** | 7 hrs (2 sol each) | 2.5 hrs (overlapped) | 6 hrs | 30 min | ~16 hrs |
| **Round 3** | 7 hrs (2 sol each) | 2 hrs (overlapped) | 5 hrs | 30 min | ~14.5 hrs |
| **Total** | | | | | **~42 hrs** |

Round 1 is fast: 1 generation per problem, no Gemini, always use Nemotron gold as chosen. Rounds 2–3 generate 2 solutions and use Gemini to judge.

---

## Round Configuration

| Setting | Round 1 | Round 2 | Round 3 |
|---|:---:|:---:|:---:|
| Temperature | 0.7 | 0.8 | 0.9 |
| Learning rate | 5e-7 | 3e-7 | 2e-7 |
| DPO beta | 0.10 | 0.12 | 0.15 |
| Epochs per round | 1 | 1 | 1 |
| Expected model error rate | ~70% | ~50% | ~35% |
| Expected DPO pairs | ~250K | ~180K | ~120K |

Temperature increases each round to generate more diverse solutions from the stronger model. Learning rate decreases to avoid overwriting gains from previous rounds. Beta increases to tighten the KL constraint as the model gets closer to optimal.

---

## What Gemini Catches That Regex Doesn't

```
Case 1: Right answer + right reasoning
  Regex:  reward 1.0
  Gemini: score 1.0
  → Same. No difference.

Case 2: Wrong answer
  Regex:  reward 0.0
  Gemini: score 0.0
  → Same. No difference.

Case 3: Right answer + WRONG reasoning (the dangerous case)
  Regex:  reward 1.0  ← WRONG! Reinforces bad reasoning
  Gemini: score 0.5   ← Catches it, partial credit only

Case 4: Wrong answer + GOOD reasoning (arithmetic slip)
  Regex:  reward 0.0  ← Harsh, throws away good reasoning
  Gemini: score 0.3   ← Partial credit, learns from near-misses
```

Cases 3 and 4 are why Gemini is worth the $150–200.

---

## DPO Pair Construction Logic

```
For each problem, model generates Solution A and Solution B:

score_a, score_b = Gemini scores (0.0 to 1.0)

if abs(score_a - score_b) < 0.1:
    → SKIP (too close, no clear signal)

elif score_a > score_b:
    → PAIR: chosen=A, rejected=B

elif score_b > score_a:
    → PAIR: chosen=B, rejected=A

if score_a < 0.3 and score_b < 0.3:
    → PAIR: chosen=Nemotron 120B gold, rejected=worst of A/B
```

---

## What "Iterative" Means

The model improves each round. Its mistakes change. Same problems, different errors.

```
Round 1: Model is weak, fails 70%  → 250K pairs, big improvement
Round 2: Model is better, fails 50% → 180K pairs, different errors
Round 3: Model is strong, fails 35% → 120K pairs, hardest problems only
```

If you trained 3 epochs on the same Round 1 pairs, the model would overfit to those specific errors. Iterative means re-generating with the improved model each round so every pair is fresh and relevant.

---

## Kaggle Upload After Each Round

After each round, merge LoRA and upload so you have 3 checkpoints to choose from on the leaderboard:

```
Kaggle Models:
  your-username/qwen3-32b-aimo3-round1  ← baseline
  your-username/qwen3-32b-aimo3-round2  ← stronger
  your-username/qwen3-32b-aimo3-round3  ← strongest (usually)
```

Submit all 3 to the leaderboard and pick the best scorer.

---

## Full Implementation

### Step 0: Environment Setup

```python
# Install
# pip install --upgrade unsloth unsloth_zoo
# pip install trl datasets google-genai vllm

import os
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
```

### Step 1: Load Model

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-32B",
    max_seq_length=8192,
    load_in_fp8=True,
    fast_inference=True,
    max_lora_rank=64,
    gpu_memory_utilization=0.92,
)
model = FastLanguageModel.get_peft_model(
    model, r=64,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=64,
    use_gradient_checkpointing="unsloth",
)
```

### Step 2: Load Dataset

```python
from datasets import load_dataset

ds = load_dataset("your-username/nemotron-math-v2-filtered-high")
eval_set = ds.select(range(0, 5000))
train_set = ds.select(range(5000, len(ds)))  # ~435K

def format_math_prompt(question):
    return (
        "Please reason step by step, and put your final "
        "answer within \\boxed{}.\n\n" + question
    )
```

### Step 3: Helper Functions

```python
import re, json

def extract_boxed_answer(text):
    matches = re.findall(r'\\boxed\{([^}]+)\}', text)
    return matches[-1].strip() if matches else None

def check_answer(predicted, ground_truth):
    if predicted is None:
        return False
    try:
        return abs(float(predicted) - float(ground_truth)) < 1e-6
    except (ValueError, TypeError):
        return predicted.strip() == str(ground_truth).strip()
```

### Step 4: Gemini Judge (Async, Parallel)

```python
import asyncio
import os
from google import genai
from google.genai.types import HttpOptions

# Set environment variables for Vertex AI
os.environ["GOOGLE_CLOUD_PROJECT"] = "your-gcp-project"
os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

client = genai.Client(http_options=HttpOptions(api_version="v1"))

JUDGE_PROMPT = """Math problem answer: {ground_truth}

Solution: {solution}

Is the answer correct AND is the reasoning valid?
Reply ONLY with JSON:
{{"answer_correct": true/false, "reasoning_valid": true/false, "score": 0.0-1.0}}

Scoring:
1.0 = correct answer + sound reasoning
0.5 = correct answer but flawed/lucky reasoning
0.3 = wrong answer but good partial reasoning
0.0 = wrong answer + bad reasoning"""


def judge_one(item):
    """Synchronous single judgment"""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=JUDGE_PROMPT.format(
                ground_truth=item["ground_truth"],
                solution=item["solution"][:3000],
            ),
        )
        return json.loads(response.text)
    except:
        return None


async def judge_batch(items, concurrency=50):
    """
    Parallel judging using asyncio + thread pool.
    google-genai SDK is synchronous, so we wrap in threads.
    50 concurrent calls → 880K items in ~2-3 hours.
    """
    semaphore = asyncio.Semaphore(concurrency)
    loop = asyncio.get_event_loop()

    async def judge_one_async(item):
        async with semaphore:
            return await loop.run_in_executor(None, judge_one, item)

    tasks = [judge_one_async(item) for item in items]
    return await asyncio.gather(*tasks)
```

### Step 5: The Main Loop (3 Rounds)

```python
from vllm import LLM, SamplingParams
from trl import DPOTrainer, DPOConfig
from datasets import Dataset

round_configs = [
    {"temp": 0.7, "lr": 5e-7, "beta": 0.10, "n": 1, "use_gemini": False},
    {"temp": 0.8, "lr": 3e-7, "beta": 0.12, "n": 2, "use_gemini": True},
    {"temp": 0.9, "lr": 2e-7, "beta": 0.15, "n": 2, "use_gemini": True},
]

for round_num, cfg in enumerate(round_configs):
    print(f"\n{'='*60}")
    print(f"  ROUND {round_num + 1} / 3")
    print(f"{'='*60}\n")

    # ── GENERATE ──────────────────────────────
    FastLanguageModel.for_inference(model)

    prompts = [
        format_math_prompt(ex["question"]) for ex in train_set
    ]

    sampling = SamplingParams(
        temperature=cfg["temp"],
        top_p=0.95,
        top_k=20,
        max_tokens=4096,
        n=cfg["n"],  # Round 1: 1 solution, Rounds 2-3: 2 solutions
    )

    print(f"  Generating {cfg['n']} solution(s) per problem...")
    all_outputs = llm.generate(prompts, sampling)

    # ── ROUND 1: Always use Nemotron gold as chosen ──
    if not cfg["use_gemini"]:
        pairs = []
        skipped = 0

        for i, output in enumerate(all_outputs):
            example = train_set[i]
            prompt = prompts[i]
            sol = output.outputs[0].text

            pred = extract_boxed_answer(sol)
            if check_answer(pred, example["answer"]):
                skipped += 1  # Model got it right, skip
                continue

            # Wrong → pair with Nemotron 120B gold
            pairs.append({
                "prompt": prompt,
                "chosen": example["solution"],  # Always Nemotron gold
                "rejected": sol,                # Model's wrong attempt
            })

        print(f"  Round 1: {len(pairs)} pairs, skipped {skipped} correct")

    # ── ROUNDS 2-3: Gemini judges model vs model ─────
    else:
        # Build items for Gemini
        judge_items = []
        for i, output in enumerate(all_outputs):
            example = train_set[i]
            for j, completion in enumerate(output.outputs):
                judge_items.append({
                    "problem_idx": i,
                    "solution_idx": j,
                    "solution": completion.text,
                    "ground_truth": example["answer"],
                })

        print(f"  Judging {len(judge_items)} solutions with Gemini Flash...")
        judgments = asyncio.run(judge_batch(judge_items, concurrency=50))

        # Build pairs from Gemini scores
        pairs = []
        skipped = 0

        for i in range(len(all_outputs)):
            example = train_set[i]
            prompt = prompts[i]

            sol_a = all_outputs[i].outputs[0].text
            sol_b = all_outputs[i].outputs[1].text

            idx_a = i * 2
            idx_b = i * 2 + 1

            # Get Gemini scores (with regex fallback)
            if judgments[idx_a] is not None:
                score_a = judgments[idx_a]["score"]
            else:
                pred_a = extract_boxed_answer(sol_a)
                score_a = 1.0 if check_answer(pred_a, example["answer"]) else 0.0

            if judgments[idx_b] is not None:
                score_b = judgments[idx_b]["score"]
            else:
                pred_b = extract_boxed_answer(sol_b)
                score_b = 1.0 if check_answer(pred_b, example["answer"]) else 0.0

            # Both bad → use Nemotron gold
            if score_a < 0.3 and score_b < 0.3:
                pairs.append({
                    "prompt": prompt,
                    "chosen": example["solution"],
                    "rejected": sol_a if score_a <= score_b else sol_b,
                })
                continue

            # Too close → skip
            if abs(score_a - score_b) < 0.1:
                skipped += 1
                continue

            # Clear winner
            if score_a > score_b:
                pairs.append({
                    "prompt": prompt, "chosen": sol_a, "rejected": sol_b,
                })
            else:
                pairs.append({
                    "prompt": prompt, "chosen": sol_b, "rejected": sol_a,
                })

        print(f"  Round {round_num+1}: {len(pairs)} pairs, skipped {skipped}")

    # ── TRAIN DPO ─────────────────────────────
    FastLanguageModel.for_training(model)

    trainer = DPOTrainer(
        model=model,
        args=DPOConfig(
            output_dir=f"dpo-round{round_num + 1}",
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            num_train_epochs=1,
            learning_rate=cfg["lr"],
            beta=cfg["beta"],
            bf16=True,
            max_length=8192,
            max_prompt_length=2048,
            packing=True,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            save_steps=5000,
            logging_steps=20,
            max_grad_norm=1.0,
        ),
        train_dataset=Dataset.from_list(pairs),
        tokenizer=tokenizer,
    )
    trainer.train()

    # ── EVALUATE ──────────────────────────────
    FastLanguageModel.for_inference(model)
    correct = 0
    for ex in eval_set:
        prompt = format_math_prompt(ex["question"])
        output = generate_single(model, tokenizer, prompt)
        pred = extract_boxed_answer(output)
        if check_answer(pred, ex["answer"]):
            correct += 1
    score = correct / len(eval_set)
    print(f"  Round {round_num + 1} pass@1: {score:.3f}")

    # ── MERGE & UPLOAD TO KAGGLE ──────────────
    model.save_pretrained_merged(
        f"qwen3-32b-aimo3-round{round_num + 1}",
        tokenizer,
        save_method="merged_16bit",
    )

    import subprocess
    subprocess.run([
        "kaggle", "models", "create",
        "-p", f"qwen3-32b-aimo3-round{round_num + 1}",
    ])

    print(f"  Round {round_num + 1} uploaded to Kaggle ✅")
```

---

## Cost Summary

| Item | Cost |
|---|---:|
| Gemini Flash — Round 1 | $0 (no Gemini, regex only) |
| Gemini Flash — Round 2 (~880K judgments) | ~$65 |
| Gemini Flash — Round 3 (~700K judgments) | ~$50 |
| H200 compute (~42 hrs × ~$3-5/hr) | ~$125–210 |
| **Total** | **~$240–325** |

---

## Expected Results Per Round

| Round | Model Error Rate | DPO Pairs | Expected pass@1 Gain |
|:---:|:---:|:---:|:---:|
| Baseline (no training) | ~70% | — | — |
| After Round 1 | ~50% | ~250K | +15–20% |
| After Round 2 | ~35% | ~180K | +5–10% |
| After Round 3 | ~25% | ~120K | +3–5% |

---

## Kaggle Outputs

After training completes, you'll have 3 models on Kaggle:

```
your-username/qwen3-32b-aimo3-round1   (65 GB, baseline DPO)
your-username/qwen3-32b-aimo3-round2   (65 GB, stronger)
your-username/qwen3-32b-aimo3-round3   (65 GB, strongest)
```

Submit all 3 to the AIMO3 leaderboard and pick the highest scorer.

---

## Key Design Decisions

**Why Qwen3-32B, not 30B-A3B?**
32B dense is better for DPO fine-tuning. All parameters active during every forward pass. MoE router fine-tuning is unstable.

**Why DPO, not GRPO?**
GRPO has known stability issues at 32B scale — model collapse, repetitive outputs, reward destabilization. DPO is supervised learning on pairs — zero collapse risk.

**Why Gemini Flash, not Pro?**
Flash is 15× cheaper and 3× faster. For binary judging ("is answer correct + is reasoning valid"), Flash is more than sufficient.

**Why iterative (3 rounds), not 3 epochs on static pairs?**
Same data for 3 epochs → overfitting to specific errors. Re-generating each round → fresh pairs from the improved model → different errors → better generalization.

**Why generate 1 solution in Round 1, then 2 in Rounds 2–3?**
Round 1 always uses Nemotron gold as chosen — you only need the model's wrong attempt as rejected. One generation is enough. Rounds 2–3 compare two model outputs against each other (with Gemini judging quality), so you need 2. This saves ~3.5 hours in Round 1.

**Why merge + upload after each round?**
Insurance. If Round 3 somehow degrades (unlikely with DPO but possible), you still have Round 2 checkpoint on Kaggle ready to submit.
