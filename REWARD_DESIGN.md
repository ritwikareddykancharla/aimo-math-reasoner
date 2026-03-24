# 🎯 Reward Function Design — AIMO3 GRPO

## The Core Problem

GRPO learns by **contrast**. Within a group of 8 rollouts for the same problem, it asks:
> "Which responses were better than average? Do more of those."

This means the reward function is not just a score — it's a **learning signal**. Every design decision directly shapes what the model learns to do.

---

## Design Principles

### 1. Variance is Everything

If all 8 rollouts get the same reward, GRPO learns nothing:

```
rewards = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
advantage = (reward - mean) / std
std = 0 → advantage = undefined → gradient = 0 → no learning
```

Every design decision is made to **maximize reward variance** within a group while keeping the signal honest.

### 2. Correctness is Primary, Everything Else is Secondary

```
Correct answer:    always positive
Wrong answer:      always negative
No \boxed{}:       maximum penalty

Nothing overrides this ordering.
```

### 3. Partial Credit Creates a Spectrum, Not Discrete Levels

Original 3-level reward:
```
+1.0  correct
-0.5  wrong
-1.0  no answer
```

Problem: all 8 rollouts often land on the same level → zero variance → no learning.

10-level reward:
```
+1.2 → +1.0 → +0.6 → +0.1 → -0.1 → -0.3 → -0.5 → -0.6 → -0.75 → -1.0
```

Now even a batch where all answers are wrong has rich gradient signal.

---

## Reward Table

| Situation | Reward | Reasoning |
|-----------|--------|-----------|
| Correct + full reasoning + matches gold path | **+1.2** | Best possible — right answer, right method |
| Correct + full reasoning (>300 chars) | **+1.0** | Standard correct |
| Correct + some reasoning (100-300 chars) | **+0.6** | Correct but incomplete work |
| Correct + no reasoning (<100 chars) | **+0.1** | Likely memorized — discourage |
| Wrong + hit >70% gold intermediate steps | **-0.1** | Right track, arithmetic error at end |
| Wrong + hit 40-70% gold steps | **-0.3** | Partial understanding of approach |
| Wrong + hit 10-40% gold steps + work shown | **-0.5** | Some overlap with correct method |
| Wrong + work shown, no gold overlap | **-0.6** | Tried but wrong approach |
| Wrong + minimal work | **-0.75** | Barely tried |
| No `\boxed{}` or no work | **-1.0** | Maximum penalty |

---

## Component 1 — Answer Extraction

### Why Bracket-Counting Not Regex

Simple regex `\boxed{([^}]+)}` fails on nested braces:

```
\boxed{\frac{a+b}{c}}
       ^         ^
       opens     regex stops here, misses rest
```

Bracket-counting handles arbitrary nesting depth:

```python
depth = 1
while depth > 0:
    if char == '{': depth += 1
    elif char == '}': depth -= 1
```

This alone fixes ~10-15% of false negatives on competition math answers.

---

## Component 2 — Answer Normalization

Competition answers appear in many equivalent forms. Without normalization, correct answers are marked wrong:

| Model Output | Ground Truth | Naive Match | After Normalization |
|-------------|--------------|-------------|---------------------|
| `42.0` | `42` | ❌ | ✅ |
| ` 42 ` | `42` | ❌ | ✅ |
| `$42$` | `42` | ❌ | ✅ |
| `\text{42}` | `42` | ❌ | ✅ |

Normalization steps applied in order:
1. Strip whitespace
2. Remove trailing `.0`
3. Remove `$` signs
4. Unwrap `\text{}`
5. Remove `\left` / `\right`
6. Lowercase

---

## Component 3 — Numeric Equivalence

String matching fails even after normalization for mathematically equal expressions:

| Model Output | Ground Truth | String Match | Numeric Match |
|-------------|--------------|-------------|---------------|
| `0.5` | `1/2` | ❌ | ✅ |
| `0.333` | `1/3` | ❌ | ✅ (within tolerance) |
| `1/4` | `0.25` | ❌ | ✅ |

Three-level numeric comparison:
1. Direct `float()` conversion
2. `Fraction()` for exact rationals
3. Manual `a/b` splitting

Relative tolerance `1e-6` handles floating point imprecision.

---

## Component 4 — Correct Answer Length Penalty

### The Memorization Problem

Without a length check, the model can reward-hack by memorizing training answers:

```
Problem: "Find x such that x² + 3x = 10"
Model outputs: "\boxed{2}"   ← 8 chars, reward = +1.0
```

The model gets full reward without showing any reasoning. At test time on novel problems, it fails completely.

### The Fix

```
response_len < 100  → reward = +0.1  (correct but suspicious)
response_len < 300  → reward = +0.6  (some work shown)
response_len ≥ 300  → reward = +1.0  (full reasoning)
```

### Why +0.1 Not 0.0

If reward = 0.0 for lazy-correct:
```
mean of group might be -0.3 (mostly wrong)
advantage of 0.0 = (0.0 - (-0.3)) / std = positive!
Model still incentivized to be lazy.
```

`+0.1` is positive enough to not be penalized for being correct, but low enough that `+1.0` always dominates. The model learns: showing work is always better.

---

## Component 5 — Gold Step Similarity

### What It Measures

Gold solutions contain **intermediate values** — the calculation waypoints on the path to the final answer.

```
Gold solution:
  "Let x = 5. Then x² = 25.
   Adding 3 gives 28. Answer is \boxed{28}"

Intermediate steps = {5, 25}   (28 = final answer, excluded)
```

If the model's response also contains {5, 25}, it was on the right reasoning path — even if it got the final answer wrong.

### Why Numbers Not Words

Text similarity (ROUGE, cosine) compares words. Math solutions can use completely different words but the same calculations:

```
Gold:  "Let the base be 5. The area is 5 × 4 / 2 = 10."
Model: "Setting b=5, we compute A = (1/2)(5)(4) = 10."

Word overlap: low
Number overlap: {5, 4, 10} = 100%
```

Number overlap is the mathematically meaningful signal.

### Step Similarity Score

```
similarity = |gold_steps ∩ response_numbers| / |gold_steps|
```

Threshold at 0.1 to avoid rewarding coincidental matches (common numbers like 2, 3, 4 appear everywhere).

### Depth Coverage

For long multi-step solutions, we also estimate how far into the solution the model got:

```python
step_indicators = count("therefore", "thus", "we get", "=", ...)
depth_coverage = steps_hit / total_depth
```

A model that gets halfway through a 10-step proof deserves more credit than one that gets one step in.

---

## Component 6 — The -0.1 Tier (Most Important)

This is the key innovation. Consider:

```
Model A:
  Followed gold solution exactly
  Hit all 8 intermediate values
  Made arithmetic error on final step
  Final answer: 43 (correct: 42)
  → Current reward: -0.5 (just "wrong + work shown")

Model B:
  Random approach
  Hit 0 intermediate values
  Got lucky with one step
  Final answer: 99
  → Current reward: -0.5 (same!)
```

Model A was *almost right*. GRPO should strongly reinforce its approach.

With the -0.1 tier:
```
Model A: sim = 0.9 → reward = -0.1
Model B: sim = 0.0 → reward = -0.6

Advantage of A vs B = (-0.1 - mean) vs (-0.6 - mean)
= A has much higher advantage
= GRPO strongly reinforces Model A's reasoning path
= Model learns: "follow this calculation structure"
```

---

## What GRPO Learns From Each Tier

| Reward | GRPO Signal | Model Learns |
|--------|-------------|--------------|
| +1.2 | Strong reinforce | "Use this exact approach" |
| +1.0 | Reinforce | "Show full reasoning" |
| +0.6 | Weak reinforce | "Being correct matters most" |
| +0.1 | Near neutral | "Don't skip work" |
| -0.1 | Weak suppress | "Close but fix the last step" |
| -0.3 | Suppress | "Partially right approach" |
| -0.5 | Suppress | "Wrong but keep trying" |
| -1.0 | Strong suppress | "Never do this" |

---

## Failure Modes This Design Prevents

| Failure Mode | Prevention |
|---|---|
| Memorizing training answers | Length penalty (+0.1 for short correct) |
| Ignoring \boxed{} format | -1.0 for missing \boxed{} |
| Zero-variance groups (no learning) | 10-level spectrum ensures variance |
| False negatives from format variation | Normalization + numeric equivalence |
| Penalizing valid alternative solution paths | Similarity is bonus only, not required |
| Reward hacking via long rambling | Similarity bonus requires matching gold VALUES not word count |
| KL divergence explosion | Correctness dominates — model can't hack by drifting far from base |

---

## Hyperparameter Choices

| Parameter | Value | Why |
|-----------|-------|-----|
| Similarity threshold | 0.3 | Below this, number overlap is coincidental |
| Max similarity bonus | +0.2 | Small enough that correctness always dominates |
| Length threshold (full) | 300 chars | ~50 words, enough for 2-3 reasoning steps |
| Length threshold (minimal) | 100 chars | Enough for one sentence of work |
| Step sim tier 1 | 0.7 | >70% gold steps = clearly on right track |
| Step sim tier 2 | 0.4 | 40-70% = partial understanding |
| Numeric tolerance | 1e-6 | Handles floating point, tight enough to avoid false positives |

---

## Files

```
training/grpo/
├── reward_fn.py          ← Full implementation
└── REWARD_DESIGN.md      ← This document
```
