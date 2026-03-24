"""
training/grpo/reward_fn.py

GRPO Reward Function for AIMO3
==============================
10-level reward spectrum with:
  - Nested brace answer extraction
  - Answer normalization + numeric equivalence  
  - Length penalty (anti-memorization)
  - Gold solution step similarity (partial credit)
  - Broken gold solution detection (fallback)
"""

import re
from fractions import Fraction


# ============================================================
# ANSWER EXTRACTION
# ============================================================

def extract_boxed(text):
    """Bracket-counting extractor — handles nested braces."""
    idx = text.find(r'\boxed{')
    if idx == -1:
        idx = text.find(r'\boxed {')
        if idx == -1:
            return None
        start = idx + 8
    else:
        start = idx + 7

    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1

    return text[start:i-1].strip() if depth == 0 else None


def normalize_answer(ans):
    if ans is None:
        return None
    ans = ans.strip()
    ans = re.sub(r'\.0+$', '', ans)
    ans = ans.replace('$', '').strip()
    ans = re.sub(r'\\text\{([^}]+)\}', r'\1', ans)
    ans = ans.replace(r'\left', '').replace(r'\right', '')
    ans = re.sub(r'\s+', '', ans)
    return ans.lower()


def try_numeric_equal(pred, gt, tol=1e-6):
    def to_float(s):
        try:
            return float(s)
        except Exception:
            pass
        try:
            return float(Fraction(s))
        except Exception:
            pass
        try:
            parts = s.split('/')
            if len(parts) == 2:
                return float(parts[0]) / float(parts[1])
        except Exception:
            pass
        return None

    p, g = to_float(pred), to_float(gt)
    if p is not None and g is not None:
        if g == 0:
            return abs(p) < tol
        return abs(p - g) / (abs(g) + 1e-10) < tol
    return None


def answers_match(pred, gt):
    if pred is None or gt is None:
        return False
    pred_norm = normalize_answer(pred)
    gt_norm   = normalize_answer(gt)
    if pred_norm == gt_norm:
        return True
    numeric = try_numeric_equal(pred_norm, gt_norm)
    if numeric is not None:
        return numeric
    pred_clean = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', pred_norm)
    gt_clean   = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', gt_norm)
    if pred_clean == gt_clean:
        return True
    numeric2 = try_numeric_equal(pred_clean, gt_clean)
    if numeric2 is not None:
        return numeric2
    return False


# ============================================================
# GOLD SOLUTION VALIDATION
# ============================================================

BROKEN_SIGNALS = [
    'Timed out',
    'NameError',
    'Traceback',
    'Error:',
    'Exception',
    'not defined',
    'SyntaxError',
    'AttributeError',
    'IndexError',
    'KeyError',
    'RuntimeError',
    'ZeroDivisionError',
]

def is_valid_gold(gold):
    """
    Returns False if gold solution is a broken/failed trace.
    Broken gold = noise, not signal. Fall back to simpler reward.
    
    Stats from our dataset:
      acc=0.125: 36% broken (hardest problems timeout more)
      acc=0.250: 20% broken
      acc=0.375: 12% broken
    """
    if not gold or len(gold) < 50:
        return False
    for signal in BROKEN_SIGNALS:
        if signal in gold:
            return False
    return True


# ============================================================
# STEP SIMILARITY
# ============================================================

def extract_numbers(text):
    """Extract intermediate numeric values from solution text."""
    numbers = set()

    # LaTeX fractions: \frac{a}{b}
    for frac in re.finditer(r'\\frac\{(\d+)\}\{(\d+)\}', text):
        try:
            val = float(frac.group(1)) / float(frac.group(2))
            numbers.add(round(val, 4))
        except Exception:
            pass

    # Plain numbers (not part of variable names)
    for num in re.finditer(r'(?<![a-zA-Z])(\d+\.?\d*)(?![a-zA-Z])', text):
        try:
            val = float(num.group(1))
            if 0 < val <= 9999:      # skip likely non-math numbers
                numbers.add(round(val, 4))
        except Exception:
            pass

    return numbers


def compute_step_similarity(response, gold_solution):
    """
    Fraction of gold intermediate steps that appear in response.
    Final answer is excluded from gold steps (handled by correctness reward).
    
    Returns (similarity: float, depth: float) both in [0, 1]
    """
    all_gold_numbers = extract_numbers(gold_solution)

    # Remove final answer from gold steps
    final = extract_boxed(gold_solution)
    if final:
        try:
            ans_val = round(float(normalize_answer(final)), 4)
            all_gold_numbers.discard(ans_val)
        except Exception:
            pass

    if not all_gold_numbers:
        return 0.0, 0.0

    response_numbers = extract_numbers(response)
    overlap = all_gold_numbers & response_numbers
    similarity = len(overlap) / len(all_gold_numbers)

    # Depth: how many reasoning indicators in gold vs response
    gold_depth = max(1, len(re.findall(
        r'(therefore|thus|so |hence|we get|we have|gives|equals|=)',
        gold_solution.lower()
    )))
    depth_coverage = min(1.0, len(overlap) / gold_depth)

    return similarity, depth_coverage


# ============================================================
# MAIN REWARD FUNCTION
# ============================================================

def compute_reward(responses, ground_truths, gold_solutions=None, **kwargs):
    """
    GRPO reward function for AIMO3 math competition problems.

    Reward spectrum (10 levels):
    ┌─────────────────────────────────────────────────┬────────┐
    │ Correct + full reasoning + matches gold path    │  +1.2  │
    │ Correct + full reasoning (>300 chars)           │  +1.0  │
    │ Correct + some reasoning (100-300 chars)        │  +0.6  │
    │ Correct + no reasoning (<100 chars, memorized)  │  +0.1  │
    ├─────────────────────────────────────────────────┼────────┤
    │ Wrong + >70% gold steps (right track, arith err)│  -0.1  │
    │ Wrong + 40-70% gold steps                       │  -0.3  │
    │ Wrong + 10-40% gold steps + work shown          │  -0.5  │
    │ Wrong + work shown, no gold overlap             │  -0.6  │
    │ Wrong + minimal work                            │  -0.75 │
    │ No \\boxed{} or gave up                         │  -1.0  │
    └─────────────────────────────────────────────────┴────────┘

    gold_solutions: optional list of gold solution strings
        If None or broken → falls back to simpler reward
        VERL passes this automatically from reward_model dict
    """

    has_gold = (
        gold_solutions is not None
        and len(gold_solutions) == len(responses)
    )

    rewards = []

    for i, (response, gt) in enumerate(zip(responses, ground_truths)):

        pred         = extract_boxed(response)
        response_len = len(response)

        # ── No \boxed{} ─────────────────────────────────────────
        if pred is None:
            rewards.append(-1.0)
            continue

        # ── CORRECT ─────────────────────────────────────────────
        if answers_match(pred, str(gt)):

            if response_len < 100:
                # Too short — likely memorized answer
                reward = 0.1

            elif response_len < 300:
                # Some work shown but not full reasoning
                reward = 0.6

            else:
                # Full reasoning shown
                reward = 1.0

                # Similarity bonus: did model follow gold path?
                gold = gold_solutions[i] if has_gold else None
                if gold and is_valid_gold(gold):
                    sim, _ = compute_step_similarity(response, gold)
                    if sim > 0.5:
                        reward += 0.2 * sim   # up to +0.2 bonus

        # ── WRONG ───────────────────────────────────────────────
        else:
            has_reasoning = response_len > 200
            has_steps     = any(
                kw in response.lower()
                for kw in [
                    'therefore', 'thus', 'so ', 'we have',
                    'note that', 'since', 'because', '=',
                    'hence', 'gives', 'compute', 'calculate',
                ]
            )

            # Get gold similarity if available and valid
            gold = gold_solutions[i] if has_gold else None
            use_sim = gold and is_valid_gold(gold) and has_reasoning

            if use_sim:
                sim, depth = compute_step_similarity(response, gold)

                if sim >= 0.7:
                    # Right track — hit most intermediate steps
                    # Small arithmetic error at end
                    # GRPO should strongly reinforce this path
                    reward = -0.1

                elif sim >= 0.4:
                    # Partial understanding of correct approach
                    reward = -0.3

                elif sim >= 0.1 and has_steps:
                    # Some overlap, showed reasoning
                    reward = -0.5

                else:
                    # Showed work but completely different approach
                    reward = -0.6

            elif has_reasoning and has_steps:
                # No valid gold — fallback to work-based reward
                reward = -0.6

            elif has_reasoning:
                reward = -0.75

            else:
                # Gave up — minimal output
                reward = -1.0

        rewards.append(float(reward))

    return rewards
