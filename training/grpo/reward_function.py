"""
training/grpo/reward_function.py

Reward requires BOTH correct answer AND reasoning that matches
the gold solution trajectory. Correct answer alone = low reward.
This prevents memorization of answers without understanding.
"""

import re
import os
import time
import hashlib
import json
from fractions import Fraction
from pathlib import Path

CACHE_DIR = Path("/tmp/gemini_reward_cache")
CACHE_DIR.mkdir(exist_ok=True)

try:
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    GEMINI_MODEL = genai.GenerativeModel("gemini-2.0-flash")
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False


# ============================================================
# ANSWER EXTRACTION + MATCHING
# ============================================================

def extract_boxed(text):
    text = re.sub(
        r'<parameter name="think">.*?</parameter>',
        '', text, flags=re.DOTALL
    )
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


def answers_match(pred, gt, tol=1e-6):
    if pred is None or gt is None:
        return False
    pred_norm = normalize_answer(pred)
    gt_norm   = normalize_answer(gt)
    if pred_norm == gt_norm:
        return True
    p, g = to_float(pred_norm), to_float(gt_norm)
    if p is not None and g is not None:
        if g == 0:
            return abs(p) < tol
        return abs(p - g) / (abs(g) + 1e-10) < tol
    def defrac(s):
        return re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', s)
    pred_c = defrac(pred_norm)
    gt_c   = defrac(gt_norm)
    if pred_c == gt_c:
        return True
    p2, g2 = to_float(pred_c), to_float(gt_c)
    if p2 is not None and g2 is not None:
        if g2 == 0:
            return abs(p2) < tol
        return abs(p2 - g2) / (abs(g2) + 1e-10) < tol
    return False


# ============================================================
# GOLD SOLUTION VALIDATION
# ============================================================

BROKEN_SIGNALS = [
    'Timed out', 'NameError', 'Traceback', 'Error:',
    'Exception', 'not defined', 'SyntaxError',
    'AttributeError', 'IndexError', 'KeyError',
    'RuntimeError', 'ZeroDivisionError',
]

def is_valid_gold(gold):
    if not gold or len(gold) < 50:
        return False
    for signal in BROKEN_SIGNALS:
        if signal in gold:
            return False
    return True


# ============================================================
# GEMINI FULL JUDGE
# Evaluates BOTH answer correctness AND reasoning quality
# Called for ALL responses with substantial content
# ============================================================

FULL_JUDGE_PROMPT = """\
You are a math olympiad judge evaluating a student's complete solution.

PROBLEM:
{problem}

GOLD SOLUTION (correct answer + correct approach):
{gold_solution}

STUDENT SOLUTION:
{student_solution}

Evaluate TWO things independently:

━━━ PART 1: ANSWER ━━━
Is the student's final boxed answer correct?
Compare to the gold solution's final answer.
answer_correct: true | false

━━━ PART 2: REASONING TRAJECTORY ━━━
Does the student's reasoning follow the same mathematical approach as the gold solution?
Ignore whether the final answer is right or wrong for this part.

Focus on:
- Same core method? (induction, construction, casework, algebraic manipulation, etc.)
- Same key insight or setup identified?
- Same intermediate steps or transformations?
- All major steps present and logically connected?

Classify the reasoning:

"complete":
  Same approach as gold AND all major steps present AND logically sound.
  A correct answer with this reasoning = genuine understanding, not memorization.

"correct_approach_incomplete":
  Right method, right key insight, but missing some intermediate steps.
  Could be correct understanding with sloppy writeup.

"correct_approach_wrong_execution":
  Right method, right setup, but calculation/algebra error somewhere.
  Understanding is there, execution failed.

"adjacent_approach":
  Different valid method that could also solve the problem.
  Not the gold approach but mathematically legitimate.

"wrong_approach":
  Fundamentally wrong method. Cannot lead to correct answer.
  Even if answer happens to be right, it's a coincidence.

"memorized":
  Answer appears with minimal or no reasoning.
  Less than 3 meaningful mathematical steps shown.
  Looks like a memorized answer, not derived reasoning.

"incoherent":
  No meaningful mathematical content. Gave up or produced nonsense.

━━━ RESPOND WITH ONLY THIS JSON ━━━
{
  "answer_correct": true | false,
  "reasoning_class": "complete" | "correct_approach_incomplete" | "correct_approach_wrong_execution" | "adjacent_approach" | "wrong_approach" | "memorized" | "incoherent",
  "key_insight_present": true | false,
  "reason": "one sentence identifying what is correct or wrong about the approach"
}
"""


def get_cache_key(solution: str, gold: str) -> str:
    content = f"{gold[:200]}||{solution[:400]}"
    return hashlib.md5(content.encode()).hexdigest()


def gemini_full_judge(
    problem: str,
    student_solution: str,
    gold_solution: str,
) -> dict:
    """
    Full evaluation: answer correctness + reasoning trajectory.
    Returns dict with answer_correct, reasoning_class, key_insight_present.
    Falls back to safe defaults on any error.
    """
    FALLBACK = {
        "answer_correct": False,
        "reasoning_class": "wrong_approach",
        "key_insight_present": False,
    }

    if not GEMINI_AVAILABLE:
        return FALLBACK

    cache_key  = get_cache_key(student_solution, gold_solution)
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text())
        except Exception:
            pass

    prompt = FULL_JUDGE_PROMPT.format(
        problem=problem or "Math olympiad problem",
        gold_solution=gold_solution[:2500],
        student_solution=student_solution[:2500],
    )

    for attempt in range(3):
        try:
            response = GEMINI_MODEL.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=200,
                )
            )
            raw = response.text.strip()
            raw = re.sub(r'^```json\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)
            result = json.loads(raw)

            # Validate required fields exist
            assert "answer_correct" in result
            assert "reasoning_class" in result
            assert "key_insight_present" in result

            cache_file.write_text(json.dumps(result))
            return result

        except Exception:
            if attempt < 2:
                time.sleep(2 ** attempt)
            continue

    return FALLBACK


# ============================================================
# REWARD TABLE
# ============================================================

def compute_reward_from_judgment(judgment: dict) -> float:
    """
    Reward requires BOTH correct answer AND correct reasoning.
    Correct answer alone is not enough — prevents memorization.

    Full reward table:
    ┌─────────────────────────────────────────────────────┬──────┐
    │ Correct answer + complete reasoning                 │ 1.0  │
    │ Correct answer + correct approach, incomplete steps │ 0.7  │
    │ Correct answer + correct approach, execution error  │ 0.5  │
    │   (answer right but reasoning had error = lucky)    │      │
    │ Correct answer + adjacent valid approach            │ 0.6  │
    │ Correct answer + wrong approach (MEMORIZED)         │ 0.1  │
    │ Correct answer + memorized (no steps)               │ 0.05 │
    ├─────────────────────────────────────────────────────┼──────┤
    │ Wrong answer  + complete reasoning (calc error)     │ 0.4  │
    │ Wrong answer  + correct approach, incomplete        │ 0.25 │
    │ Wrong answer  + correct approach, wrong execution   │ 0.2  │
    │ Wrong answer  + adjacent approach                   │ 0.15 │
    │ Wrong answer  + wrong approach                      │ 0.05 │
    │ Wrong answer  + memorized / incoherent              │ 0.0  │
    └─────────────────────────────────────────────────────┴──────┘

    Key design decisions:
    1. Correct answer + memorized = 0.05 (near zero, not rewarded)
       This is the core anti-memorization mechanism.
    2. Wrong answer + complete reasoning = 0.4
       Right track with calculation error should be encouraged.
    3. Gap between correct+complete (1.0) and correct+memorized (0.05)
       is huge — model learns reasoning IS the target, not the answer.
    """
    answer_correct   = judgment.get("answer_correct", False)
    reasoning_class  = judgment.get("reasoning_class", "incoherent")
    key_insight      = judgment.get("key_insight_present", False)

    # ── Correct answer ───────────────────────────────────────
    if answer_correct:
        if reasoning_class == "complete":
            return 1.0

        elif reasoning_class == "correct_approach_incomplete":
            # Right answer, right approach, missing steps
            # Good but not perfect — encourage completeness
            return 0.7

        elif reasoning_class == "correct_approach_wrong_execution":
            # Right answer despite reasoning error = somewhat lucky
            # Still reward but less than clean solution
            return 0.5

        elif reasoning_class == "adjacent_approach":
            # Valid different method, correct answer
            return 0.6

        elif reasoning_class == "wrong_approach":
            # Got answer right but approach is wrong = memorized
            # This is the key anti-memorization penalty
            return 0.1

        elif reasoning_class in ("memorized", "incoherent"):
            # No reasoning at all = pure memorization
            return 0.05

    # ── Wrong answer ─────────────────────────────────────────
    else:
        if reasoning_class == "complete":
            # Full correct reasoning but wrong answer
            # = calculation error at the very end, strong signal
            return 0.4

        elif reasoning_class == "correct_approach_incomplete":
            return 0.25

        elif reasoning_class == "correct_approach_wrong_execution":
            return 0.2

        elif reasoning_class == "adjacent_approach":
            return 0.15

        elif reasoning_class == "wrong_approach":
            return 0.05

        else:
            # memorized / incoherent + wrong = nothing
            return 0.0


# ============================================================
# MAIN REWARD FUNCTION
# ============================================================

def compute_score(data_source, solution_str, ground_truth,
                  extra_info=None, **kwargs):
    """
    Full hybrid reward for AIMO3 GRPO training.

    Gemini is called for ALL responses with substantial content
    (>400 chars) when a valid gold solution exists.
    Rule-based fallback when gold is unavailable or response is short.
    """
    if extra_info is None:
        extra_info = {}

    gt            = str(ground_truth) if ground_truth is not None else None
    pred          = extract_boxed(solution_str)
    gold_solution = extra_info.get("gold_solution", "")
    problem       = extra_info.get("problem", "")

    # ── No \boxed{} → always 0.0 ────────────────────────────
    if pred is None:
        return 0.0

    # ── Short response → rule-based only, no Gemini ─────────
    # Not enough content to judge reasoning quality
    if len(solution_str) < 400:
        if answers_match(pred, gt):
            return 0.05   # correct but suspiciously short = memorized
        return 0.0

    # ── No valid gold → rule-based fallback ─────────────────
    # Can't judge trajectory without a reference
    if not is_valid_gold(gold_solution):
        if answers_match(pred, gt):
            return 0.6    # correct + long enough, assume some reasoning
        return 0.05

    # ── Full Gemini evaluation ───────────────────────────────
    # Has boxed answer + substantial content + valid gold solution
    judgment = gemini_full_judge(
        problem=problem,
        student_solution=solution_str,
        gold_solution=gold_solution,
    )

    # Override answer_correct with our own rule-based check
    # Don't trust Gemini for exact answer matching —
    # it can make mistakes on numeric equivalence
    judgment["answer_correct"] = answers_match(pred, gt)

    return compute_reward_from_judgment(judgment)
