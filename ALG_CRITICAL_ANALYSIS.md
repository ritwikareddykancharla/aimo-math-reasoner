# ALG Critical Analysis: Can It Get 50/50?

**Honest Assessment:** ALG *theoretically* can achieve 50/50, but *practically* has significant failure modes that must be addressed.

---

## 1. Why ALG Could Achieve 50/50

### The Mathematical Advantage
- **Rigorous verification** catches errors that slip through in free-form reasoning
- **Structured decomposition** prevents "reasoning drift" (getting lost mid-problem)
- **Python as ground truth** eliminates arithmetic/algebraic hallucinations
- **Exhaustive search** guarantees correct counts (vs human/LLM estimation)

### Historical Evidence
- **AlphaGeometry** uses similar graph-of-lemmas approach and achieves SOTA
- **Formal theorem provers** (Lean, Coq) prove that structured verification works
- **Human IMO medalists** solve problems using step-by-step lemma proofs

---

## 2. Critical Failure Modes (Why It Might Fail)

### FAILURE MODE 1: Wrong Lemma Graph
**Problem:** The LLM decomposes the problem incorrectly from the start.

**Example - Shifty Functions:**
- LLM might say: "Lemma 1: Shifty functions correspond to roots of unity"
- **This is wrong** - they correspond to polynomial divisors, not roots
- Every subsequent lemma builds on false foundation
- **Result:** Complete failure, no recovery possible

**Why it happens:**
- 120B model still hallucinates mathematical structures
- Problem statements can be misinterpreted
- Complex problems have multiple valid approaches; LLM might pick wrong one

**Current Mitigation:** None - we trust the LLM's decomposition

---

### FAILURE MODE 2: Verification Code is Wrong
**Problem:** The LLM writes buggy verification code that passes but proves nothing.

**Example:**
```python
# LLM thinks this verifies "all divisors are cyclotomic"
for d in divisors:
    assert is_cyclotomic(d), "Not cyclotomic"  # Bug: is_cyclotomic() is wrong
print("Verified!")
```

**Why it happens:**
- LLMs write code with subtle bugs
- Library misuse (sympy functions used incorrectly)
- Off-by-one errors, edge case misses

**Current Mitigation:** Assertion statements - but assertions can be wrong too

---

### FAILURE MODE 3: Infinite/Unbounded Repair Loops
**Problem:** LLM keeps failing verification, keeps retrying with same wrong approach.

**Example:**
- Lemma: "Count is 72"
- Python: `assert count == 72` → FAILS (actual is 80)
- LLM repair: "Maybe I missed some" → new code → count = 75
- LLM repair: "Let me try different approach" → new code → count = 73
- ... (continues until max retries)

**Why it happens:**
- LLM doesn't truly understand the error
- Keeps making variations of same mistake
- No "debugging insight" - just trial and error

**Current Mitigation:** Max retries (3) - but then we fail the problem

---

### FAILURE MODE 4: Time Exhaustion
**Problem:** Sequential verification is SLOW.

**Estimated Timings:**
| Phase | Time per Lemma | Total (8 lemmas) |
|-------|---------------|------------------|
| LLM generation | 10-30s | 80-240s |
| Python execution | 2-10s | 16-80s |
| Repair attempts | 3× base | 240-960s |
| **Total** | | **5-20 minutes per problem** |

**Kaggle constraints:**
- 50 problems in 5 hours = 6 minutes per problem
- ALG might exceed time budget on hard problems

**Current Mitigation:** Timeout handling - but leads to incomplete proofs

---

### FAILURE MODE 5: Problems That Don't Decompose
**Problem:** Some AIMO problems resist clean lemma decomposition.

**Examples:**
- Pure insight problems: "Notice that 2025 = 45²..."
- Single-shot constructions: "Build this specific example"
- Pattern recognition: "The answer is the Fibonacci numbers"

**Why it happens:**
- Not all math is "theorem-proof-theorem"
- Some solutions are "elegant insights" not lemma chains
- ALG forces structure where none naturally exists

---

### FAILURE MODE 6: Context Window Overflow
**Problem:** Long lemma chains exceed model context.

**Math:**
- 8 lemmas × 500 tokens each = 4,000 tokens
- Plus verification results = 8,000 tokens
- Plus repair history = 16,000+ tokens
- **120B model context:** 65,536 tokens (limit is 128k but we use 64k)

**Why it happens:**
- Each lemma adds to conversation
- Repair attempts accumulate
- Python outputs can be verbose

---

### FAILURE MODE 7: False Confidence in Verification
**Problem:** Code runs without error but proves the wrong thing.

**Example:**
```python
# LLM thinks this verifies L1
result = compute_something()
assert result > 0  # Always true, proves nothing about L1
print("L1 verified!")
```

**Why it happens:**
- LLM writes "placeholder" verification
- Assertions are trivially true
- No connection between code and lemma statement

---

## 3. Honest Probability Assessment

| Scenario | Probability | Result |
|----------|-------------|--------|
| Perfect lemma graph + perfect verification | 20% | ✓ Correct (50/50 possible) |
| Minor error caught by repair | 30% | ✓ Correct |
| Wrong graph / unrepairable error | 25% | ✗ Wrong |
| Timeout on complex problem | 15% | ✗ No answer |
| Verification code bug | 10% | ✗ Wrong |

**Expected accuracy:** 50-65% (25-33/50) without fixes

**To reach 47-50/50:** Need to address failure modes aggressively

---

## 4. Critical Fixes Required

### FIX 1: Multiple Graph Hypotheses (Addresses Failure Mode 1)
**Instead of:** One graph decomposition
**Do:** Generate 2-3 alternative decompositions, try each

```python
# New Phase 1
graphs = [
    build_lemma_graph(problem, approach="algebraic"),
    build_lemma_graph(problem, approach="combinatorial"),
    build_lemma_graph(problem, approach="number_theoretic")
]

for graph in graphs:
    result = traverse_graph(graph)
    if result.success:
        return result.answer
```

**Why:** If one graph is wrong, another might be right

---

### FIX 2: Code Review Step (Addresses Failure Mode 2, 7)
**Instead of:** Run code immediately
**Do:** LLM reviews code before execution

```
CODE REVIEW:
"Before executing, verify this code actually tests L1:"
1. Does it test the right condition? 
2. Are there edge cases missing?
3. Is the assertion non-trivial?

If review finds issues, regenerate code.
```

---

### FIX 3: Cross-Verification (Addresses Failure Mode 2, 7)
**Instead of:** One verification code
**Do:** Two independent verifications must agree

```python
# Verification A: Algebraic approach
count_a = verify_algebraically()

# Verification B: Computational approach  
count_b = verify_computationally()

assert count_a == count_b, f"Discrepancy: {count_a} vs {count_b}"
```

---

### FIX 4: Graph Restructuring on Failure (Addresses Failure Mode 3)
**Instead of:** Keep retrying same lemma
**Do:** Ask if lemma is wrong

```
REPAIR LEVEL 1: Different reasoning approach
REPAIR LEVEL 2: Different verification strategy  
REPAIR LEVEL 3: Is the lemma statement itself wrong?
  → Split lemma into L3a, L3b
  → Or merge with adjacent lemma
  → Or ask LLM to reconsider entire graph
```

---

### FIX 5: Hybrid Approach (Addresses Failure Mode 4)
**Instead of:** Pure sequential ALG
**Do:** Parallel ALG attempts with different seeds

```python
# Run 2-3 ALG traversals in parallel
# Each uses different random seed for LLM
# Take answer if 2/3 agree

attempts = [
    alg_solve(problem, seed=42),
    alg_solve(problem, seed=123),
    alg_solve(problem, seed=999)
]

# Vote on final answers
return majority_vote(attempts)
```

**Why:** Keeps rigor of ALG but adds redundancy

---

### FIX 6: Direct Solve Fallback (Addresses Failure Mode 5)
**Instead of:** Force lemma structure
**Do:** Detect if problem is "insight-based"

```python
if is_insight_problem(problem):
    # Use original streaming approach
    return direct_solve(problem)
else:
    # Use ALG
    return alg_solve(problem)
```

**Detection heuristics:**
- Short problem statement
- Mentions "find all" without constraints
- Geometry construction problem
- Pattern/matching problem

---

### FIX 7: Context Compression (Addresses Failure Mode 6)
**Instead of:** Full history
**Do:** Summarize verified lemmas

```
[CONVERSATION STATE]
Verified Lemmas:
- L1: [One-line summary] ✓
- L2: [One-line summary] ✓

Current: L3 (in progress)
Failed attempts: 2 (see summary below)

[COMPRESSED HISTORY]
```

---

## 5. Revised Architecture: ALG-v2

```
┌─────────────────────────────────────────────────────────────┐
│  PROBLEM CLASSIFIER                                          │
│  Is this insight-based or lemma-decomposable?               │
└─────────────────────────────────────────────────────────────┘
              ↓                              ↓
    ┌─────────────────┐            ┌─────────────────┐
    │  DIRECT SOLVE   │            │   ALG SOLVER    │
    │  (Original)     │            │   (New)         │
    └─────────────────┘            └─────────────────┘
                                              ↓
                          ┌──────────────────────────────────┐
                          │  GENERATE 2-3 LEMMA GRAPHS       │
                          │  (different approaches)          │
                          └──────────────────────────────────┘
                                              ↓
                          ┌──────────────────────────────────┐
                          │  PARALLEL TRAVERSAL (2-3 graphs) │
                          │  Each with code review + cross-v │
                          └──────────────────────────────────┘
                                              ↓
                          ┌──────────────────────────────────┐
                          │  VOTE ON FINAL ANSWERS           │
                          │  If 2/3 agree → return           │
                          │  Else → run 4th tiebreaker       │
                          └──────────────────────────────────┘
```

---

## 6. Revised Success Probability (ALG-v2)

| Scenario | Probability | Result |
|----------|-------------|--------|
| At least 1 correct graph + verification | 70% | ✓ Correct |
| Cross-verification catches error | 15% | ✓ Correct |
| Direct solve fallback works | 5% | ✓ Correct |
| All graphs wrong | 7% | ✗ Wrong |
| Timeout despite optimizations | 3% | ✗ No answer |

**Expected accuracy:** 85-90% (42-45/50)

**With fine-tuning on lemma decomposition:** 94-98% (47-49/50)

---

## 7. The Hard Truth

### What ALG Guarantees:
- ✓ No arithmetic errors (Python computes)
- ✓ No "forgotten cases" in counting (exhaustive search)
- ✓ Logical consistency within the proof

### What ALG Cannot Guarantee:
- ✗ The proof structure is correct
- ✗ The lemmas are the "right" ones
- ✗ The verification code is bug-free
- ✗ The model understands the problem

### The 50/50 Question:
**Can ALG get 50/50?** 
- With current 120B model: **Unlikely** (human-level reasoning gaps)
- With ALG-v2 + some parallelization: **Possible** (42-47/50)
- With fine-tuned model on lemma decomposition: **Probable** (47-50/50)

---

## 8. Recommended Implementation Path

### Phase 1: ALG-Core (MVP)
- Single graph, sequential traversal
- Basic repair (3 retries)
- Simple verification

**Expected:** 35-40/50 (better than 44/50? maybe not)

### Phase 2: ALG-v2 (Robust)
- Multiple graph hypotheses
- Cross-verification
- Graph restructuring
- Direct-solve fallback

**Expected:** 42-45/50

### Phase 3: ALG-Advanced (Optimized)
- Fine-tuned decomposition model
- Lemma library (reuse common lemmas)
- Learned repair strategies
- Parallel confidence-based voting

**Expected:** 47-50/50

---

## 9. Conclusion

**ALG is theoretically sound but practically fragile.**

The core issue: **Garbage In, Garbage Out**
- If the lemma graph is wrong → wrong answer
- If verification code is wrong → wrong answer
- We need mechanisms to detect and correct these meta-errors

**ALG-v2 fixes (multiple graphs, cross-verification, restructuring) are NOT optional.**
They are required for the system to be robust.

**Should we build ALG?**
- Yes, but start with ALG-v2 features
- Don't build "pure" ALG - it's too fragile
- Hybrid approach (ALG + parallel verification) is the practical path to 50/50
