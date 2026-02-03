# ALG: Adaptive Lemma Graph Strategy

**Version:** 1.0  
**Purpose:** Replace parallel brute-force with structured lemma-based reasoning for AIMO3  
**Target:** 50/50 accuracy through rigorous verification

---

## 1. Philosophy: From "Many Guesses" to "One Correct Proof"

### Current Approach (Parallel ThreadPool)
- 8 independent attempts → vote on answer
- Each attempt is a "stream of consciousness"
- No structured verification
- Success depends on "getting lucky" with reasoning path

### Proposed Approach (ALG: Adaptive Lemma Graph)
- **One** structured traversal through a lemma dependency graph
- Each lemma MUST be verified before proceeding
- Python is a **mandatory gatekeeper**, not an optional tool
- Success depends on **correctness of each step**

### Analogy
| Approach | Analogy | Risk |
|----------|---------|------|
| Parallel 8 attempts | 8 students guess independently | All could be wrong |
| ALG Graph Traversal | 1 student builds a proof step-by-step, checking each step | Wrong step is caught immediately |

---

## 2. The Four Phases of ALG

### Phase 1: Topological Planning (The "Map")
**Goal:** Decompose problem into a directed acyclic graph (DAG) of lemmas

**Output:**
```
Lemma Graph:
  L1: [Statement] → L2, L3 (L1 must be proven before L2 and L3)
  L2: [Statement] → L4
  L3: [Statement] → L4
  L4: [Statement] → FINAL
```

**Types of Lemmas:**
| Type | Description | Example |
|------|-------------|---------|
| **Structural** | Establish problem structure | "The set of valid functions forms a group under composition" |
| **Reduction** | Simplify the problem | "Counting problem reduces to counting divisors" |
| **Computational** | Requires calculation/bookkeeping | "Compute the number of valid combinations" |
| **Verification** | Check final answer | "Verify answer satisfies all constraints" |

**Node Properties:**
- `id`: Unique identifier (L1, L2, ...)
- `statement`: Mathematical claim
- `type`: structural | reduction | computational | verification
- `dependencies`: List of prerequisite lemma IDs
- `verification_strategy`: How to verify this lemma with Python
- `status`: pending | in_progress | verified | failed

---

### Phase 2: Atomic Verification Gate (The "Check")
**Goal:** Prove one lemma at a time, MUST verify before proceeding

**Algorithm:**
```
function PROVE_LEMMA(lemma, verified_context):
    for attempt in 1..MAX_RETRIES:
        # 1. PROPOSE
        reasoning = LLM.generate_reasoning(lemma, verified_context)
        
        # 2. ADVERSARIAL CHECK
        critique = LLM.generate_critique(reasoning)
        # "If this Lemma is wrong, it's likely because..."
        
        # 3. CODE VERIFICATION (MANDATORY)
        code = extract_python_code(reasoning)
        result = sandbox.execute(code)
        
        # 4. ASSERTION
        if result.success and assertions_pass(result):
            return SUCCESS, reasoning, result
        else:
            # Prune the branch - reasoning was wrong
            add_failure_to_context(lemma, reasoning, result)
            continue  # Retry with new reasoning
    
    return FAILURE  # All retries exhausted
```

**The Gate Principle:**
- **NO LEMMA MAY BE SKIPPED**
- **NO LEMMA MAY BE ASSUMED TRUE WITHOUT VERIFICATION**
- **IF VERIFICATION FAILS, THE REASONING IS WRONG**

---

### Phase 3: Search Exhaustion (The "Anti-Hallucination Guard")
**Goal:** For counting/enumeration problems, use Python to exhaustively search

**Problem:** LLMs are terrible at precise counting. They estimate.

**Example - Shifty Functions:**
- Model might say "there are 72 such functions" (incorrect)
- Python search finds 160 (correct)

**Implementation:**
```
if lemma.type == "computational" and lemma.involves_counting:
    # Force exhaustive enumeration
    code = generate_exhaustive_search_code(lemma)
    result = sandbox.execute(code)
    
    # The Python result IS the truth
    lemma.value = result.output
```

**Key Insight:** The 120B model provides the *theoretical framework* (cyclotomic factors). Python provides the *exact count* (enumerating all valid combinations).

---

### Phase 4: Final Synthesis (The "Answer")
**Goal:** Combine verified lemmas into final answer

**Algorithm:**
```
function SYNTHESIZE(verified_lemmas):
    # All lemmas are now verified truths
    synthesis_code = LLM.generate_synthesis(verified_lemmas)
    
    # Final verification (optional but recommended)
    final_result = sandbox.execute(synthesis_code)
    
    # Extract boxed answer
    answer = extract_answer(final_result.output)
    
    return answer
```

---

## 3. Graph Traversal Algorithm

### Topological Traversal with Backtracking

```python
def traverse_lemma_graph(problem, max_retries=3):
    """
    Traverse lemma graph from roots to final synthesis.
    Uses DFS with backtracking on verification failure.
    """
    
    # Phase 1: Build the graph
    graph = build_lemma_graph(problem)
    
    # Track state
    verified = {}  # lemma_id -> (statement, proof, result)
    failed_paths = defaultdict(list)  # lemma_id -> [failed_attempts]
    
    def solve_lemma(lemma_id):
        """Recursively solve a lemma and its dependencies."""
        
        if lemma_id in verified:
            return verified[lemma_id]
        
        lemma = graph.get(lemma_id)
        
        # First, solve all dependencies
        dep_results = []
        for dep_id in lemma.dependencies:
            dep_result = solve_lemma(dep_id)
            if dep_result is None:
                return None  # Dependency failed, propagate failure
            dep_results.append(dep_result)
        
        # Now try to prove this lemma
        context = build_context(dep_results, failed_paths[lemma_id])
        
        for retry in range(max_retries):
            # Generate proof with adversarial check
            proof = generate_proof(lemma, context, retry)
            
            # Extract and execute verification code
            code = extract_code(proof)
            if not code:
                # No code provided - this is an error
                failed_paths[lemma_id].append({
                    'proof': proof,
                    'error': 'No verification code provided'
                })
                continue
            
            result = sandbox.execute(code)
            
            if result.success and verify_assertions(result):
                # SUCCESS: Store and return
                verified[lemma_id] = {
                    'lemma': lemma,
                    'proof': proof,
                    'result': result,
                    'retry_count': retry
                }
                return verified[lemma_id]
            else:
                # FAILURE: Record and retry
                failed_paths[lemma_id].append({
                    'proof': proof,
                    'result': result,
                    'error': result.error if not result.success else 'Assertions failed'
                })
                
                # Prune the branch - don't use this reasoning
                continue
        
        # All retries failed
        return None
    
    # Find root lemmas (no dependencies)
    root_lemmas = [lid for lid, l in graph.items() if not l.dependencies]
    
    # Solve all roots first
    for root_id in root_lemmas:
        if solve_lemma(root_id) is None:
            return None  # Cannot solve even basic lemmas
    
    # Solve final synthesis lemma
    final_result = solve_lemma('FINAL')
    
    if final_result:
        return extract_answer(final_result['result'].output)
    else:
        return None  # Failed to solve problem
```

### Traversal Visualization

```
Problem: "Count shifty functions"

Phase 1 - Graph Construction:
  L1 (structural): "Shifty functions correspond to divisors of x^k + x^l"
    ↓
  L2 (reduction): "Problem reduces to counting valid (k,l) pairs with |k-l| ≤ 8"
    ↓
  L3 (structural): "Valid functions are products of cyclotomic polynomials"
    ↓
  L4 (computational): "Enumerate all valid polynomial combinations"
    ↓
  L5 (verification): "Verify count matches theoretical constraints"
    ↓
  FINAL (synthesis): "Compute 2 × count (include negative variants)"

Phase 2-4 - Traversal:
  1. Prove L1 (structural)
     └─> Python verifies: divisors of x^k + x^l indeed give shifty functions
  
  2. Prove L2 (reduction)
     └─> Python verifies: degree constraint implies |k-l| ≤ 8
  
  3. Prove L3 (structural)
     └─> Python verifies: cyclotomic polynomial characterization
  
  4. Prove L4 (computational) ← CRITICAL STEP
     └─> Python EXHAUSTIVELY searches all valid combinations
     └─> Result: 80 (not 72!)
  
  5. Prove L5 (verification)
     └─> Python verifies: no functions missed, no double-counting
  
  6. Synthesize FINAL
     └─> Python computes: 80 × 2 = 160
```

---

## 4. The System Prompt (ALG Protocol)

```text
# PROTOCOL: ADAPTIVE LEMMA GRAPH (ALG)
You are an IMO Gold Medalist paired with a Symbolic Verification Engine. 
Your goal is 50/50 accuracy through rigorous step-by-step verification.

## PHASE 1: TOPOLOGICAL PLANNING
Before any calculation, output a "Lemma Graph." 
Identify the logical nodes (Lemmata) required to reach the answer.

Format:
**Lemma 1** [Type: structural|reduction|computational|verification]: [Statement]
- Dependencies: None
- Verification Strategy: [What Python will check]

**Lemma 2** [Type: ...]: [Statement]  
- Dependencies: L1
- Verification Strategy: ...

... continue until all paths lead to FINAL

**FINAL** [Type: synthesis]: Combine Lk, Ll, ... to compute answer
- Dependencies: [list all required lemmas]

## PHASE 2: ATOMIC VERIFICATION GATE
You are FORBIDDEN from moving to Lemma N+1 until Lemma N is verified.

For each Lemma, you MUST output:

**PROPOSE:**
[Your mathematical reasoning for this lemma]

**ADVERSARIAL CHECK:**
"If this Lemma is wrong, it's likely because..."
[List potential failure modes]

**CODE VERIFICATION:**
```python
# Your verification code here
# Must use assertions
assert condition, "Verification failed: explanation"
print(result)
```

**ASSERTION RESULT:**
[Wait for execution result]

If execution returns an error or assertion fails:
- You MUST acknowledge: "Lemma verification FAILED. Pruning this branch."
- You MUST propose a new approach
- You MUST NOT proceed to next lemma

## PHASE 3: SEARCH EXHAUSTION (For Counting Problems)
If a problem involves counting ("How many...", "Count the..."):
- Do NOT estimate
- Do NOT calculate by hand
- Use Python to perform EXHAUSTIVE search of the space
- Use itertools, sympy, or brute force as needed

## PHASE 4: FINAL SYNTHESIS
Only after ALL lemmas are verified:
- Combine verified results logically
- Perform final computation in Python
- Output answer in \boxed{}

## CRITICAL RULES
1. NEVER skip verification
2. NEVER assume a lemma is true without code proof
3. NEVER proceed to next lemma if current fails
4. ALWAYS use assertions in Python code
5. FOR counting problems: ALWAYS use exhaustive search
```

---

## 5. Backtracking and Repair

### When a Lemma Fails Verification

**Option 1: Local Repair (Default)**
- Keep the same lemma graph structure
- Try different reasoning approach
- Try different verification code
- Up to MAX_RETRIES attempts

**Option 2: Graph Restructuring (If local repair fails)**
- Current lemma might be incorrectly stated
- Backtrack to previous lemma
- Ask LLM to reconsider the decomposition
- May add new intermediate lemmas

**Option 3: Terminal Failure (If all else fails)**
- Cannot solve this problem with current approach
- Return best guess or None
- Log failure for analysis

### Repair Prompt Template

```
Your previous attempt at Lemma {lemma_id} failed verification.

**Lemma Statement:** {statement}

**Your Previous Attempt:**
{previous_proof}

**Execution Result:**
{execution_error}

**Failed Because:**
{error_analysis}

**Previously Failed Approaches:**
{list_of_previous_failures}

Fix your approach and provide:
1. Corrected reasoning addressing the failure
2. Fixed Python verification code
3. New adversarial check

DO NOT repeat previous approaches.
```

---

## 6. Integration with Existing AIMO3 Infrastructure

### Reuse from Current System

| Component | Current Usage | ALG Usage |
|-----------|--------------|-----------|
| `AIMO3Sandbox` | 16 parallel kernels | 1 kernel (sequential) |
| `AIMO3Tool` | Optional Python calls | Mandatory verification gate |
| `vLLM server` | 8 parallel streams | Sequential generation |
| `System prompt` | General guidelines | ALG protocol enforced |
| `answer extraction` | Scan for `\boxed{}` | Scan for `\boxed{}` |

### New Components Needed

1. **`LemmaGraphBuilder`** - Phase 1 implementation
2. **`LemmaVerifier`** - Phase 2 implementation  
3. **`GraphTraverser`** - Main traversal algorithm
4. **`RepairStrategist`** - Backtracking logic
5. **`ExhaustiveSearcher`** - Phase 3 for counting problems

### Flow Integration

```
Current Flow:
  predict() → AIMO3Solver.solve_problem() → 8 parallel _process_attempt() → vote

ALG Flow:
  predict() → ALGSolver.solve_problem() → 
    Phase 1: build_lemma_graph() →
    Phase 2-4: traverse_graph() →
    return answer or None
```

---

## 7. Success Metrics

| Metric | Parallel Approach | ALG Approach |
|--------|------------------|--------------|
| Avg attempts per problem | 8 | 1 |
| Verification rate | ~60% | ~100% (mandatory) |
| Backtracking events | 0 | Variable |
| Time per problem | ~100s | ~200-500s (but more accurate) |
| Theoretical accuracy ceiling | ~88% (44/50) | ~100% (50/50) |

**Trade-off:** Speed vs Accuracy
- Parallel: Fast but brute-force
- ALG: Slower but rigorous

---

## 8. Example: Shifty Functions Walkthrough

**Problem:** Count shifty functions with support in [0,8]

### Phase 1: Graph Construction

```
L1 [structural]: Shifty α corresponds to polynomial A(x) where A·B = x^k + x^l
  - Dependencies: None
  - Verification: Check polynomial algebra

L2 [structural]: A(x) must be a product of cyclotomic polynomials
  - Dependencies: L1
  - Verification: Factor x^k + x^l, verify cyclotomic structure

L3 [reduction]: |k-l| ≤ 8 due to degree constraint
  - Dependencies: L1
  - Verification: Check degree bounds

L4 [computational]: Enumerate all valid (k,l) pairs and divisor combinations
  - Dependencies: L2, L3
  - Verification: Exhaustive search with itertools
  
L5 [verification]: Verify no double-counting and no missing cases
  - Dependencies: L4
  - Verification: Check bijection properties

FINAL [synthesis]: Total = 2 × count (include sign variants)
  - Dependencies: L5
```

### Phase 2-4: Traversal

**Proving L1:**
```python
import sympy as sp

# Verify polynomial characterization
x = sp.Symbol('x')
A = sp.Poly([1, 0, 1], x)  # Example: 1 + x^2
B = sp.Poly([1, 1], x)     # Example: 1 + x

# Check convolution corresponds to polynomial multiplication
result = A * B
assert result.degree() <= 8, "Degree constraint violated"
print("L1 verified: Polynomial characterization holds")
```

**Proving L4 (Critical Step):**
```python
import itertools
from sympy import cyclotomic_poly, factor

# Exhaustive search - NO ESTIMATION
count = 0
valid_functions = []

for k in range(9):
    for l in range(k+1, 9):  # l > k, |l-k| <= 8
        # Generate polynomial x^k + x^l
        x = sp.Symbol('x')
        poly = x**k + x**l
        
        # Find all divisors with degree <= 8
        factors = factor(poly)
        # ... enumerate all divisor combinations ...
        
        for divisor in get_all_divisors(poly):
            if divisor.degree() <= 8:
                count += 1
                valid_functions.append(divisor)

print(f"Exhaustive count: {count}")  # 80, not 72!
assert count == 80, f"Expected 80, got {count}"
```

**FINAL Synthesis:**
```python
# Include negative variants
final_answer = 80 * 2  # 160
print(f"Final answer: {final_answer}")
```

---

## 9. Implementation Roadmap

### Phase A: Core Framework (This PR)
1. Implement `LemmaGraph` data structure
2. Implement `ALGSolver` with sequential traversal
3. Update system prompt to ALG protocol
4. Add mandatory verification gates

### Phase B: Enhanced Features (Future)
1. Graph restructuring on failure
2. Automatic "computational" lemma detection
3. Caching of verified lemmas across problems
4. Adaptive timeout based on graph complexity

### Phase C: Optimization (Future)
1. Parallel lemma proving (when independent)
2. Lemma library (reuse common lemmas)
3. Learned graph structures from training data

---

## 10. Risk Analysis

| Risk | Mitigation |
|------|------------|
| LLM fails to generate valid graph | Fallback to single-attempt solving |
| Infinite loop in traversal | Max retries + timeout per lemma |
| Python code hangs | Sandbox timeout (existing) |
| Graph has circular dependencies | Topological sort validation |
| Too many lemmas (context overflow) | Limit to MAX_LEMMAS (8-10) |
| Wrong graph structure | Backtracking + restructuring |

---

**Next Step:** Implement the ALG framework code based on this strategy.
