# Inference Strategy Implementation Plan

This document details the technical implementation of the "H100 Strategy" for AIMO3, focusing on mimicking the "Test-Time Compute" capabilities of closed-source models.

## 1. Neuro-Symbolic Inference Loop ("The Engine")

We will implement a custom inference loop in Python that integrates the LLM with a Python code execution environment.

### Core Architecture
*   **Model**: `DeepSeek-R1-Distill-Qwen-32B` (or `QwQ-32B`).
*   **Execution Environment**: `huggingface/tool_use` or a custom sandbox for safety (standard `multiprocessing` with timeouts).
*   **Inference Library**: `vLLM` for high-throughput generation.

### Workflow Code (Pseudocode)
```python
def solve_problem(problem_text):
    # 1. Plan & Code Generation
    prompt = f"Solve this potential math olympiad problem. Use Python to verify steps.\nProblem: {problem_text}"
    reasoning_traces = llm.generate(prompt, n=64, temperature=0.7)
    
    candidates = []
    
    # 2. Execution & Refinement
    for trace in reasoning_traces:
        code_blocks = extract_code(trace)
        try:
            result = execute_python(code_blocks)
            # 3. Self-Correction Opportunity
            if "error" in result:
                # Re-prompt model with error (Reflexion)
                new_trace = llm.generate(f"Previous code failed: {result}. Fix it.")
                trace = new_trace
                result = execute_python(extract_code(new_trace))
            
            final_answer = parse_answer(trace)
            candidates.append(final_answer)
        except:
            continue
            
    # 4. Majority Voting (Massive Search)
    best_answer = majority_vote(candidates)
    return best_answer
```

## 2. Massive Search (Tree of Thoughts / Best-of-N)

**Why:** Scaling $N$ is the easiest way to improve performance. o1 typically generates thousands of candidates.
**Implementation:**
*   **Best-of-N**: Generate $N=1024$ solutions.
    *   *Cost*: 110 problems * 1024 samples * 1000 tokens ≈ 100M tokens.
    *   *Time*: On 1xH100 with vLLM, ~4 hours. Feasible.
*   **Policy**: Use `voting_verifiers`. Instead of simple majority vote, weight answers by:
    *   Did the code execute successfully? (Weight x2)
    *   Is the answer an integer in range [0, 99999]? (Weight x1, else 0)

## 3. Self-Correction (Verification)

We will use a "Reflexion" pattern rather than training a full PRM (Process Reward Model) due to complexity, unless existing PRMs (like `Math-Shepherd`) are open-sourced compatible.

**Pattern**:
1.  **Draft**: Model produces an answer.
2.  **Critique**: "Review your answer. Check for: calculation errors, edge cases (n=0, n=1), and logic gaps."
3.  **Refine**: Model outputs corrected answer.

## 4. Proposed File Structure

```
aimo-solution/
├── inference/
│   ├── engine.py       # Main vLLM wrapper
│   ├── sandbox.py      # Safe Python executor
│   ├── strategy.py     # Best-of-N and Voting logic
│   └── prompts.py      # TIR and CoT templates
├── training/
│   ├── fine_tune.py    # SFT script (using unsloth or axolotl)
│   └── data_prep.py    # Formatting NuminaMath-TIR
└── main.py             # Kaggle submission entry point
```

## 5. Verification Plan

*   **Unit Tests**:
    *   `test_sandbox.py`: Verify that infinite loops in generated code are killed.
    *   `test_parsing.py`: Verify we can extract `\boxed{123}` correctly.
*   **Benchmark**:
    *   Run `main.py` on 50 sample problems from AIMO2.
    *   Measure accuracy with N=1 vs N=64.
