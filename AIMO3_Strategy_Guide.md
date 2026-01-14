# AIMO Progress Prize 3: Strategy to Score 47/50

To achieve a score of 47/50 in AIMO3, you need a system that rivals the best closed-source models (like o1/Sonnet 3.5). The previous winning score was 34/50. Closing this gap requires a multifaceted approach combining **Reasoning Models**, **Tool-Integrated Reasoning (TIR)**, and **Massive Sampling/Verification**.

## 1. The Winning Strategy: "Think, Code, Verify"

The most effective strategy for AIMO3 is not just "prompting a model" but building an **inference engine**.

### Core Architecture
1.  **Hybrid Approach**: Do not rely on a single model mode.
    *   **Model A (Reasoning/CoT)**: A strong reasoning model (like DeepSeek-R1 or QwQ) to plan and break down the problem.
    *   **Model B (TIR/Python)**: A model fine-tuned to write and execute Python code to solve calculation-heavy or brute-forceable components.
2.  **Iterative Execution (The "Agentic" Loop)**:
    *   Similar to AlphaGeometry or Numina's approach, the model should propose a step, write code to verify/compute it, read the output, and refine the plan.
3.  **Massive Majority Voting (SC-TIR)**:
    *   Generate **64 to 256 solutions** per problem.
    *   Filter out solutions where code execution failed.
    *   Cluster the remaining answers and pick the majority vote.
    *   *Note*: With H100s available, you can afford more inference compute than previous years.

## 2. Best Open-Source Models (Jan 2026)

Since you have H100 access (80GB VRAM) for inference, you can run larger models than previous contestants, but strict time limits (5 hours for 110 problems) mean efficiency is key.

| Model Category | Top Recommendation | Why? |
| :--- | :--- | :--- |
| **SOTA Reasoning** | **DeepSeek-R1 (Distill-Llama-70B or Qwen-32B)** | Best open-source reasoning capabilities. The "Distill" versions retain the "Aha!" moments of the massive R1 model but fit in GPU memory (especially if 4-bit quantized). |
| **Balanced** | **Qwen2.5-Math-72B-Instruct** | Excellent balance of pure math knowledge and instruction following. Can be quantized to fit H100. |
| **Code/TIR Specialist** | **NuminaMath-7B-TIR** (or newer 32B variant) | Specifically trained to interleave text and code. You might want to fine-tune a newer **Qwen-2.5-Coder-32B** on the NuminaTIR dataset to upgrade this. |

**Recommendation:** Start with **DeepSeek-R1-Distill-Qwen-32B** or **QwQ-32B**. They are small enough to generate many samples quickly on an H100 but smart enough to solve hard problems.


## 3. Why Python? (The "Calculator" Advantage)

You might ask: *Why do I need code for math?*
LLMs (like GPT-4 or DeepSeek) are **probabilistic text generators**, not calculators. They are excellent at planning a solution but terrible at executing it reliably.

**Example**:
*   **Problem**: "Find the remainder when $14^{2025}$ is divided by $100$."
*   **Pure LLM**: might hallucinate "14, 96, 44..." because it's predicting the next token.
*   **LLM + Python**:
    1.  LLM writes: `print(pow(14, 2025, 100))`
    2.  Python executes: `24`
    3.  LLM reads output and concludes: "The answer is 24."

**Key Use Cases in AIMO**:
1.  **Arithmetic**: Multiplying large numbers or modular arithmetic without error.
2.  **Brute Force**: Writing a loop to check the first 100,000 integers for a property.
3.  **Algebra**: Using libraries like `sympy` to solve systems of equations symbolically.


## 4. Closing the Gap: Lessons from Closed Models

**How do models like o1/o3-preview score 50/50?**
Research shows their dominance comes from **Test-Time Compute** (or "System 2" thinking). They don't just answer; they "think" for seconds or minutes before outputting a token.
1.  **Massive Search**: They generate thousands of internal reasoning paths (like searching a chess move tree) and discard the bad ones.
2.  **Self-Correction**: They have internal "verifiers" that check their own work. If a step looks wrong, they backtrack.
3.  **Neuro-Symbolic**: Systems like **AlphaGeometry** combine a neural network (for intuition) with a symbolic engine (for rigorous logical proof).

**How we emulate this (The "Open o1" Strategy)**:
*   **Sequential Revision**: Instead of asking for the answer once, chain prompt the model: "Review your previous step. Is it correct? If not, fix it."
*   **Reasoning Tokens**: Use models like DeepSeek-R1 that are trained to output `<think>` tags, mimicking the hidden thought process of o1.

### Massive Search (Best-of-N) Explained

Imagine you are a teacher grading a math problem.
*   **Scenario A**: You ask **1 student** to solve it. They might make a silly arithmetic mistake. You trust their answer 50%.
*   **Scenario B**: You ask **100 students** to solve it independently.
    *   65 students say the answer is `42`.
    *   20 students say the answer is `12` (they forgot to carry a one).
    *   15 students fail to finish.
    *   **Trust**: You are 99% sure the answer is `42`.

**Massive Search** treats the AI as those 100 students. Since the AI is "probabilistic" (it rolls dice to pick words), running it 100 times generates 100 slightly different reasoning paths.

#### Implementation
1.  **Parallel Generation (`vLLM`)**: We don't run the model 100 times in a loop (too slow). We use `vLLM` to generate **parallel** branches.
    ```python
    self.sampling_params = SamplingParams(
        n=16,               # Generate 16 solutions at once
        temperature=0.6,    # "Creativity" - ensures diversity
    )
    ```
2.  **Execution Filter**: Only filter student answers that "showed their work" (wrote code) and got a valid number.
3.  **Majority Vote**: Count the valid votes (e.g. `counts.most_common(1)`).

**Why H100 is Key**: To run **N=64** or **N=128** (which is where you get the "superhuman" accuracy boost), you need massive GPU memory bandwidth. That is why we chose the **32B model** (fits easily) over the 120B model.


## 6. Training & Datasets

To reach 47/50, you likely need to **fine-tune** your own model, specifically for the "TIR" (Tool Integrated Reasoning) capability.

### Datasets to Train On
1.  **NuminaMath-TIR**: The gold standard from AIMO1. Contains ~70k problems with interleaved Python code execution steps.
2.  **OpenMathReasoning (Nvidia)**: 500k+ problems with step-by-step reasoning.
3.  **OpenMathInstruct-2**: A massive dataset of math problems with code-interpreter solutions.
4.  **AIMO3 Specific Constraints**:
    *   The new problems are "AI Hard" and require integers 0-99999.
    *   *Action*: Filter your training data to prioritize problems with integer answers and number theory/combinatorics, which are common in Olympiads.

### Training Recipe
1.  **Supervised Fine-Tuning (SFT)**:
    *   Take a base model (e.g., Qwen-2.5-32B).
    *   Train on **NuminaMath-TIR** + **OpenMathInstruct** to teach it the format: `Reason -> Python Block -> Output -> Reason`.

2.  **Process Reward Model (PRM) / Verifier (Optional but Pro)**:
    *   **What is it?**: A "Teacher Model" that grades every step of the reasoning.
    *   **Option A (Easy)**: Skip it. Use "Majority Voting" (Outcome Supervision) which is 90% as effective for less work.
    *   **Option B (Pro)**: Download a pre-trained PRM like [`Qwen/Qwen2.5-Math-PRM-7B`](https://huggingface.co/Qwen/Qwen2.5-Math-PRM-7B) or `math-shepherd-mistral-7b-prm` from Hugging Face.
    *   **Usage**: Run this small model to score your 64 generated solutions and pick the one with the highest "logic score", rather than just the most common answer.

## 7. Hardware & Technical Setup (Kaggle Specifics)

*   **H100 Usage**: You have 1x H100 (80GB). This allows you to run **70B parameter models** in 4-bit or 8-bit quantization comfortably, or **32B models** in FP16.
*   **Time Management**:
    *   5 hours = 18,000 seconds.
    *   110 Problems = ~160 seconds per problem.
    *   This is tight. A 32B model generates ~40 tok/sec on H100.
    *   You can generate ~6,400 tokens per problem.
    *   Split this: 8 parallel streams of 800 tokens each? Or 16 streams of 400?
    *   **Optimization**: Use **vLLM** or **SGLang** for inference. They are critical for high throughput on Kaggle.

## 8. Summary Checklist for 47/50

- [ ] **Base Model**: QwQ-32B or DeepSeek-R1-Distill-32B.
- [ ] **Method**: Tool-Integrated Reasoning (Model writes Python to solve sub-steps).
- [ ] **Inference**: vLLM engine, Best-of-N sampling (N=64 or more).
- [ ] **Data**: Fine-tune on NuminaMath-TIR to enforce the Python format.
