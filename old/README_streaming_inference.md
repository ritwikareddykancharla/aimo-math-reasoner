# Streaming Inference Code Explanation

This document explains the codebase found in `streaming-inference.ipynb`, which implements a solution for the AI Mathematical Olympiad (AIMO) Progress Prize 3 competition. The solution utilizes a local vLLM server for inference, advanced prompt engineering with iterative refinement, and a voting mechanism to select the final answer.

## Overview

The notebook is designed to run in a Kaggle environment (offline competition). It performs the following main tasks:
1.  Sets up the environment and starts a local vLLM server.
2.  Connects to the server using an OpenAI-compatible client.
3.  Implements a custom generation loop that handles reasoning, answer extraction, and multi-turn refinement.
4.  Aggregates multiple solution attempts via a voting mechanism.
5.  Serves the predictions using the competition's evaluation API.

## Key Components

### 1. Configuration & Setup
The code initializes by checking the runtime environment (Kaggle Interactive vs. Batch/Commit).
- **Environment Detection**: Functions like `is_on_kaggle()`, `is_on_kaggle_interactive()` helper determine the execution context.
- **Resource Management**: It manages time `cutoff_times` to ensure the submission finishes within the competition limits (typically 5 hours).

### 2. vLLM Server (`start_vllm_server`)
The notebook starts a local instance of the vLLM server to serve the model (referenced as `gpt-oss-120b`, likely functionality-wise).
- **Configuration**: Sets environment variables for optimization (`VLLM_ATTENTION_BACKEND`, `tensor-parallel-size`, etc.).
- **Execution**: Runs the server as a subprocess, logging output to `a-vllm.log`.

### 3. Client & Tokenization
- **OpenAI Client**: Configured to point to the local vLLM server (`http://127.0.0.1:8000/v1`) or a remote URL if configured.
- **Harmony Encoding**: Uses `openai_harmony` to handle chat templates and tokenization for the specific model architecture.
- **Prompt Building**: `build_prompt_token_ids` constructs the conversation history for the model.

### 4. Generation Logic (`generate_solution`)
This is the core logic for solving a single problem.
- **Streaming**: Uses streaming inference to process tokens as they are generated.
- **Iterative Refinement**: The loop allows for up to 3 iterations ("turns") per solution attempt.
    - If an answer isn't found or verified, the system appends a "user follow-up" prompt (e.g., "Are you sure?", "Please guess a reasonable answer") to encourage the model to correct itself or finalize an answer.
- **Answer Extraction**: `extract_boxed_text` looks for LaTeX `\boxed{answer}` format.
- **KV Cache Monitoring**: `get_gpu_kv_cache_usage` monitors GPU usage to prevent OOM errors, terminating generation if the cache is too full.

### 5. Voting Mechanism (`vote_answer`)
Multiple solutions are generated in parallel for the same problem.
- **Aggregation**: Answers are collected in a `Counter`.
- **Scoring**: A weighted scoring system is used where answers are weighted by their value (`log(1.25 + abs(value))`), possibly to prioritize non-trivial answers or handle specific distribution characteristics.
- **Selection**: The answer with the highest score that meets a "win" threshold against the runner-up is selected.

### 6. Main Solver (`solve`)
- **Parallelism**: Uses `ThreadPoolExecutor` to run `num_generations` (default: 6) of `generate_solution` concurrently.
- **Usage Strategy**: Runs generation, collects results, and calls `vote_answer` to determine the final output.

### 7. Submission Server (`predict`)
- Implements the `predict` function required by `kaggle_evaluation.aimo_3_inference_server`.
- **Routing**: For local testing or specific commit scenarios, it might skip questions for speed (processing only "Alice" or "Norwegian" related questions).
- **Gateway**: Starts the inference server which interfaces with the Kaggle evaluation system.

## Workflow Summary

1.  **Initialize**: Start vLLM server in background.
2.  **Receive Problem**: The `predict` function receives a problem ID and text.
3.  **Generate Candidates**: `solve` launches multiple parallel threads of `generate_solution`.
4.  **Refine**: Each thread prompts the model. If the model is unsure or time is running out, follow-up prompts force a guess or verification.
5.  **Extract**: Valid integers inside `\boxed{}` are extracted.
6.  **Vote**: Valid answers are weighted and the best one is chosen.
7.  **Submit**: The final integer answer is returned to the evaluation API.
