# AIMO3 — Fine-Tuning GPT-OSS-120B for Mathematical Reasoning

> **Competition**: [AI Mathematical Olympiad — Progress Prize 3](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3) (Kaggle)
> **Goal**: Improve pass@1 consistency on olympiad-level math problems
> **Model**: GPT-OSS-120B (Mixture-of-Experts, 120B parameters)
> **Hardware**: 8× NVIDIA H100 (AWS) + 1× NVIDIA H200 (GPU Droplet)

---

## The Problem

The base GPT-OSS-120B model **already solves nearly every problem** — scoring **42/50** with majority voting across 8 samples. But pass@1 accuracy is significantly lower, meaning the model *can* find the correct solution but doesn't do so *consistently*.

| Metric | Score | What It Means |
|--------|-------|---------------|
| pass@1 | Low | Model often picks wrong reasoning path on first try |
| pass@8 (majority vote) | **42/50** | Correct answer exists *somewhere* in 8 attempts |
| pass@100 | ~50/50 | Model can solve virtually everything given enough tries |

**The gap between pass@1 and pass@8 is the problem we're solving.** The model has the *capability* — it just lacks *consistency*. We need it to pick the right trajectory on the first attempt, every time.

## Our Approach

### Core Insight
> *Don't teach the model new knowledge — teach it to consistently choose the trajectories it already knows work.*

We fine-tune **only on hard problems where majority voting fails** (accuracy < 0.5). These are problems where fewer than 4 out of 8 attempts produce the correct answer. Training on these forces the model to internalize the successful reasoning patterns for its weakest problems.

### Training Pipeline

```
nvidia/Nemotron-Math-v2 (7M+ trajectories)
  → Filter: accuracy < 0.5 (hard problems only)
  → Filter: integer answers only (AIMO format: 0-999)
  → Filter: with-tool trajectories (Python code execution)
  → Compress: multi-turn → 2-turn format
  → Deduplicate: shortest correct trajectory per problem
  → Fine-tune via SFT / DPO / GRPO
```

### Why These Filters?

| Filter | Rationale |
|--------|-----------|
| **acc < 0.5** | Problems the model already solves consistently (acc ≥ 0.5) don't benefit from more training — focus compute on the hard tail |
| **Integer answers** | AIMO competition requires integer answers (0–999). Training on LaTeX/symbolic answers wastes capacity |
| **With-tool only** | Competition format uses Python code execution. Tool-use trajectories match inference conditions |
| **Shortest trajectory** | More concise reasoning = less noise, faster inference, better signal |

## Methods

We explore three fine-tuning approaches, all targeting the same goal: increasing pass@1 on hard problems.

### 1. SFT (Supervised Fine-Tuning)
Train the model to reproduce the correct trajectory for each hard problem. Uses curriculum learning — start with the hardest problems (acc=0.125), then progressively easier ones (0.25, 0.375) with decreasing learning rate.

### 2. DPO (Direct Preference Optimization)
Generate multiple solutions per problem, use a Gemini judge to score quality, then train the model to prefer higher-quality trajectories. 3-round iterative process with increasing KL penalty to prevent drift.

### 3. GRPO (Group Relative Policy Optimization)
Reinforcement learning approach — reward correct answers, penalize incorrect ones. Groups multiple samples per problem and optimizes relative to the group's performance.

## Infrastructure

| Resource | Spec | Purpose |
|----------|------|---------|
| **AWS 8×H100** | 640GB total VRAM | Full fine-tuning (DeepSpeed ZeRO-3, multi-GPU) |
| **GPU Droplet H200** | 141GB VRAM | Single-GPU QLoRA experiments, evaluation |
| **Kaggle** | Competition notebooks | Inference & submission |

### Training Config (8×H100)
- **DeepSpeed ZeRO Stage 3** with offloading
- **Accelerate** for multi-GPU orchestration
- **bf16** precision
- **Gradient checkpointing** enabled

### Training Config (H200 QLoRA)
- **4-bit QLoRA** (BitsAndBytes NF4)
- **LoRA rank 32**, targeting all attention + MLP layers
- **`merged_4bit_forced`** merge (required for MoE models)

## Repository Structure

```
├── training/
│   ├── sft/                    # SFT scripts (v1 → v6, single-GPU + multi-GPU)
│   ├── dpo/                    # DPO scripts (v1 → v3, iterative with Gemini judge)
│   ├── data/                   # Dataset preparation (v1 → v3)
│   └── eval/                   # Evaluation (pass@1)
├── eda/                        # Exploratory data analysis scripts
├── scripts/                    # Shell scripts, configs (accelerate, DeepSpeed)
├── eval/                       # Competition evaluation data
├── input/                      # Competition input data
├── old/                        # Archived early-stage experiments
└── TRAINING_STRATEGIES.md      # Detailed documentation of all approaches
```

## Key Scripts

| Script | Description |
|--------|-------------|
| `training/data/prepare_dataset_v3.py` | Build corrected SFT dataset from HuggingFace |
| `training/sft/train_sft_120b_v6.py` | Curriculum SFT (3-stage, H200) |
| `training/sft/train_full_8xh100.py` | Full SFT on 8×H100 |
| `training/dpo/train_dpo_v3.py` | 3-round iterative DPO with Gemini judge |
| `eda/eda_nemotron_raw.py` | Raw EDA on Nemotron-Math-v2 from HuggingFace |
| `eda/eda_check.py` | EDA on Kaggle datasets |

## Datasets

| Dataset | Size | Description |
|---------|------|-------------|
| [nemotron-math-v2-sft-hard-tools](https://www.kaggle.com/datasets/ritwikakancharla/nemotron-math-v2-sft-hard-tools) | 147K | v2: hard problems, with tools (needs rebuild) |
| [nemotron-math-v2-sft-high-medium-tools](https://www.kaggle.com/datasets/ritwikakancharla/nemotron-math-v2-sft-high-medium-tools) | 147K | v2: same data, tagged with depth (needs rebuild) |
| nemotron-math-v2-sft-v3 *(coming)* | TBD | v3: corrected depth, integer-only answers |

## Getting Started

```bash
# Clone
git clone git@github.com:ritwikareddykancharla/aimo-math-reasoner.git
cd aimo-math-reasoner

# Set up credentials
cp .env.example .env  # Add HF_TOKEN, KAGGLE_API_TOKEN, KAGGLE_USERNAME

# Run EDA on raw HuggingFace data
python3 eda/eda_nemotron_raw.py --split high_part00 --max-rows 5000

# Build corrected dataset
python3 training/data/prepare_dataset_v3.py --splits high_part00 --no-upload

# Train (8×H100)
accelerate launch --config_file scripts/accelerate_config.yaml \
    training/sft/train_full_8xh100.py
```

## License

MIT — see [LICENSE](LICENSE)

## Acknowledgments

- **[nvidia/Nemotron-Math-v2](https://huggingface.co/datasets/nvidia/Nemotron-Math-v2)** — Source dataset (CC BY 4.0)
- **[GPT-OSS-120B](https://huggingface.co/unsloth/gpt-oss-120b-unsloth-bnb-4bit)** — Base model
- **[Unsloth](https://github.com/unslothai/unsloth)** — Efficient fine-tuning framework
