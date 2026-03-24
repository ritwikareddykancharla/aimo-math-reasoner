# AIMO3 — Fine-Tuning GPT-OSS-120B for Mathematical Reasoning

> **Competition**: [AI Mathematical Olympiad — Progress Prize 3](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3)
> **Goal**: Improve pass@1 consistency on olympiad-level math problems
> **Model**: GPT-OSS-120B (MoE, 120B parameters)
> **Hardware**: 8× NVIDIA H100 (AWS) + 1× NVIDIA H200

---

## The Problem

The base GPT-OSS-120B model already scores **42/50** with majority voting (8 samples). However, pass@1 is significantly lower — the model *can* solve nearly every problem, it just doesn't do so *consistently*.

| Metric | Score | What It Means |
|--------|-------|---------------|
| pass@1 | Low | Model often picks wrong reasoning path first try |
| pass@8 majority vote | **42/50** | Correct answer exists in 8 attempts |
| pass@100 | ~50/50 | Given enough tries, model solves everything |

**Goal**: Close the gap between pass@1 and pass@8 — teach the model to pick the right trajectory the first time.

---

## Training Approach

> *Don't teach the model new knowledge — teach it to consistently choose the trajectories it already knows work.*

We fine-tune **only on hard AoPS competition problems where majority voting fails** (`reason_high_with_tool < 0.5`). These are problems where fewer than 4 out of 8 high-reasoning-depth attempts produce the correct answer.

### Methods (in order of priority)

| Method | Strategy |
|--------|----------|
| **SFT** | Train on correct high-reasoning trajectories for hard problems |
| **GRPO** | RL reward for correct answers, penalize incorrect; group relative optimization |
| **DPO** | Prefer high-quality trajectories over low-quality ones (Gemini judge) |

---

## Dataset — Nemotron-Math-v2 SFT v3

### Why This Filtering

The `nvidia/Nemotron-Math-v2` dataset contains 7M+ trajectories across 347K problems, solved under 6 configurations (high/medium/low reasoning × with/without Python tools). We apply strict filters to get only the most relevant training signal:

| Filter | Value | Why |
|--------|-------|-----|
| **Splits** | `high_part00`, `high_part01` only | High reasoning depth trajectories = best quality. `high_part02` is 100% StackExchange → skipped. `medium` = lower quality trajectories for the same problems. |
| **Source** | `aops` only | AoPS = Art of Problem Solving = competition math. StackExchange is university/research math — different domain from AIMO. |
| **Accuracy key** | `reason_high_with_tool` | Competition inference uses `ReasoningEffort.HIGH` + Python tools. This is the exact evaluation setting. |
| **Difficulty** | `acc < 0.5` | Only problems where majority vote at high reasoning fails. Easy problems (acc ≥ 0.5) don't need SFT. |
| **Trajectories** | With-tool only | Competition notebook always has Python sandbox available. |
| **Dedup** | Shortest correct trajectory per problem | Less noise, faster inference, cleaner signal. |

### Dataset Stats (v3)

| Accuracy | Count | What It Means |
|----------|-------|---------------|
| 0.125 | ~20K | Model gets it right 1/8 times — very hard |
| 0.25 | ~30K | Model gets it right 2/8 times — hard |
| 0.375 | ~47K | Model gets it right 3/8 times — moderate |
| **Total** | **~97K** | Hard AoPS problems, high-reasoning trajectories |

> Source splits: `high_part00` (100% AoPS, ~40K hard) + `high_part01` (AoPS subset, ~57K hard)

### Raw Data Breakdown (from EDA)

| Split | Total Rows | AoPS | StackExchange | Hard @ high_with_tool | Used? |
|-------|-----------|------|---------------|----------------------|-------|
| high_part00 | 696K | **100%** | 0% | 40K (5.8%) | ✅ |
| high_part01 | 1.07M | 10.8% | 89.2% | 57K AoPS hard | ✅ AoPS only |
| high_part02 | 1.10M | 0% | **100%** | — | ❌ Skip |
| medium | 2.50M | 29.6% | 70.4% | — | ❌ Lower quality trajectories |

**Note on "medium" split**: Despite having 739K AoPS problems, the `medium` split contains medium-depth reasoning trajectories for those same problems. Since competition inference runs at `ReasoningEffort.HIGH`, training on medium-depth trajectories would teach the model to produce weaker reasoning chains. Always prefer high-depth trajectories.

---

## Curriculum Training

```python
import pandas as pd, json
df = pd.read_parquet("data/nemotron-sft-v3/data.parquet")
df["messages"] = df["messages"].apply(json.loads)

stage1 = df[df["accuracy"] == 0.125]  # Hardest — model almost never solves
stage2 = df[df["accuracy"] == 0.25]   # Hard
stage3 = df[df["accuracy"] == 0.375]  # Moderate — model sometimes solves
```

Train stage1 → stage2 → stage3 with decreasing learning rate. This forces the model to internalize the hardest patterns first.

---

## Infrastructure

| Resource | Spec | Purpose |
|----------|------|---------|
| **AWS 8×H100** | 640GB VRAM | Full fine-tuning (DeepSpeed ZeRO-3) |
| **GPU Droplet H200** | 141GB VRAM | Single-GPU QLoRA experiments |
| **Kaggle** | P100/T4 | Competition inference & submission |

### Training Config (8×H100)
- DeepSpeed ZeRO Stage 3 with CPU offloading
- Accelerate multi-GPU orchestration
- bf16 precision, gradient checkpointing

---

## Repository Structure

```
├── training/
│   ├── sft/                    # SFT scripts
│   ├── dpo/                    # DPO scripts (iterative with Gemini judge)
│   ├── data/
│   │   ├── prepare_dataset_v3.py   # ← Dataset builder (this creates the dataset)
│   │   └── prepare_dataset_v2.py   # Legacy
│   └── eval/
├── eda/
│   ├── eda_nemotron_raw.py     # Raw HuggingFace EDA (run first to explore data)
│   └── eda_check.py            # Kaggle dataset EDA
├── scripts/                    # accelerate_config.yaml, DeepSpeed config
├── data/                       # Local dataset outputs (gitignored — lives on Kaggle)
└── README.md
```

---

## Getting Started

```bash
git clone git@github.com:ritwikareddykancharla/aimo-math-reasoner.git
cd aimo-math-reasoner
cp .env.example .env  # Add HF_TOKEN, KAGGLE_API_TOKEN, KAGGLE_USERNAME

# 1. Explore raw data
python3.12 eda/eda_nemotron_raw.py --split high_part00 --max-rows 5000

# 2. Build dataset (saves to ./data/ + uploads to Kaggle)
python3.12 training/data/prepare_dataset_v3.py

# 3. Train (8×H100)
accelerate launch --config_file scripts/accelerate_config.yaml \
    training/sft/train_full_8xh100.py
```

---

## Dataset on Kaggle

- [nemotron-math-v2-sft-v3](https://www.kaggle.com/datasets/ritwikakancharla/nemotron-math-v2-sft-v3) — v3 (current, corrected)
- [nemotron-math-v2-sft-hard-tools](https://www.kaggle.com/datasets/ritwikakancharla/nemotron-math-v2-sft-hard-tools) — v2 (legacy, incorrect filters)

---

## License

MIT — Source data: `nvidia/Nemotron-Math-v2` (CC BY 4.0)
