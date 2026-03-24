# AIMO3 — Fine-Tuning GPT-OSS-120B for Mathematical Reasoning

> **Competition**: [AI Mathematical Olympiad — Progress Prize 3](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3)
> **Goal**: Improve pass@1 consistency on olympiad-level math problems
> **Model**: GPT-OSS-120B (MoE, 120B parameters, 5.1B active)
> **Hardware**: 8× NVIDIA H100 (AWS)

---

## The Problem

The base GPT-OSS-120B scores **42/50** with majority voting (8 samples), but pass@1 is significantly lower. The model *can* solve nearly every problem — it just doesn't do so *consistently*.

| Metric | Score | What It Means |
|--------|-------|---------------|
| pass@1 | Low | Wrong reasoning path on first try |
| pass@8 majority vote | **42/50** | Correct answer exists in 8 attempts |
| pass@100 | ~50/50 | Model eventually solves everything |

**Goal**: Close the gap between pass@1 and pass@8 by teaching the model to consistently pick winning trajectories.

---

## Dataset — Nemotron-Math-v2 SFT v4

### Source
`nvidia/Nemotron-Math-v2` — 7M+ trajectories across 347K problems, solved under 6 configurations (high/medium/low reasoning × with/without Python tools).

### Filters Applied

| Filter | Value | Why |
|--------|-------|-----|
| **Splits** | `high_part00`, `high_part01`, `high_part02` | High reasoning depth trajectories = best quality |
| **Source** | ALL (AoPS + StackExchange) | Hard math is hard regardless of source |
| **Accuracy key** | `reason_high_with_tool` | Matches competition inference: `ReasoningEffort.HIGH` + tools |
| **Difficulty** | `acc < 0.5` | Only problems where majority vote fails |
| **Dedup** | Shortest trajectory per problem | Less noise, cleaner signal |

### Dataset Stats (v4)

| Source | Count | % |
|--------|-------|---|
| StackExchange | ~66K | 83.4% |
| AoPS | ~13K | 16.6% |
| **Total unique** | **79,374** | |

| Accuracy | Count | Meaning |
|----------|-------|---------|
| 0.125 | 30,363 | Model gets it right 1/8 — very hard |
| 0.250 | 23,000 | Model gets it right 2/8 — hard |
| 0.375 | 26,011 | Model gets it right 3/8 — moderate |

| Metric | Value |
|--------|-------|
| \\boxed answer | 99.0% |
| Code blocks | 30.3% |
| Median words | 287 |
| p90 words | 618 |

### Upsampling Strategy

Hardest problems are oversampled so the model sees them more during training:

```
acc=0.125 → 4× (30K × 4 = 121K)
acc=0.250 → 2× (23K × 2 =  46K)
acc=0.375 → 1× (26K × 1 =  26K)
                          ------
Effective training size:  ~193K examples
```

---

## Training — Full Fine-Tuning on 8×H100

**Method**: Full parameter training (no LoRA, no quantization)
**Infra**: DeepSpeed ZeRO Stage 3 across 8×H100 (640GB VRAM)

| Setting | Value |
|---------|-------|
| Precision | bf16 |
| Batch size | 2/GPU × 8 GPUs × 2 grad_accum = **32 effective** |
| Learning rate | 2e-5 (cosine decay) |
| Warmup | 3% |
| Epochs | 1 |
| Max seq len | 4096 |
| Packing | ✅ |
| Gradient clipping | 1.0 |
| Checkpoints | Every 200 steps, keep last 3 |
| Logging | W&B (`aimo3-full-ft-8xh100`) |

### Launch

```bash
# Install
pip install deepspeed transformers trl datasets accelerate torch wandb
pip install flash-attn --no-build-isolation

# Build dataset (saves locally + uploads to Kaggle)
python3.12 training/data/prepare_dataset_v4.py

# Train
deepspeed --num_gpus=8 training/sft/train_full_8xh100.py
```

---

## Repository Structure

```
├── training/
│   ├── sft/
│   │   └── train_full_8xh100.py    ← Full FT script (DeepSpeed ZeRO-3)
│   ├── data/
│   │   ├── prepare_dataset_v4.py   ← Dataset builder (current)
│   │   └── prepare_dataset_v3.py   ← Legacy (AoPS-only)
│   ├── dpo/                        ← DPO scripts
│   └── eval/                       ← Evaluation scripts
├── eda/
│   ├── eda_nemotron_raw.py         ← Raw HuggingFace EDA
│   └── eda_check.py                ← Kaggle dataset EDA
├── data/                           ← Local dataset outputs (gitignored)
└── README.md
```

---

## Datasets on Kaggle

| Dataset | Version | Description |
|---------|---------|-------------|
| [nemotron-math-v2-sft-v4](https://www.kaggle.com/datasets/ritwikakancharla/nemotron-math-v2-sft-v4) | v4 (current) | All sources, 79K hard, high reasoning |
| [nemotron-math-v2-sft-v3](https://www.kaggle.com/datasets/ritwikakancharla/nemotron-math-v2-sft-v3) | v3 (legacy) | AoPS-only, 24K |
| [nemotron-math-v2-sft-hard-tools](https://www.kaggle.com/datasets/ritwikakancharla/nemotron-math-v2-sft-hard-tools) | v2 (legacy) | Incorrect filters |

---

## License

MIT — Source data: `nvidia/Nemotron-Math-v2` (CC BY 4.0 / CC BY-SA 4.0)
