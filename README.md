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

## Strategy

### Why GRPO over SFT

SFT teaches the model to *imitate* correct solutions. GRPO teaches the model to *prefer* correct reasoning paths by contrasting wins vs losses within a group of rollouts — directly optimizing for consistency.
```
SFT:  "copy this correct solution"               → better format
GRPO: "your correct attempts scored higher,      → better reasoning
       do more of what made them correct"           consistency
```

For a model already scoring 42/50, the bottleneck is not capability — it's variance. GRPO directly addresses this.

### Curriculum RL — Three Rounds

Training proceeds from hardest → easier, each round starting from the previous checkpoint:

### AoPS Hard Subset (GRPO)

| Round | Accuracy | AoPS Count | Meaning | Est. Time |
|-------|----------|------------|---------|-----------|
| 1 | 0.125 | 5,311 | Wrong 7/8 — hardest | ~3h |
| 2 | 0.250 | 3,769 | Wrong 6/8 — hard | ~2h |
| 3 | 0.375 | 4,123 | Wrong 5/8 — moderate | ~2.5h |
| **Total** | | **13,203** | All AoPS hard problems | ~7.5h |

Each round uses AoPS-only problems (competition math = same distribution as AIMO3).

---

## Dataset — Nemotron-Math-v2 SFT v4

### Source
`nvidia/Nemotron-Math-v2` — 7M+ trajectories across 347K problems, solved under 6 configurations (high/medium/low reasoning × with/without Python tools).

### Filters Applied

| Filter | Value | Why |
|--------|-------|-----|
| **Splits** | `high_part00`, `high_part01`, `high_part02` | High reasoning depth = best quality |
| **Source** | ALL (AoPS + StackExchange) | Hard math is hard regardless of source |
| **Accuracy key** | `reason_high_with_tool` | Matches competition inference |
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

### AoPS Hard Subset (GRPO)

| Accuracy | AoPS Count | Used For |
|----------|------------|----------|
| 0.125 | 5,311 | GRPO Round 1 |
| 0.250 | TBD | GRPO Round 2 |
| 0.375 | TBD | GRPO Round 3 |

---

## Training

### Phase 1 — SFT Full Fine-Tuning (completed)

**Method**: Full parameter training (no LoRA, no quantization)
**Infra**: DeepSpeed ZeRO Stage 3 across 8×H100

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
| Dataset | 193K (79K hard + upsampling) |

Upsampling strategy:
```
acc=0.125 → 4× (30K × 4 = 121K)
acc=0.250 → 2× (23K × 2 =  46K)
acc=0.375 → 1× (26K × 1 =  26K)
                          ------
Effective training size:  ~193K examples
```
```bash
deepspeed --num_gpus=8 training/sft/train_full_8xh100.py
```

### Phase 2 — GRPO Curriculum RL (current)

**Method**: Group Relative Policy Optimization
**Infra**: VERL + vLLM on 8×H100
**Key insight**: GPT-OSS-120B is MoE (5.1B active params) — generation is fast even at 120B scale

| Setting | Value |
|---------|-------|
| Framework | VERL 0.7.1 + vLLM 0.12.0 |
| Rollouts per problem | 8 |
| Generation precision | MXFP4 (native, Triton 3.6.0) |
| Training precision | bf16 |
| Learning rate | 5e-7 |
| KL penalty (beta) | 0.001 |
| Max completion length | 2048 |
| Reward: correct | +1.0 |
| Reward: wrong + reasoning | -0.5 |
| Reward: no \boxed{} | -1.0 |
```bash
# Round 1
python3.12 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=./grpo_aops_acc0125.parquet \
  data.train_batch_size=32 \
  data.max_prompt_length=512 \
  data.max_response_length=2048 \
  actor_rollout_ref.model.path=openai/gpt-oss-120b \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_parallel_size=8 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.use_remove_padding=True \
  custom_reward_function.path=./reward_fn.py \
  custom_reward_function.name=compute_reward \
  trainer.n_gpus_per_node=8 \
  trainer.project_name=aimo3-grpo \
  trainer.experiment_name=gpt-oss-120b-round1

# Round 2 — from Round 1 checkpoint
# actor_rollout_ref.model.path=./grpo-checkpoint-round1/
# data.train_files=./grpo_aops_acc0250.parquet

# Round 3 — from Round 2 checkpoint  
# actor_rollout_ref.model.path=./grpo-checkpoint-round2/
# data.train_files=./grpo_aops_acc0375.parquet
```

---

## Repository Structure
```
├── training/
│   ├── sft/
│   │   └── train_full_8xh100.py       ← Full FT script (DeepSpeed ZeRO-3)
│   ├── grpo/
│   │   ├── reward_fn.py               ← GRPO reward function
│   │   └── prepare_grpo_datasets.py   ← Builds all 3 GRPO parquets
│   ├── data/
│   │   ├── prepare_dataset_v4.py      ← SFT dataset builder (current)
│   │   └── prepare_dataset_v3.py      ← Legacy (AoPS-only)
│   ├── dpo/                           ← DPO scripts
│   └── eval/                          ← Evaluation scripts
├── eda/
│   ├── eda_nemotron_raw.py            ← Raw HuggingFace EDA
│   └── eda_check.py                   ← Kaggle dataset EDA
├── grpo_aops_acc0125.parquet          ← GRPO Round 1 data (5,311 problems)
├── grpo_aops_acc0250.parquet          ← GRPO Round 2 data
├── grpo_aops_acc0375.parquet          ← GRPO Round 3 data
├── data/                              ← Local dataset outputs (gitignored)
└── README.md
```

---

## WandB Monitoring

| Metric | What To Watch |
|--------|---------------|
| `reward/mean` | Should trend up — model getting more correct |
| `reward/std` | Should stay > 0 — diversity maintained |
| `kl_divergence` | Should stay < 0.5 — not drifting from base |
| `loss` | Noisier than SFT, normal for RL |

---

## Datasets on Kaggle

| Dataset | Version | Description |
|---------|---------|-------------|
| [nemotron-math-v2-sft-v4](https://www.kaggle.com/datasets/ritwikakancharla/nemotron-math-v2-sft-v4) | v4 (current) | All sources, 79K hard, high reasoning |
| [nemotron-math-v2-sft-v3](https://www.kaggle.com/datasets/ritwikakancharla/nemotron-math-v2-sft-v3) | v3 (legacy) | AoPS-only, 24K |
| [nemotron-math-v2-sft-hard-tools](https://www.kaggle.com/datasets/ritwikakancharla/nemotron-math-v2-sft-hard-tools) | v2 (legacy) | Incorrect filters |

---

## Requirements
```
Python:     3.12
VERL:       0.7.1
vLLM:       0.12.0
Ray:        2.54.0
Triton:     3.6.0  ← required for MXFP4 (no dequantization)
PyTorch:    2.x
CUDA:       12.x
```

---

## License

MIT — Source data: `nvidia/Nemotron-Math-v2` (CC BY 4.0 / CC BY-SA 4.0)
