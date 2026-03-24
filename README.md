# AIMO3 — Fine-Tuning GPT-OSS-120B for Mathematical Reasoning

> **Competition**: [AI Mathematical Olympiad — Progress Prize 3](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3)
> **Goal**: Improve pass@1 consistency on olympiad-level math problems
> **Model**: GPT-OSS-120B (MoE, 120B parameters, 5.1B active)
> **Hardware**: 8× NVIDIA H100 (AWS)

---

## Quick Start (New Cluster)

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/aimo-math-reasoner.git
cd aimo-math-reasoner
```

> The GRPO training parquets are already in the repo under `data/` — no separate download needed.

### 2. Install system dependencies

```bash
# Amazon Linux / RHEL
sudo dnf install -y python3.12 python3.12-pip git wget curl

# Ubuntu / Debian
sudo apt-get install -y python3.12 python3.12-pip git wget curl
```

### 3. Install Python packages

```bash
# Core training stack
pip3.12 install verl --break-system-packages
pip3.12 install vllm==0.12.0 --break-system-packages
pip3.12 install ray --break-system-packages
pip3.12 install torch torchvision torchaudio --break-system-packages
pip3.12 install triton --upgrade --break-system-packages

# HuggingFace
pip3.12 install transformers accelerate huggingface_hub --break-system-packages

# Data + logging
pip3.12 install pandas pyarrow wandb --break-system-packages

# FlashInfer (optional but recommended — improves MXFP4 MoE performance on H100)
pip3.12 install flashinfer --break-system-packages
```

### 4. Set up credentials

```bash
# WandB (training monitoring)
wandb login

# HuggingFace (to download model)
export HF_TOKEN=your_token_here
# or: huggingface-cli login

# Kaggle (only needed if uploading checkpoints)
mkdir -p ~/.kaggle
echo '{"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

### 5. Download the model

```bash
# Uses unsloth/gpt-oss-120b (bf16/U8) — works with standard vLLM
# NOTE: Do NOT use openai/gpt-oss-120b — it requires custom MXFP4 kernels
#       (vllm==0.10.1+gptoss) that are incompatible with standard veRL
python3.12 -c "
from huggingface_hub import snapshot_download
snapshot_download('unsloth/gpt-oss-120b', max_workers=8)
print('Done!')
"
# ~65GB download, takes ~5-10 mins on AWS (same region as HuggingFace CDN)
```

### 6. Verify GPU setup

```bash
nvidia-smi
python3.12 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

### 7. Fix conflicting env vars (critical!)

vLLM's `cumem_allocator` is **incompatible** with `expandable_segments:True`.
Unset these before training:

```bash
unset PYTORCH_CUDA_ALLOC_CONF
unset PYTORCH_ALLOC_CONF
```

Add to `~/.bashrc` to make permanent:
```bash
echo "unset PYTORCH_CUDA_ALLOC_CONF" >> ~/.bashrc
echo "unset PYTORCH_ALLOC_CONF" >> ~/.bashrc
source ~/.bashrc
```

### 8. Launch training

```bash
# Always use nohup — training takes 12-18h per round
# SSH disconnects will kill the job without it

# Round 1
nohup python3.12 training/grpo/train_grpo.py --round 1 > ~/round1.log 2>&1 &
echo "PID: $!"
tail -f ~/round1.log

# Round 2 (after round 1 finishes)
nohup python3.12 training/grpo/train_grpo.py --round 2 > ~/round2.log 2>&1 &

# Round 3
nohup python3.12 training/grpo/train_grpo.py --round 3 > ~/round3.log 2>&1 &
```

### 9. Monitor

```bash
# GPU utilization
watch -n 5 nvidia-smi

# Training logs
tail -f ~/round1.log

# Check if process is still alive
ps aux | grep train_grpo

# WandB dashboard
# https://wandb.ai/ritwikareddykancharla-n-a/aimo3-grpo
```

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

| Round | Accuracy | AoPS Count | Meaning | Est. Time |
|-------|----------|------------|---------|-----------|
| 1 | 0.125 | 5,311 | Wrong 7/8 — hardest | ~12-18h |
| 2 | 0.250 | 3,769 | Wrong 6/8 — hard | ~9-13h |
| 3 | 0.375 | 4,123 | Wrong 5/8 — moderate | ~10-15h |
| **Total** | | **13,203** | All AoPS hard problems | ~31-46h |

---

## Dataset — Nemotron-Math-v2 SFT v4

### Source
`nvidia/Nemotron-Math-v2` — 7M+ trajectories across 347K problems, solved under 6 configurations
(high/medium/low reasoning × with/without Python tools).

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

### GRPO Parquets (already in repo)

| File | Accuracy | Count | Round |
|------|----------|-------|-------|
| `data/grpo_aops_acc0125.parquet` | 0.125 | 5,311 | Round 1 |
| `data/grpo_aops_acc025.parquet` | 0.250 | 3,769 | Round 2 |
| `data/grpo_aops_acc0375.parquet` | 0.375 | 4,123 | Round 3 |
| `data/grpo_hard_aops.parquet` | all | ~13K | Full AoPS hard set |

Each parquet includes `gold_solution` for partial credit reward scoring.

---

## Training

### Phase 1 — SFT Full Fine-Tuning (completed)

| Setting | Value |
|---------|-------|
| Precision | bf16 |
| Batch size | 2/GPU × 8 GPUs × 2 grad_accum = **32 effective** |
| Learning rate | 2e-5 (cosine decay) |
| Warmup | 3% |
| Epochs | 1 |
| Max seq len | 4096 |
| Packing | ✅ |
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

| Setting | Value |
|---------|-------|
| Framework | veRL + vLLM 0.12.0 |
| Model | unsloth/gpt-oss-120b (bf16/U8) |
| Rollouts per problem | 8 |
| Generation | bf16 (vLLM) |
| Training | bf16 (FSDP) |
| Learning rate | 5e-7 |
| KL penalty (beta) | 0.001 |
| Max completion length | 2048 |
| vLLM gpu_memory_utilization | 0.2 |
| Checkpoint | model weights only (no optimizer states) |

Reward function (`training/grpo/reward_fn.py`):
```
Correct answer:          +1.0
Wrong + has reasoning:   -0.5
No \boxed{} at all:      -1.0
```

```bash
python3.12 training/grpo/train_grpo.py --round 1  # ~12-18h
python3.12 training/grpo/train_grpo.py --round 2  # ~9-13h
python3.12 training/grpo/train_grpo.py --round 3  # ~10-15h
```

---

## Known Issues & Fixes

### `expandable_segments` conflict with vLLM
vLLM's `cumem_allocator` asserts that `expandable_segments:True` is NOT set.
**Fix**: `train_grpo.py` automatically pops both `PYTORCH_CUDA_ALLOC_CONF` and
`PYTORCH_ALLOC_CONF` from the environment before launching veRL.

### MXFP4 kernel warning (harmless)
```
MXFP4 quantization requires Triton and kernels... will default to dequantizing to bf16
```
Expected — `unsloth/gpt-oss-120b` inherits `quantization_config: mxfp4` from the base
model's `config.json`, but its weights are already bf16/U8. The warning is harmless,
loading proceeds correctly in bf16.

### Do NOT use `openai/gpt-oss-120b` for training
The original model requires a custom vLLM build (`vllm==0.10.1+gptoss`) with private
MXFP4 kernels. Without them, it loads in fp32 causing Flash Attention errors and OOM.
Use `unsloth/gpt-oss-120b` instead — same model, dequantized to bf16/U8, works out of the box.

### vLLM OOM on wake_up / weight sync
If OOM occurs during `rollout.resume(tags=["weights"])` (after a training step),
lower `gpu_memory_utilization` further:
```python
"actor_rollout_ref.rollout.gpu_memory_utilization=0.15",
```
The cumem_allocator pre-reserves virtual memory at init — if FSDP consumes that space
during training, wake_up fails when vLLM tries to reclaim it.

---

## Repository Structure

```
├── training/
│   ├── sft/
│   │   └── train_full_8xh100.py       ← Full FT script (DeepSpeed ZeRO-3)
│   ├── grpo/
│   │   ├── train_grpo.py              ← GRPO launcher — use this ✅
│   │   ├── reward_fn.py               ← GRPO reward function
│   │   └── prepare_grpo_datasets.py   ← Rebuilds GRPO parquets
│   ├── data/
│   │   ├── prepare_dataset_v4.py      ← SFT dataset builder (current)
│   │   └── prepare_dataset_v3.py      ← Legacy (AoPS-only)
│   ├── dpo/                           ← DPO scripts
│   └── eval/                          ← Evaluation scripts
├── eda/
│   ├── eda_nemotron_raw.py
│   └── eda_check.py
├── data/                              ← GRPO parquets (tracked in git)
│   ├── grpo_aops_acc0125.parquet      ← Round 1 (5,311 problems)
│   ├── grpo_aops_acc025.parquet       ← Round 2 (3,769 problems)
│   ├── grpo_aops_acc0375.parquet      ← Round 3 (4,123 problems)
│   └── grpo_hard_aops.parquet         ← Full AoPS hard set
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

Dashboard: https://wandb.ai/ritwikareddykancharla-n-a/aimo3-grpo

---

## Datasets on Kaggle

| Dataset | Version | Description |
|---------|---------|-------------|
| [nemotron-math-v2-sft-v4](https://www.kaggle.com/datasets/ritwikakancharla/nemotron-math-v2-sft-v4) | v4 (current) | All sources, 79K hard, high reasoning |
| [nemotron-math-v2-sft-v3](https://www.kaggle.com/datasets/ritwikakancharla/nemotron-math-v2-sft-v3) | v3 (legacy) | AoPS-only, 24K |

---

## Requirements

```
Python:       3.12
veRL:         latest
vLLM:         0.12.0
PyTorch:      2.x
CUDA:         12.x
Triton:       3.5.0+
Transformers: 4.57+
```

---

## License

MIT — Source data: `nvidia/Nemotron-Math-v2` (CC BY 4.0 / CC BY-SA 4.0)
