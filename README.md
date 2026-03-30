# AIMO3 — Fine-Tuning GPT-OSS-120B for Mathematical Reasoning

> **Competition**: [AI Mathematical Olympiad — Progress Prize 3](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3)
> **Goal**: Improve pass@1 consistency on olympiad-level math problems
> **Model**: GPT-OSS-120B (MoE, 120B parameters, 5.1B active)
> **Hardware**: 2× AWS p5.48xlarge (16× NVIDIA H100 total)

---

## Quick Start (New Cluster)

### 1. Mount NVMe storage (do this first on both nodes)

```bash
# RAID0 across all 8 NVMe instance store drives (~28TB total)
sudo dnf install -y mdadm
sudo mdadm --create /dev/md0 --level=0 --raid-devices=8 \
    /dev/nvme1n1 /dev/nvme2n1 /dev/nvme3n1 /dev/nvme4n1 \
    /dev/nvme5n1 /dev/nvme6n1 /dev/nvme7n1 /dev/nvme8n1
sudo mkfs.ext4 -F /dev/md0
sudo mkdir -p /data
sudo mount /dev/md0 /data
sudo chown $(whoami):$(whoami) /data
```

### 2. Clone the repo

```bash
cd /data
git clone https://github.com/YOUR_USERNAME/aimo-math-reasoner.git
cd aimo-math-reasoner
```

### 3. Install system dependencies

```bash
# Amazon Linux 2023 (curl conflicts with curl-minimal — exclude it)
sudo dnf install -y python3.12 python3.12-pip python3.12-devel git wget
```

### 4. Install Python packages (run on both nodes simultaneously)

```bash
# PyTorch (CUDA 12.8)
pip3.12 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Ray
pip3.12 install "ray[default]==2.54.1"

# verl + SGLang (SGLang replaces vLLM for faster rollout inference)
pip3.12 install verl
pip3.12 install "sglang[all]==0.5.9" --find-links https://flashinfer.ai/whl/cu128/torch2.9/

# HuggingFace + misc
pip3.12 install transformers accelerate peft datasets huggingface_hub openai \
    sentencepiece protobuf pandas pyarrow wandb
```

### 5. Set up credentials

```bash
# WandB (training monitoring)
wandb login

# HuggingFace (to download model)
export HF_TOKEN=your_token_here
# or: huggingface-cli login
```

### 6. Download the model

```bash
python3.12 -c "
from huggingface_hub import snapshot_download
snapshot_download('unsloth/gpt-oss-120b', local_dir='/data/models/gpt-oss-120b', max_workers=8)
print('Done!')
"
# ~65GB download, takes ~5-10 mins on AWS
```

### 7. Fix conflicting env vars (critical!)

```bash
echo "unset PYTORCH_CUDA_ALLOC_CONF" >> ~/.bashrc
echo "unset PYTORCH_ALLOC_CONF" >> ~/.bashrc
source ~/.bashrc
```

### 8. Apply the freeze patch

```bash
python3.12 training/grpo/apply_freeze_patch.py
```

### 9. Start Ray cluster

```bash
# On Node 1 (head)
ray start --head --num-gpus=8 --num-cpus=192 --port=6379

# On Node 2 (worker) — replace with your Node 1 private IP
ray start --address=NODE1_IP:6379 --num-gpus=8 --num-cpus=192

# Verify both nodes joined
ray status
```

### 10. Launch training

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

### 11. Monitor

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

### Why SGLang over vLLM for Rollout

SGLang is used for both training rollout and inference because:

- **RadixAttention** — KV cache is shared across all n=8 rollouts of the same prompt,
  since they share the same prefix. vLLM recomputes this prefix 8 times. SGLang does it once.
- **Async mode** — SGLang overlaps rollout generation with FSDP actor/ref forward passes,
  reducing idle GPU time during training.
- **Native multi-turn tool calling** — SGLang has first-class support for the gpt-oss
  agent loop format used in this pipeline.
- **Faster inference** — ~1.5-2x throughput improvement on multi-turn agentic workloads
  vs vLLM, allowing more majority votes within the same time budget.

For inference (Kaggle submission), SGLang's prefix caching gives a direct speedup since
all 8+ attempts share the same system prompt + problem prefix in KV cache.

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

---

## Training

### Phase 1 — SFT Full Fine-Tuning (compute constrained)

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

```bash
deepspeed --num_gpus=8 training/sft/train_full_8xh100.py
```

### Phase 2 — GRPO Curriculum RL (current)

| Setting | Value |
|---------|-------|
| Framework | veRL + SGLang 0.5.9 |
| Model | unsloth/gpt-oss-120b (65GB, MoE experts frozen) |
| Rollout engine | SGLang async, TP=8 per node |
| Rollouts per problem | 8 |
| Max prompt length | 2048 tokens |
| Max response length | 32768 tokens |
| Model context length | 65536 tokens |
| Sampling | temp=1.0, top_p=1.0, min_p=0.02 |
| Training | bf16 FSDP across 16 GPUs |
| Learning rate | 5e-7 |
| KL penalty (beta) | 0.001 |
| Frozen params | MoE expert layers (~63GB, 4-bit) |
| Trained params | Dense layers only (~2.13B, bf16) |

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

## Inference (Kaggle Submission)

The inference notebook uses SGLang as the serving backend (replacing vLLM) for faster
per-token generation and prefix cache reuse across majority vote attempts.

### Why more votes = better score
SGLang's ~1.5-2x speedup over vLLM means within the same time budget you can run
more majority vote attempts per problem:

```
vLLM:   8 attempts  → 42/50
SGLang: 12-16 attempts → potentially higher
```

### SGLang wheels for Kaggle (offline environment)

Since Kaggle disables internet access during inference, all packages must be
pre-downloaded as wheels and uploaded as a Kaggle dataset. See `scripts/build_wheels.sh`
for the wheel building script.

---

## Known Issues & Fixes

### NCCL fails cross-node on EFA instances
**Symptom**: `ncclRemoteError` or `ncclInternalError` during FSDP init.
**Fix**: Do NOT set `NCCL_IB_DISABLE=1` on EFA instances. The aws-ofi-nccl plugin
routes through the IB verbs path even on EFA. Setting it to 1 forces socket-only
fallback which fails during cross-node parameter broadcast.
```python
"NCCL_IB_DISABLE": "0",  # correct
```

### `expandable_segments` conflict
vLLM/SGLang's cumem allocator is incompatible with `expandable_segments:True`.
**Fix**: `train_grpo.py` automatically pops both env vars before launching veRL.

### MXFP4 kernel warning (harmless)
```
MXFP4 quantization requires Triton and kernels... will default to dequantizing to bf16
```
Expected — `unsloth/gpt-oss-120b` inherits `quantization_config: mxfp4` from the base
model's `config.json`, but its weights are already bf16/U8. Loading proceeds correctly.

### Do NOT use `openai/gpt-oss-120b` for training
The original model requires a custom vLLM build (`vllm==0.10.1+gptoss`) with private
MXFP4 kernels. Use `unsloth/gpt-oss-120b` instead — same model, dequantized to bf16/U8.

---

## Repository Structure

```
├── training/
│   ├── sft/
│   │   └── train_full_8xh100.py       ← Full FT script (DeepSpeed ZeRO-3)
│   ├── grpo/
│   │   ├── train_grpo.py              ← GRPO launcher (SGLang rollout) ✅
│   │   ├── apply_freeze_patch.py      ← Patches verl to freeze MoE experts
│   │   ├── reward_fn.py               ← GRPO reward function
│   │   ├── agent.yaml                 ← SGLang agent loop config
│   │   └── prepare_grpo_datasets.py   ← Rebuilds GRPO parquets
│   ├── data/
│   │   ├── prepare_dataset_v4.py      ← SFT dataset builder (current)
│   │   └── prepare_dataset_v3.py      ← Legacy (AoPS-only)
│   ├── dpo/                           ← DPO scripts
│   └── eval/                          ← Evaluation scripts
├── scripts/
│   └── build_wheels.sh                ← Builds SGLang wheels for Kaggle offline use
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

## Hardware

| Component | Spec |
|-----------|------|
| Instances | 2× AWS p5.48xlarge |
| GPUs | 16× NVIDIA H100 80GB HBM3 |
| CPU | 2× 192 vCPU |
| RAM | 2× 2TB |
| Storage | 2× 28TB NVMe RAID0 |
| Network | EFA + GDRDMA (3200 Gbps) |

---

## Requirements

```
Python:       3.12
veRL:         latest
SGLang:       0.5.9
PyTorch:      2.9.1+cu128
CUDA:         12.8
Ray:          2.54.1
Transformers: 4.57+
```

---

## License

MIT — Source data: `nvidia/Nemotron-Math-v2` (CC BY 4.0 / CC BY-SA 4.0)
