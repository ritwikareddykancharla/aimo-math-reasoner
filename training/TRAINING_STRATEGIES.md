# AIMO3 Training Strategies — Complete Documentation

> **Competition**: AI Mathematical Olympiad — Progress Prize 3 (Kaggle)
> **Hardware**: 1× NVIDIA H200 (141 GB) GPU Droplet
> **Base Models**: GPT-OSS-120B (MoE, 4-bit QLoRA) • Qwen3-32B

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [1. Dataset Preparation](#1-dataset-preparation)
- [2. Supervised Fine-Tuning (SFT)](#2-supervised-fine-tuning-sft)
- [3. Direct Preference Optimization (DPO)](#3-direct-preference-optimization-dpo)
- [4. Evaluation](#4-evaluation)
- [5. Inference](#5-inference)
- [6. Version History & Lessons Learned](#6-version-history--lessons-learned)
- [File Reference](#file-reference)

---

## Overview

The training pipeline follows a multi-stage approach to produce a competitive math reasoning model for AIMO3:

```
Raw Data (Nemotron Math v2)
  → Dataset Preparation (filtering, compression, curriculum splits)
    → Supervised Fine-Tuning (SFT) on hard problems
      → Direct Preference Optimization (DPO) with Gemini judge
        → Merged 4-bit model → Kaggle submission
```

Two model families were explored:
- **GPT-OSS-120B**: Mixture-of-Experts model, primary focus for SFT
- **Qwen3-32B**: Dense model, used for DPO experiments

---

## Pipeline Architecture

```
┌──────────────────────────────────────────────────────┐
│                DATASET PREPARATION                    │
│  nvidia/Nemotron-Math-v2 (HuggingFace)               │
│  → Filter: HIGH reasoning depth, accuracy < 0.5      │
│  → Compress: Multi-turn → 2-turn (inline tool calls) │
│  → Deduplicate: Keep shortest trajectory per problem  │
│  → Tag: with_tools / without_tools                   │
│  → Split by accuracy: 0.125, 0.25, 0.375             │
└───────────────────────┬──────────────────────────────┘
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
┌─────────────────────┐   ┌─────────────────────────┐
│   SFT (GPT-OSS-120B)│   │   DPO (Qwen3-32B)       │
│   4-bit QLoRA, LoRA  │   │   3 iterative rounds     │
│   r=32, 7 targets    │   │   LoRA r=64              │
│                      │   │                          │
│  v1-v2: Full 130K    │   │  R1: Direct pairs        │
│  v3-v5: Hard+tools   │   │  R2: Gemini-judged       │
│  v6: Curriculum      │   │  R3: High-temp Gemini    │
│      (3 stages)      │   │                          │
└──────────┬───────────┘   └────────────┬─────────────┘
           │                            │
           ▼                            ▼
┌──────────────────────────────────────────────────────┐
│              MERGE & UPLOAD                           │
│  merged_4bit_forced → Kaggle Model Hub → Inference    │
└──────────────────────────────────────────────────────┘
```

---

## 1. Dataset Preparation

### Source
**nvidia/Nemotron-Math-v2** — Large-scale math reasoning trajectories with verified answers.

### Filtering Strategy (v2 — Final)
| Filter | Value | Rationale |
|--------|-------|-----------|
| Reasoning depth | `high` only | Better signal for olympiad-level problems |
| Accuracy | `< 0.5` (i.e., 0.125, 0.25, 0.375) | Focus on genuinely hard problems the base model struggles with |
| Tool usage | Both with-tools and without-tools | Tagged for flexible training |
| Deduplication | Shortest trajectory per problem UUID | Less noise, more concise reasoning |

### Compression
Multi-turn conversation threads (user → assistant → tool → assistant → ...) are compressed into a **2-turn format**:
- **Turn 1 (user)**: Original problem
- **Turn 2 (assistant)**: Complete solution with tool calls and outputs inlined

### Output Columns
| Column | Description |
|--------|-------------|
| `uuid` | Unique problem identifier |
| `messages` | 2-turn compressed `[user, assistant]` |
| `expected_answer` | Verified integer answer (0–99999) |
| `data_source` | Problem source/origin |
| `accuracy` | Base model accuracy: 0.125, 0.25, or 0.375 |
| `reasoning_depth` | `"high"` or `"medium"` |
| `has_tools` | Boolean — whether tool calls are present |

### Answer Verification (`check_answers.py`)
Multiple extraction patterns applied in priority order:
1. `\boxed{N}` — standard LaTeX boxed answer
2. `"the answer is N"` / `"answer: N"` — natural language patterns
3. Last standalone integer in final 5 lines — fallback

### Scripts
- **`prepare_dataset.py`** (v1): Initial version, both with/without tools
- **`prepare_dataset_v2.py`** (v2): HIGH-split only, with-tools only, improved compression
- **`check_answers.py`**: Answer extraction & validation pipeline

---

## 2. Supervised Fine-Tuning (SFT)

### Model Configuration (All Versions)
| Parameter | Value |
|-----------|-------|
| Base model | `unsloth/gpt-oss-120b-unsloth-bnb-4bit` |
| Quantization | 4-bit (BitsAndBytes NF4) |
| LoRA rank | 32 |
| LoRA alpha | 32 |
| LoRA dropout | 0 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Gradient checkpointing | `unsloth` (memory-efficient) |
| Precision | bf16 |
| Seed | 42 |

### Version Evolution

#### v1 (`train_sft_120b_old.py`) — Baseline
- **Dataset**: Full 130K Nemotron Math v2
- **Sequence length**: Not optimized
- **Split**: First 2000 samples for eval, rest for training
- **Merge**: Basic `merged_4bit`

#### v2 (`train_sft_120b.py` / `train_sft_120b_v2.py`) — Checkpointing
- **Dataset**: Full 130K Nemotron Math v2
- **Key change**: Periodic merge + upload to Kaggle every 500 steps
- **Merge strategy**: Full merge → 4-bit → upload → delete local (disk space management on H200)

#### v3–v5 (`train_sft_120b_v3.py` through `train_sft_120b_v5.py`) — Hard Problems Focus
**Major improvements**:
1. ✅ Keep ALL message turns (user/assistant/tool) — not just first 2
2. ✅ `max_seq_length` = 8192 (was 2048 — was cutting off hard problems)
3. ✅ Merge uses `merged_4bit_forced` (was failing with `merged_4bit` on MoE models)
4. ✅ Batch size adjusted for longer sequences
5. ✅ `warmup_ratio` = 0.03 for training stability

**Dataset**: Filtered to hard problems with tool usage only

**Checkpointing strategy**:
- Every 500 steps: LoRA adapter save (fast, small)
- Every 1000 steps: full merge → `4bit_forced` → upload to Kaggle → delete local
- Final: full merge + upload

**Training config (v5)**:
| Parameter | Value |
|-----------|-------|
| `max_seq_length` | 4096 (compressed data fits) |
| `per_device_train_batch_size` | 32 |
| `gradient_accumulation_steps` | 1 |
| `learning_rate` | 2e-5 |
| `lr_scheduler` | cosine |
| `warmup_ratio` | 0 |
| `max_grad_norm` | 0.5 |
| `save_steps` | 200 |
| Packing | enabled |

#### v6 (`train_sft_120b_v6.py`) — Curriculum Learning (Final Strategy)
**Key innovation**: 3-stage curriculum training, starting with the hardest problems:

| Stage | Accuracy | Label | Learning Rate | Epochs | Rationale |
|-------|----------|-------|---------------|--------|-----------|
| 1 | 0.125 | Hardest | 2e-5 | 2 | Smallest set, needs more exposure |
| 2 | 0.25 | Hard | 1e-5 | 1 | Medium difficulty |
| 3 | 0.375 | Easiest-hard | 5e-6 | 1 | Refinement on "easier" hard problems |

**Curriculum flow**:
```
Base model (unsloth/gpt-oss-120b-unsloth-bnb-4bit)
  → Stage 1: Train on acc=0.125 → merge → upload
    → Stage 2: Load stage1 → train on acc=0.25 → merge → upload
      → Stage 3: Load stage2 → train on acc=0.375 → merge → upload
```

Each stage:
1. Loads the previous stage's merged model (or base model for stage 1)
2. Applies QLoRA
3. Filters dataset by accuracy
4. Trains with `SFTTrainer` + packing
5. Merges to 4-bit → uploads to Kaggle → cleans local disk
6. Clears GPU memory for next stage

---

## 3. Direct Preference Optimization (DPO)

### Model
**Qwen3-32B** — Dense model, easier to work with for DPO preference learning.

### Configuration
| Parameter | Value |
|-----------|-------|
| Quantization | 4-bit QLoRA |
| LoRA rank | 64 |
| `max_seq_length` | 16384 |
| Fast inference | vLLM (standby mode via `UNSLOTH_VLLM_STANDBY`) |

### 3-Round Iterative DPO Pipeline

#### Round 1 — Direct Preference Pairs
- **Method**: Generate 2 responses per problem, compare against gold answer
- **Pair construction**: Correct response = chosen, incorrect = rejected; if both correct/incorrect, include gold answer as chosen
- **Learning rate**: 5e-7
- **Beta (KL penalty)**: 0.10

#### Round 2 — Gemini-Judged DPO
- **Method**: Generate candidate solutions, have **Gemini** judge quality (scoring 0–1)
- **Temperature**: 0.8 (moderate diversity)
- **Pair selection**: Only keep pairs where quality score difference ≥ 0.2
- **Learning rate**: 3e-7 (reduced — fine-grained optimization)
- **Beta**: 0.12 (increased KL penalty to prevent drift)

#### Round 3 — High-Temperature Gemini-Judged DPO
- **Method**: Same as Round 2 but with higher temperature (0.9) for more diverse solutions
- **Learning rate**: 2e-7 (further reduced)
- **Beta**: 0.15 (strongest KL constraint)

### DPO Training Config (Per Round)
| Parameter | Value |
|-----------|-------|
| `per_device_train_batch_size` | 4 |
| `gradient_accumulation_steps` | 4 (effective batch = 16) |
| `num_train_epochs` | 1 |
| `lr_scheduler` | cosine |
| `warmup_ratio` | 0.1 |
| `max_grad_norm` | 0.3 |

### Evaluation Between Rounds
- **pass@1** evaluation on 200 held-out problems after each round
- Tracks: `baseline → Round 1 → Round 2 → Round 3` accuracy progression
- Results logged to W&B for monitoring

---

## 4. Evaluation

### SFT Evaluation (`eval_sft.py`)
- **Metric**: pass@1 on held-out Nemotron Math v2 problems
- **Method**: Generate solution with vLLM, extract integer answer, compare to ground truth
- **Configurable**: Number of problems, number of samples per problem
- **Breakdown**: Results by confidence level

```bash
python3 eval_sft.py --model ./merged-4bit-step-500
python3 eval_sft.py --model ./merged-4bit-step-500 --n_problems 100 --n_samples 8
```

### Competition Evaluation
Final evaluation is done via Kaggle submission notebooks that load the merged model from Kaggle Model Hub and run inference on hidden test problems.

---

## 5. Inference

Inference is handled via **vLLM** serving the merged 4-bit model:
- Model loaded from Kaggle Model Hub
- Served via OpenAI-compatible API
- Key settings: `max_model_len=65536`, `kv_cache_dtype=fp8_e4m3`, `enable_prefix_caching=True`

### Known Issue
The vLLM BitsAndBytes loader requires **complete model checkpoints**. If LoRA adapters are saved without merging (via `save_pretrained()` instead of `save_pretrained_merged()`), the MoE expert/router weights will be missing, causing a `ValueError` at load time. Always use `save_pretrained_merged()` with `save_method="merged_4bit_forced"`.

---

## 6. Version History & Lessons Learned

| Version | Key Change | Lesson |
|---------|-----------|--------|
| SFT v1 | Full 130K dataset | Baseline — too much easy data dilutes hard problem learning |
| SFT v2 | Periodic merge+upload | Disk space on H200 is finite — merge and upload iteratively |
| SFT v3 | Multi-turn messages + seq_len=8192 | Truncating at 2048 tokens was cutting off solutions |
| SFT v4 | `merged_4bit_forced` | Standard `merged_4bit` fails on MoE architectures |
| SFT v5 | Compressed data + seq_len=4096 | Pre-compression avoids truncation with shorter context |
| SFT v6 | Curriculum learning (3 stages) | Training hardest-first with decreasing LR improves generalization |
| DPO v1 | Basic preference pairs | Simple correct/incorrect pairs provide weak signal |
| DPO v3 | Gemini-judged iterative | External judge enables quality-based pair selection |
| Dataset v2 | HIGH-split only + tools | Medium reasoning depth adds noise, tools help competition format |

### Key Takeaways
1. **MoE models need `merged_4bit_forced`** — standard merge methods fail silently
2. **Curriculum learning matters** — starting with hardest problems and decreasing LR per stage
3. **Data quality > quantity** — filtering to HIGH reasoning depth, hard accuracy, tool-use only
4. **Disk management is critical** on single-GPU setups — merge, upload, delete iteratively
5. **Gemini as a judge** provides better DPO pairs than simple correct/incorrect
6. **Always use `save_pretrained_merged()`** — never `save_pretrained()` alone for MoE inference

---

## File Reference

### Training Scripts
| File | Description |
|------|-------------|
| `train_sft_120b_old.py` | SFT v1 — baseline, full 130K dataset |
| `train_sft_120b.py` | SFT v1.5 — added checkpointing |
| `train_sft_120b_v2.py` | SFT v2 — periodic merge+upload |
| `train_sft_120b_v3.py` | SFT v3 — multi-turn fix, seq_len=8192 |
| `train_sft_120b_v4.py` | SFT v4 — merged_4bit_forced |
| `train_sft_120b_v5.py` | SFT v5 — compressed data, seq_len=4096 |
| `train_sft_120b_v6.py` | SFT v6 — curriculum learning (3 stages) |
| `train_sft.py` | Earlier SFT experiments |
| `train_sft_old.py` | Earliest SFT script |
| `train_dpo.py` | DPO v1 — basic Qwen3-32B |
| `train_dpo_old.py` | Earlier DPO experiment |
| `train_dpo_v2.py` | DPO v2 — improved pair generation |
| `train_dpo_v3.py` | DPO v3 — 3-round iterative with Gemini judge |

### Data & Evaluation
| File | Description |
|------|-------------|
| `prepare_dataset.py` | Dataset prep v1 — both with/without tools |
| `prepare_dataset_v2.py` | Dataset prep v2 — HIGH-split only, tools only |
| `check_answers.py` | Answer extraction & validation |
| `eval_sft.py` | SFT evaluation (pass@1) |

### EDA & Analysis
| File | Description |
|------|-------------|
| `deep_eda.py` | Deep exploratory analysis of datasets |
| `deep_eda2.py` | Extended EDA |
| `eda_nemotron.py` | Nemotron-specific analysis |
| `kaggle_eda.py` | Kaggle competition data analysis |
| `nemotron_eda.py` | Nemotron Math v2 exploration |

### Infrastructure
| File | Description |
|------|-------------|
| `setup_and_train.sh` | Environment setup + training launcher |
| `setup_and_train_old.sh` | Earlier setup script |
| `run_eval.sh` | Evaluation runner |
