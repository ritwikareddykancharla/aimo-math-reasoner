# How veRL Loads Models onto GPUs (vLLM + GRPO)

Yes, TWO models get loaded — but it's smart about it 🧠

In veRL's GRPO pipeline, you essentially need:

```
┌─────────────────────────────────────────────────┐
│              veRL GRPO Pipeline                  │
│                                                  │
│  1. ACTOR (Training)    → FSDP (PyTorch)        │
│  2. ROLLOUT (Inference) → vLLM                  │
│  3. REFERENCE MODEL     → FSDP (frozen)         │
│  4. REWARD MODEL/FUNC   → could be rule-based   │
└─────────────────────────────────────────────────┘
```

## The Key Trick: Weight Sharing / Offloading 🔄

veRL does NOT naively load two full copies. Here's what actually happens:

### Timeline

**Phase 1: ROLLOUT (Generation)**
```
┌─────────────────────────────────────────┐
│  vLLM engine loaded on GPUs             │
│  Actor weights → vLLM format            │
│  Generate completions (sampling)        │
│  GPU Memory: ████████████░░░░ (~60-70%) │
└─────────────────────────────────────────┘
         │
         ▼  ← vLLM weights OFFLOADED / freed
```

**Phase 2: TRAINING (GRPO update)**
```
┌─────────────────────────────────────────┐
│  FSDP actor loaded on GPUs              │
│  FSDP reference model (frozen)          │
│  Compute loss, backprop, update         │
│  GPU Memory: █████████████░░░ (~70-80%) │
└─────────────────────────────────────────┘
         │
         ▼  ← FSDP weights synced back to vLLM
```

**Phase 3: ROLLOUT again (next iteration)**
```
┌─────────────────────────────────────────┐
│  Updated weights → vLLM format          │
│  Generate new completions               │
└─────────────────────────────────────────┘
... repeat ...
```

## So what's in GPU memory at each phase?

**During Rollout (vLLM):**
```
┌──────────────── GPU Memory ────────────────┐
│                                             │
│  vLLM Engine (tensor parallel)              │
│  ├── Model weights (sharded across GPUs)    │
│  ├── KV Cache (pre-allocated)               │
│  ├── Activation memory                      │
│  └── Sampling buffers                       │
│                                             │
│  FSDP Actor weights → CPU / offloaded  ❌   │
│  Reference model    → CPU / offloaded  ❌   │
└─────────────────────────────────────────────┘
```

**During Training (FSDP):**
```
┌──────────────── GPU Memory ────────────────┐
│                                             │
│  FSDP Actor (sharded across GPUs)           │
│  ├── Parameters                             │
│  ├── Gradients                              │
│  └── Optimizer states                       │
│                                             │
│  FSDP Reference Model (frozen, no grads)    │
│  ├── Parameters only (no optim states)      │
│                                             │
│  vLLM Engine → freed / offloaded  ❌        │
└─────────────────────────────────────────────┘
```

## This is why your memory is only 5.5GB right now

What you're seeing:

```
actor_rollout_init_model  ← This is the INITIALIZATION step

veRL is:
1. Loading checkpoint from disk → CPU RAM    ✅ (done)
2. Building FSDP wrapped model               🔄 (in progress)
3. Building vLLM engine                      ⏳ (waiting)
4. Setting up weight sync between them       ⏳ (waiting)

Only ~5.5GB on each GPU = just CUDA context +
partially staged parameters
```

## The Full Memory Lifecycle

Memory per GPU over time:

```
80GB ┤
     │
70GB ┤                    ┌──────┐         ┌──────┐
     │                    │TRAIN │         │TRAIN │
60GB ┤     ┌──────┐       │ FSDP │         │ FSDP │
     │     │vLLM  │       │+ref  │  ┌────┐ │+ref  │
50GB ┤     │rollout│      │      │  │vLLM│ │      │
     │     │      │       │      │  │    │ │      │
40GB ┤     │      │       │      │  │    │ │      │
     │     │      │       │      │  │    │ │      │
30GB ┤     │      │       │      │  │    │ │      │
     │     │      │       │      │  │    │ │      │
20GB ┤     │      │       │      │  │    │ │      │
     │     │      │       │      │  │    │ │      │
10GB ┤─────┘      └───────┘      └──┘    └─┘      │
     │ init                                        │
 0GB ┼────────────────────────────────────────────────
     t0   rollout  free   train  free  rollout  train
              ↑                    ↑
         weights loaded      weights synced
         into vLLM          back to vLLM
```

## For YOUR setup specifically

| Component | How it fits on 8x H100 |
|---|---|
| 116B MoE model (MXFP4) | ~30-40GB total after quantization |
| FSDP sharding across 8 GPUs | ~4-5GB weights per GPU |
| vLLM with tensor parallel=8 | ~4-5GB weights per GPU |
| KV Cache (vLLM) | Could be 20-40GB per GPU depending on batch/seq len |
| Optimizer states (training) | Adam: 2x params in FP32 ≈ ~8-10GB per GPU |
| Gradients | ~4-5GB per GPU |
| Peak during training | ~30-50GB per GPU |
| Peak during rollout | ~40-60GB per GPU (KV cache heavy) |

## TL;DR

veRL time-shares the GPU between vLLM (rollout) and FSDP (training). They don't coexist on GPU simultaneously. Weights are synced between them after each training step. You're currently in init — memory will jump soon. Your 8x H100 setup is plenty for this 117B MoE model with MXFP4. 🚀
```
