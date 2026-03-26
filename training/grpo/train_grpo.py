"""
AIMO3 GRPO Training Launcher

Usage:
  Round 1: python3.12 training/grpo/train_grpo.py --round 1
  Round 2: python3.12 training/grpo/train_grpo.py --round 2
  Round 3: python3.12 training/grpo/train_grpo.py --round 3

Model:
  Uses unsloth/gpt-oss-120b (bf16/U8) — works with standard vLLM/veRL.
  openai/gpt-oss-120b requires custom MXFP4 kernels not available here.

Strategy:
  - vLLM rollout  → bf16, gpu_memory_utilization=0.1, max_model_len=4096
                    (max_prompt=512 + max_response=2048 = 2560 tokens needed;
                     131072 default forces 32× more KV cache profiling → OOM)
  - FSDP actor    → bf16 (clean gradients, GPU resident)
  - FSDP ref      → bf16, param_offload=true (forward-only; CPU offload frees GPU at init)
  - enforce_eager → disables CUDA graph capture (required for colocated FSDP+vLLM)

Memory budget per GPU (80GB) at vLLM KV profiling:
  FSDP actor sharded:  ~15GB  (116B bf16 ÷ 8 GPUs)
  FSDP optimizer:      ~16GB  (AdamW, 2× sharded weights)
  vLLM KV cache:       ~8GB   (0.1 × 80GB, with max_model_len=4096 not 131072)
  Ref (CPU offloaded): ~0GB   on GPU
  Activations buffer:  ~10GB
  Total:               ~49GB  ✅ ~30GB headroom
"""

import subprocess
import argparse
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ROOT    = os.path.join(PROJECT_ROOT, "data")
CKPT_ROOT    = os.path.join(PROJECT_ROOT, "checkpoints")
REWARD_FN    = os.path.join(PROJECT_ROOT, "training", "grpo", "reward_fn.py")
VERL_CONFIG  = "/home/ssm-user/.local/lib/python3.12/site-packages/verl/trainer/config"

ROUNDS = {
    1: {
        "data":       os.path.join(DATA_ROOT, "grpo_aops_acc0125.parquet"),
        "experiment": "gpt-oss-120b-round1",
        "model":      "unsloth/gpt-oss-120b",
        "ckpt_dir":   os.path.join(CKPT_ROOT, "grpo-round1"),
    },
    2: {
        "data":       os.path.join(DATA_ROOT, "grpo_aops_acc025.parquet"),
        "experiment": "gpt-oss-120b-round2",
        "model":      os.path.join(CKPT_ROOT, "grpo-round1-final"),
        "ckpt_dir":   os.path.join(CKPT_ROOT, "grpo-round2"),
    },
    3: {
        "data":       os.path.join(DATA_ROOT, "grpo_aops_acc0375.parquet"),
        "experiment": "gpt-oss-120b-round3",
        "model":      os.path.join(CKPT_ROOT, "grpo-round2-final"),
        "ckpt_dir":   os.path.join(CKPT_ROOT, "grpo-round3"),
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True, choices=[1, 2, 3])
    args = parser.parse_args()

    r = ROUNDS[args.round]

    if not os.path.exists(r["data"]):
        raise FileNotFoundError(f"Data not found: {r['data']}")
    if args.round > 1 and not os.path.exists(r["model"]):
        raise FileNotFoundError(
            f"Checkpoint not found: {r['model']}\n"
            f"Did round {args.round - 1} finish successfully?"
        )

    print(f"\n{'='*60}")
    print(f"  GRPO Round {args.round}")
    print(f"  Data:    {r['data']}")
    print(f"  Model:   {r['model']}")
    print(f"  Exp:     {r['experiment']}")
    print(f"  Ckpt:    {r['ckpt_dir']}")
    print(f"{'='*60}\n")

    cmd = [
        "python3.12", "-m", "verl.trainer.main_ppo",
        f"--config-path={VERL_CONFIG}",
        "--config-name=ppo_trainer",

        # ── Model ────────────────────────────────────────────────
        f"actor_rollout_ref.model.path={r['model']}",
        "actor_rollout_ref.model.trust_remote_code=true",

        # ── Rollout (vLLM) — bf16, colocated with FSDP ───────────
        # max_model_len=4096: prompt=512 + response=2048 = 2560 needed.
        # Default of 131072 forces vLLM to profile/reserve KV blocks for
        # sequences 64× longer than we use → OOM during _initialize_kv_caches.
        # enforce_eager=true: no CUDA graph capture (segfaults in colocated setup).
        # gpu_memory_utilization=0.1: small reservation at init; expands after FSDP sleeps.
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.n=8",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=8",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.1",
        "actor_rollout_ref.rollout.enforce_eager=true",
        "actor_rollout_ref.rollout.max_model_len=4096",
        "actor_rollout_ref.rollout.max_num_seqs=256",
        "actor_rollout_ref.rollout.temperature=1.0",
        "actor_rollout_ref.rollout.top_p=1.0",
        "actor_rollout_ref.rollout.max_num_batched_tokens=4096",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2",

        # ── Actor (FSDP) — bf16 training, GPU resident ────────────
        # model-only checkpoint saves (~15GB not ~700GB)
        "actor_rollout_ref.actor.ppo_mini_batch_size=32",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2",
        "actor_rollout_ref.actor.use_kl_loss=true",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.use_remove_padding=true",
        "actor_rollout_ref.actor.optim.lr=5e-7",
        "actor_rollout_ref.actor.fsdp_config.dtype=bfloat16",
        "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16",
        "actor_rollout_ref.actor.checkpoint.save_contents=['model']",

        # ── Reference model (FSDP) — bf16, CPU offload ───────────
        # param_offload=true: ref params live on CPU; streamed to GPU
        # one layer at a time during forward pass. No optimizer states.
        "actor_rollout_ref.ref.fsdp_config.param_offload=true",
        "actor_rollout_ref.ref.fsdp_config.dtype=bfloat16",
        "actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2",

        # ── Algorithm ────────────────────────────────────────────
        "algorithm.adv_estimator=grpo",

        # ── Data ─────────────────────────────────────────────────
        f"data.train_files={r['data']}",
        f"data.val_files={r['data']}",
        "data.train_batch_size=32",
        "data.max_prompt_length=512",
        "data.max_response_length=2048",

        # ── Reward ───────────────────────────────────────────────
        "reward.reward_model.enable=false",
        f"reward.custom_reward_function.path={REWARD_FN}",
        "reward.custom_reward_function.name=compute_reward",

        # ── Trainer ──────────────────────────────────────────────
        "trainer.nnodes=1",
        "trainer.n_gpus_per_node=8",
        "trainer.total_epochs=1",
        "trainer.project_name=aimo3-grpo",
        f"trainer.experiment_name={r['experiment']}",
        "trainer.save_freq=50",
        "trainer.val_before_train=false",
        "trainer.test_freq=-1",
        "trainer.resume_mode=auto",
        f"trainer.default_local_dir={r['ckpt_dir']}",
        "trainer.max_actor_ckpt_to_keep=3",
    ]

    # Clean env:
    # - Strip PYTORCH_CUDA_ALLOC_CONF: conflicts with vLLM cumem_allocator
    # - Strip PYTORCH_ALLOC_CONF: same reason
    env = os.environ.copy()
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    env.pop("PYTORCH_ALLOC_CONF", None)

    print(f"Command:\n  {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
