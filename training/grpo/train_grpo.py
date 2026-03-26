"""
AIMO3 GRPO Training Launcher

Usage:
  Round 1: python3.12 training/grpo/train_grpo.py --round 1
  Round 2: python3.12 training/grpo/train_grpo.py --round 2
  Round 3: python3.12 training/grpo/train_grpo.py --round 3

Model:
  unsloth/gpt-oss-120b — 120B MoE (128 experts, top-4), MXFP4 quantized.
  Active params ~30B per forward pass.

Architecture (verl colocated mode on 8×H100 80GB):
  Each GPU runs: FSDP actor shard + FSDP ref shard + vLLM rollout engine.
  They time-share GPU memory via sleep/wake:
    - Training phase: FSDP active (~40GB), vLLM asleep (weights offloaded)
    - Rollout phase:  FSDP asleep (params to CPU), vLLM active (~40GB)

Memory budget per GPU (80GB H100):
  FSDP actor sharded:  ~15GB  (120B params bf16 / 8 GPUs)
  FSDP optimizer:      ~30GB  (AdamW states, CPU offloaded between phases)
  vLLM model weights:  ~18GB  (MXFP4 quantized, loaded during rollout)
  vLLM KV cache:       ~15GB  (gpu_memory_utilization=0.4 of free memory)
  Ref (CPU offloaded): ~0GB   on GPU (forward-only, streamed from CPU)

Key constraint:
  actor.fsdp_config.param_offload=true is REQUIRED so that during vLLM
  initialization, FSDP params are on CPU, freeing GPU for vLLM weights.
  Without this, both FSDP (~37GB) + vLLM weights (~35GB) > 80GB → OOM.
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

        # ── Rollout (vLLM) — colocated with FSDP ─────────────────
        #
        # tensor_model_parallel_size=1:
        #   In verl colocated mode, each Ray actor owns 1 GPU and runs
        #   its own vLLM instance with uniproc executor. TP>1 would
        #   require cross-process coordination that doesn't work here.
        #
        # gpu_memory_utilization=0.4:
        #   Fraction of GPU memory vLLM reserves for KV cache.
        #   Reduced from 0.45 to leave headroom. During rollout,
        #   FSDP actor is asleep (params offloaded to CPU), so
        #   ~60-65GB is available. 0.4 × 80GB ≈ 32GB for KV cache.
        #
        # enforce_eager=true:
        #   Required for colocated FSDP+vLLM (no CUDA graph capture).
        #
        # max_num_seqs=32, max_num_batched_tokens=2048:
        #   Conservative limits to reduce peak vLLM memory during
        #   prefill. The 120B MoE activates 4/128 experts per token,
        #   but intermediate tensors still spike during batched prefill.
        #
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.n=8",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.4",
        "actor_rollout_ref.rollout.enforce_eager=true",
        "actor_rollout_ref.rollout.max_model_len=4096",
        "actor_rollout_ref.rollout.max_num_seqs=32",
        "actor_rollout_ref.rollout.temperature=1.0",
        "actor_rollout_ref.rollout.top_p=1.0",
        "actor_rollout_ref.rollout.max_num_batched_tokens=2048",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",

        # ── Actor (FSDP) — bf16, CPU param offload ───────────────
        #
        # param_offload=true is CRITICAL for the 120B MoE:
        #   Without it, FSDP keeps ~37GB of sharded params on GPU
        #   permanently. When vLLM initializes (even with dummy loading),
        #   it allocates MXFP4 weight buffers (~35GB) on the SAME GPU.
        #   37 + 35 > 80GB → OOM during FusedMoE.create_weights().
        #
        #   With param_offload=true, FSDP streams params from CPU
        #   during forward/backward passes. During vLLM init and
        #   rollout, params live on CPU, leaving GPU free for vLLM.
        #
        #   Trade-off: ~20-30% slower training due to CPU→GPU transfer,
        #   but this is unavoidable for 120B on 8×80GB.
        #
        "actor_rollout_ref.actor.ppo_mini_batch_size=32",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.actor.use_kl_loss=true",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.use_remove_padding=true",
        "actor_rollout_ref.actor.optim.lr=5e-7",
        "actor_rollout_ref.actor.fsdp_config.param_offload=true",
        "actor_rollout_ref.actor.fsdp_config.dtype=bfloat16",
        "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16",
        "actor_rollout_ref.actor.checkpoint.save_contents=['model']",

        # ── Reference model (FSDP) — bf16, CPU offload ───────────
        "actor_rollout_ref.ref.fsdp_config.param_offload=true",
        "actor_rollout_ref.ref.fsdp_config.dtype=bfloat16",
        "actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",

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

    # Clean env to avoid conflicts with vLLM's cumem_allocator
    env = os.environ.copy()
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    env.pop("PYTORCH_ALLOC_CONF", None)

    print(f"Command:\n  {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
