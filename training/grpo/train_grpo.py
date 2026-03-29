"""
AIMO3 GRPO Training — 2-Node Setup
  Node 1 (172.31.104.85): Rollout worker (vLLM, 8x H100)
  Node 2 (172.31.101.45): Actor/Ref trainer (FSDP, 8x H100)

Model: unsloth/gpt-oss-120b (120B MoE, 128 experts top-4, MXFP4)
  - ~115B expert params: FROZEN (MXFP4, no gradients, no optimizer states)
  - ~5B dense params:    TRAINED (attention, norms, router, embed, LM head)

Memory per node:
  Rollout node:  vLLM full model ~65GB, KV cache ~10GB   ✓
  Trainer node:  FSDP shards ~7.5GB, AdamW ~2.5GB        ✓

Usage:
  python3.12 training/grpo/train_grpo.py --round 1
  python3.12 training/grpo/train_grpo.py --round 1 --no-freeze  # full FT
  python3.12 training/grpo/train_grpo.py --revert-patch          # undo patch
"""

import subprocess
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ROOT    = os.path.join(PROJECT_ROOT, "data")
CKPT_ROOT    = os.path.join(PROJECT_ROOT, "checkpoints")
REWARD_FN    = os.path.join(PROJECT_ROOT, "training", "grpo", "reward_fn.py")
VERL_CONFIG  = "/home/ssm-user/.local/lib/python3.12/site-packages/verl/trainer/config"
PATCH_SCRIPT = os.path.join(PROJECT_ROOT, "training", "grpo", "apply_freeze_patch.py")

# ── Node IPs ────────────────────────────────────────────────────────────────
NODE1_IP = os.getenv("NODE1_IP", "172.31.104.85")  # rollout node
NODE2_IP = os.getenv("NODE2_IP", "172.31.101.45")  # trainer node
RAY_PORT = os.getenv("RAY_HEAD_PORT", "6379")

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


def verify_ray_cluster():
    """Verify Ray cluster has 2 nodes and 16 GPUs before starting."""
    import ray
    print(f"Connecting to Ray cluster at {NODE1_IP}:{RAY_PORT}...")
    ray.init(address=f"{NODE1_IP}:{RAY_PORT}", ignore_reinit_error=True)

    nodes = [n for n in ray.nodes() if n["Alive"]]
    total_gpus = sum(n["Resources"].get("GPU", 0) for n in nodes)

    print(f"  Nodes alive: {len(nodes)}")
    print(f"  Total GPUs:  {total_gpus}")
    for n in nodes:
        print(f"    {n['NodeManagerAddress']} — GPUs: {n['Resources'].get('GPU', 0)}")

    if len(nodes) < 2:
        raise RuntimeError(
            f"Expected 2 nodes but found {len(nodes)}.\n"
            f"Run on Node 2: ray start --address={NODE1_IP}:{RAY_PORT} --num-gpus=8 --num-cpus=192"
        )
    if total_gpus < 16:
        raise RuntimeError(f"Expected 16 GPUs but found {total_gpus}.")

    # Identify which node is which
    node_ips = [n["NodeManagerAddress"] for n in nodes]
    if NODE1_IP not in node_ips:
        raise RuntimeError(f"Rollout node {NODE1_IP} not found in cluster: {node_ips}")
    if NODE2_IP not in node_ips:
        raise RuntimeError(f"Trainer node {NODE2_IP} not found in cluster: {node_ips}")

    print(f"\n  Rollout node: {NODE1_IP} (vLLM)")
    print(f"  Trainer node: {NODE2_IP} (FSDP)")
    print("  Ray cluster OK ✓\n")

    ray.shutdown()


def run_patch(revert=False):
    """Apply or revert the verl freeze patch."""
    cmd = ["python3.12", PATCH_SCRIPT]
    if revert:
        cmd.append("--revert")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout.strip())
    if result.returncode != 0:
        print(result.stderr.strip())
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, choices=[1, 2, 3])
    parser.add_argument("--no-freeze", action="store_true",
                        help="Full fine-tune (no expert freezing)")
    parser.add_argument("--revert-patch", action="store_true",
                        help="Revert verl patch and exit")
    parser.add_argument("--skip-ray-check", action="store_true",
                        help="Skip Ray cluster verification")
    args = parser.parse_args()

    if args.revert_patch:
        run_patch(revert=True)
        return

    if args.round is None:
        parser.error("--round is required")

    r = ROUNDS[args.round]

    if not os.path.exists(r["data"]):
        raise FileNotFoundError(f"Data not found: {r['data']}")
    if args.round > 1 and not os.path.exists(r["model"]):
        raise FileNotFoundError(
            f"Checkpoint not found: {r['model']}\n"
            f"Did round {args.round - 1} finish successfully?"
        )

    freeze = not args.no_freeze

    print(f"\n{'='*60}")
    print(f"  GRPO Round {args.round} — 2-Node Setup")
    print(f"  Rollout node: {NODE1_IP} (vLLM, 8x H100)")
    print(f"  Trainer node: {NODE2_IP} (FSDP, 8x H100)")
    print(f"  Data:    {r['data']}")
    print(f"  Model:   {r['model']}")
    print(f"  Exp:     {r['experiment']}")
    print(f"  Ckpt:    {r['ckpt_dir']}")
    print(f"  Mode:    {'Dense-only (MoE experts frozen)' if freeze else 'Full fine-tune'}")
    print(f"{'='*60}\n")

    # Verify Ray cluster
    if not args.skip_ray_check:
        verify_ray_cluster()

    # Apply or ensure patch
    if freeze:
        run_patch()
    print()

    cmd = [
        "python3.12", "-m", "verl.trainer.main_ppo",
        f"--config-path={VERL_CONFIG}",
        "--config-name=ppo_trainer",

        # ── Ray cluster ──────────────────────────────────────────
        f"ray_init.address={NODE1_IP}:{RAY_PORT}",

        # ── Model ────────────────────────────────────────────────
        f"actor_rollout_ref.model.path={r['model']}",
        "actor_rollout_ref.model.trust_remote_code=true",

        # ── Rollout (vLLM) — pinned to Node 1 ───────────────────
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.n=16",                          # was 8, now 16 (full node)
        "actor_rollout_ref.rollout.tensor_model_parallel_size=8",  # full node for vLLM
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.85",   # dedicated node, go high
        "actor_rollout_ref.rollout.enforce_eager=true",
        "actor_rollout_ref.rollout.max_model_len=4096",
        "actor_rollout_ref.rollout.max_num_seqs=32",
        "actor_rollout_ref.rollout.temperature=1.0",
        "actor_rollout_ref.rollout.top_p=1.0",
        "actor_rollout_ref.rollout.max_num_batched_tokens=4096",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",

        # ── Node placement ───────────────────────────────────────
        f"actor_rollout_ref.rollout.ray_actor_options.resources.node:{NODE1_IP}=0.1",
        f"actor_rollout_ref.actor.ray_actor_options.resources.node:{NODE2_IP}=0.1",
        f"actor_rollout_ref.ref.ray_actor_options.resources.node:{NODE2_IP}=0.1",

        # ── Actor (FSDP) — pinned to Node 2 ─────────────────────
        "actor_rollout_ref.actor.ppo_mini_batch_size=64",          # was 32, doubled
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.actor.use_kl_loss=true",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.use_remove_padding=true",
        "actor_rollout_ref.actor.optim.lr=5e-7",
        "actor_rollout_ref.actor.fsdp_config.param_offload=false", # dedicated node, no offload
        "actor_rollout_ref.actor.fsdp_config.use_orig_params=true",
        "actor_rollout_ref.actor.fsdp_config.dtype=bfloat16",
        "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16",
        "actor_rollout_ref.actor.checkpoint.save_contents=['model']",

        # ── Reference (FSDP) — pinned to Node 2 ─────────────────
        "actor_rollout_ref.ref.fsdp_config.param_offload=false",   # dedicated node
        "actor_rollout_ref.ref.fsdp_config.dtype=bfloat16",
        "actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",

        # ── Algorithm ────────────────────────────────────────────
        "algorithm.adv_estimator=grpo",

        # ── Data ─────────────────────────────────────────────────
        f"data.train_files={r['data']}",
        f"data.val_files={r['data']}",
        "data.train_batch_size=64",                                # was 32, doubled
        "data.max_prompt_length=512",
        "data.max_response_length=2048",

        # ── Reward ───────────────────────────────────────────────
        "reward.reward_model.enable=false",
        f"reward.custom_reward_function.path={REWARD_FN}",
        "reward.custom_reward_function.name=compute_score",

        # ── Trainer ──────────────────────────────────────────────
        "trainer.nnodes=2",                                        # was 1
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

    env = os.environ.copy()
    # Critical: remove conflicting env vars
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    env.pop("PYTORCH_ALLOC_CONF", None)
    # EFA / NCCL
    env["FI_PROVIDER"] = "efa"
    env["FI_EFA_USE_DEVICE_RDMA"] = "1"
    env["FI_EFA_FORK_SAFE"] = "1"
    env["RDMAV_FORK_SAFE"] = "1"
    env["NCCL_SOCKET_IFNAME"] = "enp71s0"
    env["NCCL_IB_DISABLE"] = "0"
    env["NCCL_TIMEOUT"] = "1800"
    env["NCCL_PROTO"] = "simple"
    env["NCCL_ALGO"] = "ring"
    env["FI_EFA_SET_CUDA_SYNC_MEMOPS"] = "0"

    print(f"Command:\n  {' '.join(cmd[:6])} ...\n")
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
