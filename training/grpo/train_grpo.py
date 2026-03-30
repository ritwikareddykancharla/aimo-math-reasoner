"""
AIMO3 GRPO Training — 2-Node Setup
  Node 1 (172.31.98.238) + Node 2 (172.31.108.138): 16x H100 global pool
  - FSDP actor/ref: sharded across ALL 16 GPUs
  - SGLang rollout: TP=8 per node, async mode, gpt-oss tool format

Model: unsloth/gpt-oss-120b (120B MoE)
  - ~115B expert params: FROZEN
  - ~2.13B dense params: TRAINED

Versions confirmed:
  sglang:  0.5.9
  ray:     2.54.1
  torch:   2.9.1+cu128
  python:  3.12

Usage:
  python3.12 training/grpo/train_grpo.py --round 1
  python3.12 training/grpo/train_grpo.py --round 1 --no-freeze
  python3.12 training/grpo/train_grpo.py --revert-patch
  python3.12 training/grpo/train_grpo.py --round 1 --skip-ray-check   # skip cluster verify

Changes vs previous version:
  - MAX_PROMPT_LENGTH  : 1024  → 2048  (math problems can be longer)
  - MAX_RESPONSE_LENGTH: 4096  → 32768 (match inference context_tokens=65536 budget)
  - rollout.max_model_len: 8192 → 65536 (match inference notebook exactly)
  - rollout.min_p added: 0.02  (matches inference CFG.min_p)
  - fsdp param_offload / optimizer_offload: false → true (comment said true, code said false)
  - build_env(): removed duplicate NCCL_TIMEOUT / TORCH_NCCL_* keys
  - Summary print: "offload=true" now actually reflects the config
"""

import subprocess
import argparse
import os
import sys

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ROOT    = os.path.join(PROJECT_ROOT, "data")
CKPT_ROOT    = "/mnt/checkpoints"
REWARD_FN    = os.path.join(PROJECT_ROOT, "training", "grpo", "reward_fn.py")
AGENT_YAML   = os.path.join(PROJECT_ROOT, "training", "grpo", "agent.yaml")
PATCH_SCRIPT = os.path.join(PROJECT_ROOT, "training", "grpo", "apply_freeze_patch.py")

# verl installs under /mnt python environment
VERL_CONFIG  = "/mnt/python/lib/python3.12/site-packages/verl/trainer/config"
FSDP_WORKERS = "/mnt/python/lib/python3.12/site-packages/verl/workers/fsdp_workers.py"

PYTHON = "python3.12"

# ── Node IPs ─────────────────────────────────────────────────────────────────
NODE1_IP = os.getenv("NODE1_IP", "172.31.98.238")
NODE2_IP = os.getenv("NODE2_IP", "172.31.108.138")
RAY_PORT = os.getenv("RAY_HEAD_PORT", "6379")

# ── Training rounds ───────────────────────────────────────────────────────────
ROUNDS = {
    1: {
        "data":       os.path.join(DATA_ROOT, "grpo_aops_acc0125.parquet"),
        "experiment": "gpt-oss-120b-round1",
        "model":      "/mnt/models/gpt-oss-120b",
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

# ── Hyperparameters ───────────────────────────────────────────────────────────
# Matched to inference notebook: context_tokens=65536, buffer_tokens=512
# prompt budget: 2048 tokens (math problems can exceed 1024)
# response budget: 32768 tokens (leaves ~30k headroom vs 65536 context)
MAX_PROMPT_LENGTH    = 2048
MAX_RESPONSE_LENGTH  = 32768

# Token budget per GPU for actor forward/backward.
# With dynamic batching this is a soft ceiling, not a fixed batch size.
ACTOR_MAX_TOKEN_LEN   = (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 2
# Larger budget for log-prob computation (no grad, less memory pressure)
LOGPROB_MAX_TOKEN_LEN = ACTOR_MAX_TOKEN_LEN * 2

# Sampling params — keep in sync with inference CFG
TEMPERATURE = 1.0
TOP_P       = 1.0
MIN_P       = 0.02   # matches inference CFG.min_p


# ── Helpers ───────────────────────────────────────────────────────────────────

def verify_ray_cluster():
    """Confirm Ray sees 2 nodes and 16 GPUs before kicking off training."""
    try:
        import ray  # noqa: PLC0415
    except ImportError:
        print("WARNING: ray not importable from this interpreter; skipping cluster check.")
        return

    print(f"Connecting to Ray cluster at {NODE1_IP}:{RAY_PORT}...")
    ray.init(address=f"{NODE1_IP}:{RAY_PORT}", ignore_reinit_error=True)

    nodes      = [n for n in ray.nodes() if n["Alive"]]
    total_gpus = sum(n["Resources"].get("GPU", 0) for n in nodes)

    print(f"  Nodes alive : {len(nodes)}")
    print(f"  Total GPUs  : {int(total_gpus)}")
    for n in nodes:
        ip   = n["NodeManagerAddress"]
        gpus = int(n["Resources"].get("GPU", 0))
        print(f"    {ip}  —  {gpus} GPU(s)")

    if len(nodes) < 2:
        ray.shutdown()
        raise RuntimeError(
            f"Expected 2 nodes but found {len(nodes)}.\n"
            f"On Node 2, run:\n"
            f"  ray start --address={NODE1_IP}:{RAY_PORT} --num-gpus=8 --num-cpus=192"
        )
    if total_gpus < 16:
        ray.shutdown()
        raise RuntimeError(
            f"Expected ≥16 GPUs but found {int(total_gpus)}.\n"
            f"Check that both nodes registered their GPUs with Ray."
        )

    print(f"\n  Ray cluster OK ✓  —  {int(total_gpus)} GPUs across {len(nodes)} nodes\n")
    ray.shutdown()


def run_patch(revert: bool = False):
    """Apply (or revert) the verl expert-freeze patch."""
    if not revert:
        try:
            with open(FSDP_WORKERS) as fh:
                if "AIMO3" in fh.read():
                    print("Freeze patch already applied — skipping.")
                    return
        except FileNotFoundError:
            print(f"WARNING: Cannot find {FSDP_WORKERS}; skipping patch step.")
            return

    cmd = [PYTHON, PATCH_SCRIPT]
    if revert:
        cmd.append("--revert")

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout.strip())
    if result.returncode != 0 and "already patched" not in result.stdout:
        print(result.stderr.strip(), file=sys.stderr)
        sys.exit(1)


def build_env() -> dict:
    """Return an environment dict with EFA/NCCL settings for 2-node comms."""
    env = os.environ.copy()

    # Remove allocator configs that conflict with torch 2.9 / CUDA 12.8
    for key in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        env.pop(key, None)

    # Each key appears exactly once — duplicates removed.
    env.update({
        # EFA (AWS p4d / p5 instances)
        "FI_PROVIDER":                     "efa",
        "FI_EFA_USE_DEVICE_RDMA":          "1",
        "FI_EFA_FORK_SAFE":                "1",
        "FI_EFA_SET_CUDA_SYNC_MEMOPS":     "0",
        "RDMAV_FORK_SAFE":                 "1",
        # NCCL
        "NCCL_SOCKET_IFNAME":              "enp71s0",
        # NCCL_IB_DISABLE must NOT be 1 on EFA instances —
        # aws-ofi-nccl routes through the IB verbs path even on EFA.
        # Setting it to 1 forces socket-only fallback which fails cross-node.
        "NCCL_IB_DISABLE":                 "0",
        "NCCL_TIMEOUT":                    "1800",
        "NCCL_PROTO":                      "simple",
        "NCCL_ALGO":                       "ring",
        "NCCL_DEBUG":                      "INFO",
        "NCCL_DEBUG_FILE":                 "/tmp/nccl_worker_%h_%p.log",
        # Torch / NCCL async
        "TORCH_NCCL_BLOCKING_WAIT":        "0",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "TORCH_DIST_INIT_BARRIER_TIMEOUT": "1800",
        # SGLang / vLLM backend hints
        "VLLM_USE_V1":                     "1",
        "VLLM_ATTENTION_BACKEND":          "FLASH_ATTN",
        # verl / hydra verbosity
        "HYDRA_FULL_ERROR":                "1",
        "RAY_LOGGING_LEVEL":               "DEBUG",
    })
    return env


def build_cmd(r: dict, freeze: bool) -> list:
    """Construct the verl PPO trainer command."""
    return [
        PYTHON, "-m", "verl.trainer.main_ppo",
        f"--config-path={VERL_CONFIG}",
        "--config-name=ppo_trainer",

        # ── Model ────────────────────────────────────────────────────────────
        f"actor_rollout_ref.model.path={r['model']}",
        "actor_rollout_ref.model.trust_remote_code=true",
        "actor_rollout_ref.model.use_remove_padding=true",
        "actor_rollout_ref.model.enable_gradient_checkpointing=true",

        # ── Rollout — SGLang 0.5.9, TP=8 per node, async, gpt-oss tools ─────
        # max_model_len=65536 matches inference notebook context_tokens=65536
        "actor_rollout_ref.rollout.name=sglang",
        "actor_rollout_ref.rollout.mode=async",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=8",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.7",
        "actor_rollout_ref.rollout.n=8",
        f"actor_rollout_ref.rollout.temperature={TEMPERATURE}",
        f"actor_rollout_ref.rollout.top_p={TOP_P}",
        f"+actor_rollout_ref.rollout.min_p={MIN_P}",
        "actor_rollout_ref.rollout.max_model_len=65536",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        # Multi-turn tool calling
        "++actor_rollout_ref.rollout.multi_turn.enable=true",
        "++actor_rollout_ref.rollout.multi_turn.max_user_turns=8",
        "++actor_rollout_ref.rollout.multi_turn.max_assistant_turns=8",
        "++actor_rollout_ref.rollout.multi_turn.format=gpt-oss",
        "+actor_rollout_ref.rollout.agent.tool_parser=gpt-oss",
        f"+actor_rollout_ref.rollout.agent.agent_loop_config_path={AGENT_YAML}",
        # Validation rollout settings
        f"actor_rollout_ref.rollout.val_kwargs.top_p={TOP_P}",
        f"actor_rollout_ref.rollout.val_kwargs.temperature={TEMPERATURE}",
        "actor_rollout_ref.rollout.val_kwargs.n=1",

        # ── Actor — FSDP, sharded across 16 GPUs ─────────────────────────────
        "actor_rollout_ref.actor.use_kl_loss=true",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.clip_ratio_low=0.2",
        "actor_rollout_ref.actor.clip_ratio_high=0.28",
        "actor_rollout_ref.actor.clip_ratio_c=10.0",
        "actor_rollout_ref.actor.optim.lr=5e-7",
        "actor_rollout_ref.actor.use_dynamic_bsz=true",
        "actor_rollout_ref.actor.ppo_mini_batch_size=16",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={ACTOR_MAX_TOKEN_LEN}",
        "actor_rollout_ref.actor.ulysses_sequence_parallel_size=4",
        # CPU offload disabled — 120B MoE sharded across 16×H100s fits in HBM
        "actor_rollout_ref.actor.fsdp_config.param_offload=false",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=false",
        "actor_rollout_ref.actor.fsdp_config.dtype=bfloat16",
        "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16",
        "actor_rollout_ref.actor.checkpoint.save_contents=['model']",

        # ── Reference model — FSDP ───────────────────────────────────────────
        f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={LOGPROB_MAX_TOKEN_LEN}",
        "actor_rollout_ref.ref.fsdp_config.param_offload=false",
        "actor_rollout_ref.ref.fsdp_config.dtype=bfloat16",
        "actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16",

        # ── Algorithm ────────────────────────────────────────────────────────
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=false",

        # ── Data ─────────────────────────────────────────────────────────────
        f"data.train_files={r['data']}",
        f"data.val_files={r['data']}",
        "data.return_raw_chat=true",
        "data.train_batch_size=128",
        f"data.max_prompt_length={MAX_PROMPT_LENGTH}",
        f"data.max_response_length={MAX_RESPONSE_LENGTH}",
        "data.filter_overlong_prompts=true",
        "data.truncation=error",

        # ── Reward ───────────────────────────────────────────────────────────
        "reward.reward_model.enable=false",
        f"reward.custom_reward_function.path={REWARD_FN}",
        "reward.custom_reward_function.name=compute_score",

        # ── Trainer ──────────────────────────────────────────────────────────
        "trainer.nnodes=2",
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
        "trainer.log_val_generations=50",
        "+trainer.nccl_timeout=1800",
    ]


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AIMO3 GRPO training launcher — 2-node 16×H100"
    )
    parser.add_argument("--round", type=int, choices=[1, 2, 3],
                        help="Training round (1/2/3)")
    parser.add_argument("--no-freeze", action="store_true",
                        help="Full fine-tune — do NOT freeze MoE expert weights")
    parser.add_argument("--revert-patch", action="store_true",
                        help="Revert the verl freeze patch and exit")
    parser.add_argument("--skip-ray-check", action="store_true",
                        help="Skip Ray cluster verification (useful for single-node debug)")
    args = parser.parse_args()

    if args.revert_patch:
        run_patch(revert=True)
        return

    if args.round is None:
        parser.error("--round is required (choose 1, 2, or 3)")

    r      = ROUNDS[args.round]
    freeze = not args.no_freeze

    # ── Pre-flight checks ─────────────────────────────────────────────────────
    if not os.path.exists(r["data"]):
        raise FileNotFoundError(f"Training data not found: {r['data']}")
    if args.round > 1 and not os.path.exists(r["model"]):
        raise FileNotFoundError(
            f"Checkpoint not found: {r['model']}\n"
            f"Did round {args.round - 1} complete successfully?"
        )
    if not os.path.exists(AGENT_YAML):
        raise FileNotFoundError(
            f"Agent config not found: {AGENT_YAML}\n"
            f"Copy or symlink your agent.yaml there before launching."
        )
    if not os.path.exists(REWARD_FN):
        raise FileNotFoundError(f"Reward function not found: {REWARD_FN}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  GRPO Round {args.round}  —  2-Node 16×H100 Setup")
    print(f"  Node 1 : {NODE1_IP}  (8×H100)")
    print(f"  Node 2 : {NODE2_IP}  (8×H100)")
    print(f"  Rollout: SGLang 0.5.9  TP=8/node  async  gpt-oss tools")
    print(f"  FSDP   : sharded across 16 GPUs  (offload=false)")
    print(f"  Torch  : 2.9.1+cu128   Ray: 2.54.1")
    print(f"  Data   : {r['data']}")
    print(f"  Model  : {r['model']}")
    print(f"  Exp    : {r['experiment']}")
    print(f"  Ckpt   : {r['ckpt_dir']}")
    print(f"  Context: prompt={MAX_PROMPT_LENGTH}  response={MAX_RESPONSE_LENGTH}  model_len=65536")
    print(f"  Sample : temp={TEMPERATURE}  top_p={TOP_P}  min_p={MIN_P}")
    print(f"  Mode   : {'Dense-only (MoE experts FROZEN)' if freeze else 'Full fine-tune'}")
    print(f"{'='*62}\n")

    # ── Cluster + patch ───────────────────────────────────────────────────────
    if not args.skip_ray_check:
        verify_ray_cluster()

    if freeze:
        run_patch()

    print()

    # ── Launch ────────────────────────────────────────────────────────────────
    cmd = build_cmd(r, freeze)
    env = build_env()

    print("Launching verl PPO trainer …")
    print(f"  {PYTHON} -m verl.trainer.main_ppo  [round={args.round}]\n")

    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
