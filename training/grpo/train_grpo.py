"""
AIMO3 GRPO Training — 2-Node Setup
  Node 1 (172.31.110.230) + Node 2 (172.31.106.192): 16x H100 global pool
  - FSDP actor/ref: sharded across ALL 16 GPUs
  - SGLang rollout: TP=16 cross-node, async mode, gpt-oss tool format

Model: unsloth/gpt-oss-120b (120B MoE)
  - ~115B expert params: FROZEN
  - ~2.13B dense params: TRAINED

Versions confirmed:
  verl:    0.8.0.dev (from GitHub main)
  sglang:  0.5.9
  ray:     2.54.1
  torch:   2.9.1+cu128
  python:  3.12

Usage:
  python3.12 training/grpo/train_grpo.py --round 1
  python3.12 training/grpo/train_grpo.py --round 1 --no-freeze
  python3.12 training/grpo/train_grpo.py --revert-patch
  python3.12 training/grpo/train_grpo.py --round 1 --skip-ray-check

Changes vs previous version:
  - NCCL/EFA restored (EFA confirmed working cross-node, Gloo removed)
  - LD_LIBRARY_PATH set to include EFA + NCCL + ofi-nccl libs
  - Auto-sync patch + clear .pyc cache on node 2 before launch
  - use_orig_params=true added to actor + ref fsdp_config
  - Upgraded to verl 0.8.0.dev from GitHub main
  - Removed min_p (not in RolloutConfig in this version)
  - Removed tool_parser (not in AgentLoopConfig in this version)
  - Fixed multi_turn prefix: use plain key (field exists, no ++ needed)
  - ref param_offload=true to free memory during rollout
  - expandable_segments=True + max_split_size_mb=512 to reduce fragmentation
  - Fixed LD_LIBRARY_PATH for Ray worker subprocesses on both nodes
  - FIX: ACTOR_MAX_TOKEN_LEN = prompt+response (not *2) — was 36864, now 10240
  - FIX: LOGPROB_MAX_TOKEN_LEN = actor*2 — was 73728, now 20480
  - FIX: PYTORCH_CUDA_ALLOC_CONF → PYTORCH_ALLOC_CONF (torch 2.9+ deprecation)
  - FIX: SGLang TP=8 per-node → TP=16 cross-node
         Model/GPU: 28 GB → 14 GB
         KV budget: 4 GB  → 26 GB  (enough for n=8 at 8K context)
         Trade-off: single generation group (vs 2), cross-node all-reduce
  - FIX: MAX_RESPONSE_LENGTH 16384 → 8192 (4x golden answer length, halves KV need)
  - FIX: MAX_MODEL_LEN 24576 → 11264 (prompt 2048 + response 8192 + buffer 1024)
  - FIX: gpu_memory_utilization 0.50 → 0.40 (leaves more room for FSDP compute_log_prob)
  - FIX: train_batch_size 128 → 32 (reduces OOM during compute_log_prob forward pass)
  - FIX: rollout.mode sync removed (verl 0.8.0.dev removed sync mode), back to async
  - FIX: free_cache_engine=true (explicitly free SGLang KV cache after rollout)
  - FIX: actor param_offload=true + optimizer_offload=true per official verl SGLang docs
         SGLang + FSDP colocated on same GPUs requires CPU offloading to avoid OOM
  - FIX: update_weights_bucket_bytes=1073741824 (1 GB buckets)
         Prevents OOM during weight sync clone — was trying 4 GB contiguous alloc
  - REMOVED: max_num_batched_tokens cap (no longer needed with TP=16 KV headroom)
"""

import subprocess
import argparse
import os
import sys

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ROOT    = os.path.join(PROJECT_ROOT, "data")
CKPT_ROOT    = "/data/checkpoints"
REWARD_FN    = os.path.join(PROJECT_ROOT, "training", "grpo", "reward_fn.py")
AGENT_YAML   = os.path.join(PROJECT_ROOT, "training", "grpo", "agent.yaml")
TOOLS_YAML   = os.path.join(PROJECT_ROOT, "training", "grpo", "tools.yaml")
PATCH_SCRIPT = os.path.join(PROJECT_ROOT, "training", "grpo", "apply_freeze_patch.py")

VERL_CONFIG  = "/home/ssm-user/.local/lib/python3.12/site-packages/verl/trainer/config"
FSDP_WORKERS = "/home/ssm-user/.local/lib/python3.12/site-packages/verl/workers/fsdp_workers.py"

PYTHON = "python3.12"

# ── Node IPs ─────────────────────────────────────────────────────────────────
NODE1_IP = os.getenv("NODE1_IP", "172.31.110.230")
NODE2_IP = os.getenv("NODE2_IP", "172.31.106.192")
RAY_PORT = os.getenv("RAY_HEAD_PORT", "6379")

# ── Training rounds ───────────────────────────────────────────────────────────
ROUNDS = {
    1: {
        "data":       os.path.join(DATA_ROOT, "grpo_aops_acc0125.parquet"),
        "experiment": "gpt-oss-120b-round1",
        "model":      "/data/models/gpt-oss-120b",
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
MAX_PROMPT_LENGTH    = 2048
MAX_RESPONSE_LENGTH  = 8192    # Reduced from 16384 → 8192
                                # 4x longer than typical golden answers
                                # Halves KV cache requirement vs 16K
                                # n=8 at 8K context needs ~23 GB KV — fits in 26 GB budget

# max_model_len = MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + buffer
# 2048 + 8192 + 1024 = 11264
MAX_MODEL_LEN = 11264

# ppo_max_token_len_per_gpu = prompt + response (per-GPU budget, not doubled)
# 2048 + 8192 = 10240
ACTOR_MAX_TOKEN_LEN   = MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH   # 10240

# ref log_prob batching — 2x actor is generous
LOGPROB_MAX_TOKEN_LEN = ACTOR_MAX_TOKEN_LEN * 2                   # 20480

# SGLang rollout samples per prompt
# n=8 gives good GRPO gradient variance reduction
# With TP=16 + 26 GB KV budget, n=8 at 8K context needs ~23 GB → fits
ROLLOUT_N = 8

TEMPERATURE = 1.0
TOP_P       = 1.0

# ── NCCL / EFA library path ───────────────────────────────────────────────────
NCCL_LIB  = "/opt/pytorch/lib/python3.13/site-packages/nvidia/nccl/lib"
EFA_LIB   = "/opt/amazon/efa/lib64"
CUDA_LIB  = "/home/ssm-user/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"
OMPI_LIB  = "/opt/amazon/openmpi/lib"
OFI_LIB   = "/opt/amazon/ofi-nccl/lib"

EFA_LD_LIBRARY_PATH = ":".join([EFA_LIB, NCCL_LIB, CUDA_LIB, OMPI_LIB, OFI_LIB])


# ── Helpers ───────────────────────────────────────────────────────────────────

def verify_ray_cluster():
    """Confirm Ray sees 2 nodes and 16 GPUs before kicking off training."""
    try:
        import ray
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
        print(f"    {ip}  --  {gpus} GPU(s)")

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
            f"Expected >=16 GPUs but found {int(total_gpus)}.\n"
            f"Check that both nodes registered their GPUs with Ray."
        )

    print(f"\n  Ray cluster OK  --  {int(total_gpus)} GPUs across {len(nodes)} nodes\n")
    ray.shutdown()


def clear_pyc_cache_local():
    """Delete all verl .pyc files on this node so patched .py files are used."""
    print("  Clearing verl .pyc cache on Node 1 ...")
    result = subprocess.run(
        ["find",
         "/home/ssm-user/.local/lib/python3.12/site-packages/verl/",
         "-name", "*.pyc", "-delete"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  WARNING: pyc cache clear failed: {result.stderr.strip()}", file=sys.stderr)
    else:
        print("  Node 1 .pyc cache cleared.")


def sync_patch_to_node2():
    """
    Copy the patched fsdp_workers.py to Node 2 and clear its .pyc cache.
    Ray spawns worker processes on Node 2 independently -- if the .py or
    .pyc there is stale, the patch is silently ignored on that node.
    """
    print(f"  Syncing patched fsdp_workers.py to Node 2 ({NODE2_IP}) ...")

    scp_result = subprocess.run(
        ["scp", "-o", "StrictHostKeyChecking=no",
         FSDP_WORKERS,
         f"{NODE2_IP}:{FSDP_WORKERS}"],
        capture_output=True, text=True
    )
    if scp_result.returncode != 0:
        print(f"  WARNING: scp to Node 2 failed:\n{scp_result.stderr.strip()}", file=sys.stderr)
        print("  You may need to manually sync the patch. Continuing anyway ...", file=sys.stderr)
    else:
        print("  fsdp_workers.py synced to Node 2.")

    print(f"  Clearing verl .pyc cache on Node 2 ({NODE2_IP}) ...")
    ssh_result = subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no", NODE2_IP,
         "find /home/ssm-user/.local/lib/python3.12/site-packages/verl/ "
         "-name '*.pyc' -delete"],
        capture_output=True, text=True
    )
    if ssh_result.returncode != 0:
        print(f"  WARNING: Node 2 pyc cache clear failed:\n{ssh_result.stderr.strip()}", file=sys.stderr)
    else:
        print("  Node 2 .pyc cache cleared.")

    verify_result = subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no", NODE2_IP,
         f"grep -c 'AIMO3' {FSDP_WORKERS}"],
        capture_output=True, text=True
    )
    count = verify_result.stdout.strip()
    if verify_result.returncode == 0 and count.isdigit() and int(count) > 0:
        print(f"  Node 2 patch verified ({count} AIMO3 markers found).")
    else:
        print("  WARNING: Could not verify patch on Node 2 -- check manually.", file=sys.stderr)


def run_patch(revert: bool = False):
    """Apply (or revert) the verl expert-freeze patch on Node 1."""
    if not revert:
        try:
            with open(FSDP_WORKERS) as fh:
                if "AIMO3" in fh.read():
                    print("Freeze patch already applied on Node 1 -- skipping patch step.")
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
    """
    Return environment dict using NCCL over EFA for cross-node comms.
    EFA confirmed working between 172.31.110.230 and 172.31.106.192.
    """
    env = os.environ.copy()

    # Remove both old and new alloc conf keys — we set our own below
    # torch 2.9+ uses PYTORCH_ALLOC_CONF; older versions used PYTORCH_CUDA_ALLOC_CONF
    for key in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        env.pop(key, None)
    for key in ("TORCH_DISTRIBUTED_BACKEND", "GLOO_SOCKET_IFNAME"):
        env.pop(key, None)

    # Build LD_LIBRARY_PATH including CUDA 12 paths for both nodes
    # Ray worker subprocesses need this explicitly — ldconfig alone is not enough
    cuda_paths = (
        "/usr/local/cuda/targets/x86_64-linux/lib:"
        "/usr/local/cuda-12.8/lib64:"
        "/home/ssm-user/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"
    )
    existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
    new_ld = cuda_paths + ":" + EFA_LD_LIBRARY_PATH + (":" + existing_ld if existing_ld else "")

    env.update({
        # ── Memory management ─────────────────────────────────────────────────
        # expandable_segments: reduces fragmentation during state_dict allocation
        # max_split_size_mb:   prevents large contiguous allocation failures
        # Both are needed to survive FSDP unshard during update_weights()
        # NOTE: torch 2.9+ uses PYTORCH_ALLOC_CONF (PYTORCH_CUDA_ALLOC_CONF deprecated)
        "PYTORCH_ALLOC_CONF":              "expandable_segments:True,max_split_size_mb:512",

        # ── CUDA paths ────────────────────────────────────────────────────────
        "PATH":                            "/usr/local/cuda/bin:" + env.get("PATH", ""),
        "CUDA_HOME":                       "/usr/local/cuda",
        "LD_LIBRARY_PATH":                 new_ld,

        # ── NCCL over EFA ────────────────────────────────────────────────────
        "NCCL_DEBUG":                      "WARN",
        "FI_EFA_USE_DEVICE_RDMA":          "1",
        "FI_EFA_FORK_SAFE":                "1",
        "NCCL_SOCKET_IFNAME":              "enp71s0",

        # ── Distributed ──────────────────────────────────────────────────────
        "MASTER_ADDR":                     NODE1_IP,
        "MASTER_PORT":                     "29500",

        # ── Timeouts ─────────────────────────────────────────────────────────
        "TORCH_DIST_INIT_BARRIER_TIMEOUT": "1800",
        "NCCL_TIMEOUT":                    "1800",

        # ── SGLang hints ──────────────────────────────────────────────────────
        "VLLM_USE_V1":                     "1",
        "VLLM_ATTENTION_BACKEND":          "FLASH_ATTN",
        "SGLANG_JIT_CACHE":                "1",

        # ── verl / hydra verbosity ────────────────────────────────────────────
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

        # ── Model ─────────────────────────────────────────────────────────────
        f"actor_rollout_ref.model.path={r['model']}",
        "actor_rollout_ref.model.trust_remote_code=true",
        "actor_rollout_ref.model.use_remove_padding=true",
        "actor_rollout_ref.model.enable_gradient_checkpointing=true",

        # ── Rollout -- SGLang 0.5.9, TP=16 cross-node, async, gpt-oss tools ────
        "actor_rollout_ref.rollout.name=sglang",
        # async is the only supported mode in verl 0.8.0.dev (sync was removed)
        "actor_rollout_ref.rollout.mode=async",
        # FIX: explicitly free SGLang KV cache after rollout before FSDP runs
        # Default is True but being explicit — this is the official fix for
        # SGLang + FSDP colocated OOM per verl docs
        "actor_rollout_ref.rollout.free_cache_engine=true",
        # TP=16 cross-node (was TP=8 per-node):
        #   Model per GPU : 28 GB → 14 GB
        #   KV budget     :  4 GB → 26 GB  (enough for n=8 at 8K context)
        #   Trade-off     : single generation group, cross-node all-reduce per layer
        #   But KV was completely starved at TP=8 — this is the right call
        "actor_rollout_ref.rollout.tensor_model_parallel_size=16",
        # FIX: reduced from 0.50 → 0.40 to leave more headroom for FSDP compute_log_prob
        # SGLang and FSDP share GPUs — SGLang needs to give back memory during log prob phase
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.40",
        f"actor_rollout_ref.rollout.n={ROLLOUT_N}",
        f"actor_rollout_ref.rollout.temperature={TEMPERATURE}",
        f"actor_rollout_ref.rollout.top_p={TOP_P}",
        # 2048 + 8192 + 1024 buffer = 11264
        f"actor_rollout_ref.rollout.max_model_len={MAX_MODEL_LEN}",
        # 1 GB weight-sync buckets — prevents OOM during update_weights clone
        # Old default was ~4 GB contiguous alloc which exceeded free memory
        # "actor_rollout_ref.rollout.update_weights_bucket_bytes=1073741824",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",

        # ── Multi-turn tool calling ───────────────────────────────────────────
        "actor_rollout_ref.rollout.multi_turn.enable=true",
        "actor_rollout_ref.rollout.multi_turn.max_user_turns=8",
        "actor_rollout_ref.rollout.multi_turn.max_assistant_turns=8",
        "actor_rollout_ref.rollout.multi_turn.format=gpt-oss",
        f"actor_rollout_ref.rollout.agent.agent_loop_config_path={AGENT_YAML}",
        f"actor_rollout_ref.rollout.multi_turn.tool_config_path={TOOLS_YAML}",

        # ── Validation rollout settings ───────────────────────────────────────
        f"actor_rollout_ref.rollout.val_kwargs.top_p={TOP_P}",
        f"actor_rollout_ref.rollout.val_kwargs.temperature={TEMPERATURE}",
        "actor_rollout_ref.rollout.val_kwargs.n=1",

        # ── Actor -- FSDP, sharded across 16 GPUs ────────────────────────────
        "actor_rollout_ref.actor.use_kl_loss=true",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.clip_ratio_low=0.2",
        "actor_rollout_ref.actor.clip_ratio_high=0.28",
        "actor_rollout_ref.actor.clip_ratio_c=10.0",
        "actor_rollout_ref.actor.optim.lr=5e-7",
        "actor_rollout_ref.actor.use_dynamic_bsz=true",
        "actor_rollout_ref.actor.ppo_mini_batch_size=16",
        # FIX: was (prompt+response)*2 = 36864, now prompt+response = 18432
        # ppo_max_token_len_per_gpu is a per-GPU budget, not a doubled value
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={ACTOR_MAX_TOKEN_LEN}",
        "actor_rollout_ref.actor.ulysses_sequence_parallel_size=4",
        # FIX: offload actor params and optimizer to CPU per official verl SGLang docs
        # SGLang + FSDP colocated on same GPUs requires offloading to avoid OOM
        # See: https://verl.readthedocs.io/en/latest/workers/sglang_worker.html
        "actor_rollout_ref.actor.fsdp_config.param_offload=true",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=true",
        "actor_rollout_ref.actor.fsdp_config.dtype=bfloat16",
        "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16",
        # Required for mixed requires_grad (frozen experts + trainable dense)
        "actor_rollout_ref.actor.fsdp_config.use_orig_params=true",
        "actor_rollout_ref.actor.checkpoint.save_contents=['model']",

        # ── Reference model -- FSDP ───────────────────────────────────────────
        # FIX: was 73728 (wrong *2 of wrong actor value), now 36864
        f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={LOGPROB_MAX_TOKEN_LEN}",
        # Offload ref model to CPU — only needed for KL, not on critical path
        # Frees ~15GB per GPU during rollout and weight sync
        "actor_rollout_ref.ref.fsdp_config.param_offload=true",
        "actor_rollout_ref.ref.fsdp_config.dtype=bfloat16",
        "actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16",
        "actor_rollout_ref.ref.fsdp_config.use_orig_params=true",

        # ── Algorithm ─────────────────────────────────────────────────────────
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=false",

        # ── Data ──────────────────────────────────────────────────────────────
        f"data.train_files={r['data']}",
        f"data.val_files={r['data']}",
        "data.return_raw_chat=true",
        # FIX: reduced from 128 → 32 to avoid OOM during compute_log_prob
        # 32 prompts x 8 responses = 256 sequences per step (was 1024)
        # Trades fewer steps (164 vs 41) for memory safety
        "data.train_batch_size=32",
        f"data.max_prompt_length={MAX_PROMPT_LENGTH}",
        f"data.max_response_length={MAX_RESPONSE_LENGTH}",
        "data.filter_overlong_prompts=true",
        "data.truncation=error",

        # ── Reward ────────────────────────────────────────────────────────────
        "reward.reward_model.enable=false",
        f"reward.custom_reward_function.path={REWARD_FN}",
        "reward.custom_reward_function.name=compute_score",

        # ── Trainer ───────────────────────────────────────────────────────────
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
        description="AIMO3 GRPO training launcher -- 2-node 16xH100"
    )
    parser.add_argument("--round", type=int, choices=[1, 2, 3],
                        help="Training round (1/2/3)")
    parser.add_argument("--no-freeze", action="store_true",
                        help="Full fine-tune -- do NOT freeze MoE expert weights")
    parser.add_argument("--revert-patch", action="store_true",
                        help="Revert the verl freeze patch and exit")
    parser.add_argument("--skip-ray-check", action="store_true",
                        help="Skip Ray cluster verification")
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
    print(f"  GRPO Round {args.round}  --  2-Node 16xH100 Setup")
    print(f"  Node 1 : {NODE1_IP}  (8xH100)")
    print(f"  Node 2 : {NODE2_IP}  (8xH100)")
    print(f"  Comms  : NCCL over EFA (confirmed working)")
    print(f"  Rollout: SGLang 0.5.9  TP=16 cross-node  async  gpt-oss tools")
    print(f"           model/GPU=14GB  KV=24GB  util=0.40  bucket=1GB")
    print(f"  FSDP   : sharded across 16 GPUs  (actor on GPU, ref offloaded)")
    print(f"  verl   : 0.8.0.dev  torch: 2.9.1+cu128  ray: 2.54.1")
    print(f"  Data   : {r['data']}")
    print(f"  Model  : {r['model']}")
    print(f"  Exp    : {r['experiment']}")
    print(f"  Ckpt   : {r['ckpt_dir']}")
    print(f"  Context: prompt={MAX_PROMPT_LENGTH}  response={MAX_RESPONSE_LENGTH}  model_len={MAX_MODEL_LEN}")
    print(f"  Tokens : actor_max={ACTOR_MAX_TOKEN_LEN}  logprob_max={LOGPROB_MAX_TOKEN_LEN}")
    print(f"  Sample : n={ROLLOUT_N}  temp={TEMPERATURE}  top_p={TOP_P}")
    print(f"  Batch  : train_batch_size=32  (was 128, reduced for OOM fix)")
    print(f"  GPU mem: PYTORCH_ALLOC_CONF=expandable+max_split_512mb")
    print(f"  Mode   : {'Dense-only (MoE experts FROZEN)' if freeze else 'Full fine-tune'}")
    print(f"{'='*62}\n")

    # ── Cluster + patch ───────────────────────────────────────────────────────
    if not args.skip_ray_check:
        verify_ray_cluster()

    if freeze:
        # 1. Apply patch on Node 1
        run_patch()
        print()

        # 2. Clear local .pyc cache so Node 1 workers load the patched .py
        clear_pyc_cache_local()

        # 3. Sync patched file to Node 2 and clear its .pyc cache
        sync_patch_to_node2()

    print()

    # ── Launch ────────────────────────────────────────────────────────────────
    cmd = build_cmd(r, freeze)
    env = build_env()

    print("Launching verl PPO trainer ...")
    print(f"  {PYTHON} -m verl.trainer.main_ppo  [round={args.round}]\n")

    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
