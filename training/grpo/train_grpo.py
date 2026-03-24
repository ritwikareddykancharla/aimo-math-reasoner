"""
AIMO3 GRPO Training Launcher

Usage:
  Round 1: python3.12 training/grpo/train_grpo.py --round 1
  Round 2: python3.12 training/grpo/train_grpo.py --round 2
  Round 3: python3.12 training/grpo/train_grpo.py --round 3
"""

import subprocess
import argparse
import os

# Absolute paths — Hydra requires this
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH  = os.path.join(PROJECT_ROOT, "training", "grpo")
DATA_ROOT    = os.path.join(PROJECT_ROOT, "data")
CKPT_ROOT    = os.path.join(PROJECT_ROOT, "checkpoints")
# In train_grpo.py change cmd to:

cmd = [
    "python3.12", "-m", "verl.trainer.main_ppo",
    # Use VERL's built-in config as base, then override with ours
    "--config-path=/home/ssm-user/.local/lib/python3.12/site-packages/verl/trainer/config",
    "--config-name=ppo_trainer",
    # Override with your values
    f"actor_rollout_ref.model.path={r['model']}",
    f"actor_rollout_ref.model.trust_remote_code=true",
    f"actor_rollout_ref.rollout.name=vllm",
    f"actor_rollout_ref.rollout.n=8",
    f"actor_rollout_ref.rollout.tensor_parallel_size=8",
    f"actor_rollout_ref.rollout.gpu_memory_utilization=0.4",
    f"actor_rollout_ref.rollout.temperature=1.0",
    f"actor_rollout_ref.actor.use_kl_loss=true",
    f"actor_rollout_ref.actor.kl_loss_coef=0.001",
    f"actor_rollout_ref.actor.kl_loss_type=low_var_kl",
    f"actor_rollout_ref.actor.use_remove_padding=true",
    f"actor_rollout_ref.actor.optim.lr=5e-7",
    f"algorithm.adv_estimator=grpo",
    f"data.train_files={r['data']}",
    f"data.val_files={r['data']}",
    f"data.train_batch_size=32",
    f"data.max_prompt_length=512",
    f"data.max_response_length=2048",
    f"reward_model.enable=false",
    f"custom_reward_function.path=/home/ssm-user/aimo-math-reasoner/training/grpo/reward_fn.py",
    f"custom_reward_function.name=compute_reward",
    f"trainer.n_gpus_per_node=8",
    f"trainer.total_epochs=1",
    f"trainer.project_name=aimo3-grpo",
    f"trainer.experiment_name={r['experiment']}",
    f"trainer.save_freq=50",
    f"trainer.val_before_train=false",
    f"trainer.test_freq=-1",
    f"trainer.default_local_dir=/home/ssm-user/aimo-math-reasoner/checkpoints/{r['experiment']}",
    f"trainer.max_actor_ckpt_to_keep=3",
]

ROUNDS = {
    1: {
        "config":     "grpo_round1",
        "data":       os.path.join(DATA_ROOT, "grpo_aops_acc0125.parquet"),
        "experiment": "gpt-oss-120b-round1",
        "model":      "openai/gpt-oss-120b",
    },
    2: {
        "config":     "grpo_round2",
        "data":       os.path.join(DATA_ROOT, "grpo_aops_acc025.parquet"),
        "experiment": "gpt-oss-120b-round2",
        "model":      os.path.join(CKPT_ROOT, "grpo-round1-final"),
    },
    3: {
        "config":     "grpo_round3",
        "data":       os.path.join(DATA_ROOT, "grpo_aops_acc0375.parquet"),
        "experiment": "gpt-oss-120b-round3",
        "model":      os.path.join(CKPT_ROOT, "grpo-round2-final"),
    },
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True, choices=[1, 2, 3])
    args = parser.parse_args()

    r = ROUNDS[args.round]

    # Validate inputs exist before launching
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
    print(f"  Config:  {CONFIG_PATH}/{r['config']}.yaml")
    print(f"  Exp:     {r['experiment']}")
    print(f"{'='*60}\n")

    cmd = [
        "python3.12", "-m", "verl.trainer.main_ppo",
        f"--config-path={CONFIG_PATH}",       # absolute path fixes Hydra
        f"--config-name={r['config']}",
        f"actor_rollout_ref.model.path={r['model']}",
        f"trainer.experiment_name={r['experiment']}",
        f"data.train_files={r['data']}",
    ]

    print(f"Command:\n  {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
