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

ROUNDS = {
    1: {
        "config":     "grpo_round1",
        "data":       "./data/grpo_aops_acc0125.parquet",
        "experiment": "gpt-oss-120b-round1",
        "model":      "openai/gpt-oss-120b",   # base model
    },
    2: {
        "config":     "grpo_round2",
        "data":       "./data/grpo_aops_acc025.parquet",
        "experiment": "gpt-oss-120b-round2",
        "model":      "./checkpoints/grpo-round1-final",  # from round 1
    },
    3: {
        "config":     "grpo_round3",
        "data":       "./data/grpo_aops_acc0375.parquet",
        "experiment": "gpt-oss-120b-round3",
        "model":      "./checkpoints/grpo-round2-final",  # from round 2
    },
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True, choices=[1, 2, 3])
    args = parser.parse_args()

    r = ROUNDS[args.round]
    print(f"\n{'='*60}")
    print(f"  GRPO Round {args.round}")
    print(f"  Data:   {r['data']}")
    print(f"  Model:  {r['model']}")
    print(f"  Exp:    {r['experiment']}")
    print(f"{'='*60}\n")

    cmd = [
        "python3.12", "-m", "verl.trainer.main_ppo",
        f"--config-path=training/grpo",
        f"--config-name={r['config']}",
        # Override model path and experiment name per round
        f"actor_rollout_ref.model.path={r['model']}",
        f"trainer.experiment_name={r['experiment']}",
        f"data.train_files={r['data']}",
    ]

    print(f"Running: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
