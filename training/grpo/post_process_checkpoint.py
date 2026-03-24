# post_process_checkpoint.py
"""
Run after each training round to prepare checkpoint for inference.

Usage:
  python3.12 training/grpo/post_process_checkpoint.py --round 1
"""

import json
import glob
import os
import argparse

ORIG_MODEL_CACHE = os.path.expanduser(
    "~/.cache/huggingface/hub/models--openai--gpt-oss-120b/snapshots"
)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CKPT_ROOT    = os.path.join(PROJECT_ROOT, "checkpoints")

ROUNDS = {
    1: os.path.join(CKPT_ROOT, "grpo-round1-final"),
    2: os.path.join(CKPT_ROOT, "grpo-round2-final"),
    3: os.path.join(CKPT_ROOT, "grpo-round3-final"),
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True, choices=[1, 2, 3])
    args = parser.parse_args()

    ckpt_dir = ROUNDS[args.round]

    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_dir}\n"
            f"Did round {args.round} finish successfully?"
        )

    # Get original quantization_config from base model
    orig_configs = glob.glob(f"{ORIG_MODEL_CACHE}/*/config.json")
    if not orig_configs:
        raise FileNotFoundError(f"Original model config not found in {ORIG_MODEL_CACHE}")

    with open(orig_configs[0]) as f:
        orig = json.load(f)

    quant_config = orig.get("quantization_config")
    if not quant_config:
        raise ValueError("No quantization_config found in original model config")

    # Patch into checkpoint config
    ckpt_config_path = os.path.join(ckpt_dir, "config.json")
    if not os.path.exists(ckpt_config_path):
        raise FileNotFoundError(f"No config.json in checkpoint: {ckpt_config_path}")

    with open(ckpt_config_path) as f:
        ckpt_config = json.load(f)

    ckpt_config["quantization_config"] = quant_config

    with open(ckpt_config_path, "w") as f:
        json.dump(ckpt_config, f, indent=2)

    print(f"\n✅ Round {args.round} checkpoint ready for inference")
    print(f"   Path:        {ckpt_dir}")
    print(f"   Quant:       mxfp4 (load-time via vLLM)")
    print(f"   Skipping:    {quant_config['modules_to_not_convert']}")
    print(f"\nLoad with:")
    print(f"   from vllm import LLM")
    print(f"   llm = LLM(")
    print(f"       model='{ckpt_dir}',")
    print(f"       quantization='mxfp4',")
    print(f"       tensor_parallel_size=8,")
    print(f"       trust_remote_code=True,")
    print(f"       dtype='bfloat16',")
    print(f"   )")

if __name__ == "__main__":
    main()
