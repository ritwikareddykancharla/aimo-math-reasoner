"""
Fix parquets: move gold_solution into extra_info so veRL passes it to compute_score.

veRL's NaiveRewardManager passes:
  - ground_truth  ← from reward_model["ground_truth"]
  - extra_info    ← from non_tensor_batch["extra_info"]

So gold_solution must be in extra_info, not reward_model.

Usage:
    python3.12 training/grpo/fix_parquets.py
"""

import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "data")

PARQUETS = [
    "grpo_aops_acc0125.parquet",
    "grpo_aops_acc025.parquet",
    "grpo_aops_acc0375.parquet",
    "grpo_hard_aops.parquet",
]

def fix_parquet(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"  Skipping {filename} — not found")
        return

    df = pd.read_parquet(path)
    print(f"\n{filename}: {len(df)} rows")
    print(f"  Columns before: {df.columns.tolist()}")

    # Check if already fixed
    if "extra_info" in df.columns:
        sample = df["extra_info"].iloc[0]
        if isinstance(sample, dict) and "gold_solution" in sample:
            print(f"  Already fixed — skipping")
            return

    # Move gold_solution from reward_model → extra_info
    def build_extra_info(row):
        rm = row["reward_model"]
        gold = rm.get("gold_solution", "") if isinstance(rm, dict) else ""
        return {"gold_solution": gold}

    df["extra_info"] = df.apply(build_extra_info, axis=1)

    # Verify ground_truth is still in reward_model
    sample_rm = df["reward_model"].iloc[0]
    assert "ground_truth" in sample_rm, f"ground_truth missing from reward_model!"

    # Save backup then overwrite
    backup_path = path + ".bak"
    if not os.path.exists(backup_path):
        df_orig = pd.read_parquet(path)
        df_orig.to_parquet(backup_path, index=False)
        print(f"  Backup saved: {backup_path}")

    df.to_parquet(path, index=False)
    print(f"  Columns after: {df.columns.tolist()}")

    # Verify
    df_check = pd.read_parquet(path)
    sample_extra = df_check["extra_info"].iloc[0]
    sample_rm    = df_check["reward_model"].iloc[0]
    print(f"  ✓ extra_info keys:   {list(sample_extra.keys())}")
    print(f"  ✓ reward_model keys: {list(sample_rm.keys())}")
    print(f"  ✓ ground_truth:      {str(sample_rm['ground_truth'])[:60]}")
    print(f"  ✓ gold_solution:     {str(sample_extra['gold_solution'])[:60]}")


if __name__ == "__main__":
    print("Fixing parquets to add extra_info with gold_solution...")
    for f in PARQUETS:
        fix_parquet(f)
    print("\nAll done! ✓")
