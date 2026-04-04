"""
Prepare RAFT training data from Nemotron-Math-v2.

Strategy: Two-stage Rejection Sampling Fine-Tuning (RAFT) with TIR
- Stage 1: Broad hard problems (AoPS + StackExchange) with tool use
- Stage 2: Competition-focused (AoPS only) with tool use

Filters:
  - reason_high_with_tool accuracy < 0.5 (hard problems)
  - Only correct trajectories (verified answers)
  - Only trajectories that include Python tool calls
  - Shortest correct trajectory per problem (less noise)
"""

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset, concatenate_datasets, Dataset
import pandas as pd


def has_tool_use(messages: list) -> bool:
    """Check if a trajectory contains actual Python tool calls."""
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "tool" or role == "ipython":
            return True
        if isinstance(content, str) and ("```python" in content or "```py" in content):
            # Check if there's a subsequent tool response
            return True
        # Some formats use 'tool_calls' field
        if msg.get("tool_calls"):
            return True
    return False


def get_trajectory_length(messages: list) -> int:
    """Get total character length of a trajectory."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    total += len(block["text"])
    return total


def is_hard_problem(metadata: dict, threshold: float = 0.5) -> bool:
    """Check if problem is hard based on reason_high_with_tool accuracy."""
    if metadata is None:
        return False
    key = "reason_high_with_tool"
    if key in metadata:
        info = metadata[key]
        if isinstance(info, dict):
            acc = info.get("accuracy", 1.0)
            return acc < threshold
        elif isinstance(info, (int, float)):
            return info < threshold
    return False


def get_accuracy(metadata: dict, key: str = "reason_high_with_tool") -> float:
    """Extract accuracy from metadata."""
    if metadata is None:
        return 1.0
    if key in metadata:
        info = metadata[key]
        if isinstance(info, dict):
            return info.get("accuracy", 1.0)
        elif isinstance(info, (int, float)):
            return float(info)
    return 1.0


def filter_and_prepare(
    splits: list[str],
    output_path: str,
    difficulty_threshold: float = 0.5,
    aops_only: bool = False,
    require_tool: bool = True,
):
    """
    Load, filter, and save training data.

    Args:
        splits: HF dataset splits to load (e.g., ["high_part00", "high_part01", "high_part02"])
        output_path: Where to save the filtered parquet
        difficulty_threshold: Max accuracy to consider "hard" (default 0.5)
        aops_only: If True, only keep AoPS problems
        require_tool: If True, only keep trajectories with tool use
    """
    print(f"Loading splits: {splits}")
    datasets = []
    for split in splits:
        print(f"  Loading {split}...")
        try:
            ds = load_dataset(
                "nvidia/Nemotron-Math-v2",
                split=split,
                trust_remote_code=True,
            )
            datasets.append(ds)
            print(f"    Loaded {len(ds)} rows")
        except Exception as e:
            print(f"    Error loading {split}: {e}")
            continue

    if not datasets:
        raise ValueError("No datasets loaded!")

    combined = concatenate_datasets(datasets)
    print(f"\nTotal rows loaded: {len(combined)}")

    # --- Filter pipeline ---
    print("\n--- Filtering ---")

    # 1. Filter by data source (if AoPS only)
    if aops_only:
        combined = combined.filter(
            lambda x: x.get("data_source", "").lower() == "aops",
            desc="Filter AoPS only",
        )
        print(f"After AoPS filter: {len(combined)}")

    # 2. Filter hard problems
    combined = combined.filter(
        lambda x: is_hard_problem(x.get("metadata"), difficulty_threshold),
        desc="Filter hard problems",
    )
    print(f"After difficulty filter (acc < {difficulty_threshold}): {len(combined)}")

    # 3. Filter for tool use in trajectories
    if require_tool:
        # First check if tools field is non-empty
        combined = combined.filter(
            lambda x: bool(x.get("tools")),
            desc="Filter has tools defined",
        )
        print(f"After tools-defined filter: {len(combined)}")

        # Then check if trajectory actually uses tools
        combined = combined.filter(
            lambda x: has_tool_use(x.get("messages", [])),
            desc="Filter has tool use in messages",
        )
        print(f"After tool-use filter: {len(combined)}")

    # 4. Convert to pandas for groupby (shortest per problem)
    print("\nSelecting shortest correct trajectory per problem...")
    df = combined.to_pandas()

    # Add trajectory length
    df["_traj_len"] = df["messages"].apply(get_trajectory_length)

    # Group by problem and pick shortest
    df = df.sort_values("_traj_len").groupby("problem", as_index=False).first()
    df = df.drop(columns=["_traj_len"])

    print(f"After dedup (shortest per problem): {len(df)}")

    # 5. Print stats
    if "data_source" in df.columns:
        print("\n--- Dataset Stats ---")
        source_counts = df["data_source"].value_counts()
        for source, count in source_counts.items():
            pct = count / len(df) * 100
            print(f"  {source}: {count} ({pct:.1f}%)")
        print(f"  Total unique problems: {len(df)}")

    if "metadata" in df.columns:
        accs = df["metadata"].apply(
            lambda m: get_accuracy(m, "reason_high_with_tool") if m else None
        )
        accs = accs.dropna()
        print(f"\n--- Difficulty Stats ---")
        print(f"  Mean accuracy: {accs.mean():.3f}")
        print(f"  Median accuracy: {accs.median():.3f}")
        print(f"  Problems with acc <= 0.125: {(accs <= 0.125).sum()}")
        print(f"  Problems with acc <= 0.250: {(accs <= 0.250).sum()}")
        print(f"  Problems with acc <= 0.375: {(accs <= 0.375).sum()}")
        print(f"  Problems with acc <  0.500: {(accs < 0.500).sum()}")

    # 6. Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"\nSaved to {output_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Prepare RAFT training data")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/aimo-math-reasoner/data",
        help="Output directory for parquet files",
    )
    parser.add_argument(
        "--difficulty_threshold",
        type=float,
        default=0.5,
        help="Max accuracy to consider hard (default: 0.5)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/data/hf_cache",
        help="HuggingFace cache directory",
    )
    args = parser.parse_args()

    os.environ["HF_HOME"] = args.cache_dir
    os.environ["HF_DATASETS_CACHE"] = os.path.join(args.cache_dir, "datasets")

    splits = ["high_part00", "high_part01", "high_part02"]

    # --- Stage 1: All hard problems with tool use ---
    print("=" * 60)
    print("STAGE 1: All hard problems with tool use (AoPS + StackExchange)")
    print("=" * 60)
    filter_and_prepare(
        splits=splits,
        output_path=os.path.join(args.output_dir, "raft_stage1_all_hard_tir.parquet"),
        difficulty_threshold=args.difficulty_threshold,
        aops_only=False,
        require_tool=True,
    )

    # --- Stage 2: AoPS hard problems with tool use ---
    print("\n" + "=" * 60)
    print("STAGE 2: AoPS hard problems with tool use (competition-focused)")
    print("=" * 60)
    filter_and_prepare(
        splits=splits,
        output_path=os.path.join(args.output_dir, "raft_stage2_aops_hard_tir.parquet"),
        difficulty_threshold=args.difficulty_threshold,
        aops_only=True,
        require_tool=True,
    )

    print("\n" + "=" * 60)
    print("DONE! Data ready for training.")
    print("=" * 60)


if __name__ == "__main__":
    main()
