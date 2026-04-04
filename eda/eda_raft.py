"""
EDA for RAFT training parquet files.

Usage:
    python3.12 eda/eda_raft.py
"""

import pandas as pd


DATA_DIR = "data"
FILES = ["raft_stage1_all_hard_tir", "raft_stage2_aops_hard_tir"]


def total_char_len(msgs):
    t = 0
    for m in msgs:
        c = m.get("content", "")
        t += len(c) if isinstance(c, str) else sum(len(str(b)) for b in c)
    return t


def main():
    frames = {}
    for name in FILES:
        path = f"{DATA_DIR}/{name}.parquet"
        df = pd.read_parquet(path)
        frames[name] = df
        print(f"=== {name} ===")
        print(f"Rows: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        if "data_source" in df.columns:
            print(f"Sources: {df['data_source'].value_counts().to_dict()}")
        print()

    # Check if stage1 and stage2 are identical
    s1, s2 = frames[FILES[0]], frames[FILES[1]]
    print("=== Stage 1 vs Stage 2 ===")
    print(f"Same shape: {s1.shape == s2.shape}")
    print(f"Identical: {s1.equals(s2)}")
    if "problem" in s1.columns and "problem" in s2.columns:
        print(f"Same problems: {set(s1['problem']) == set(s2['problem'])}")
        overlap = set(s1["problem"]) & set(s2["problem"])
        print(f"Overlap: {len(overlap)} / s1={len(s1)} / s2={len(s2)}")
    print()

    # Inspect first 3 examples
    for idx in range(min(3, len(s1))):
        msgs = s1.iloc[idx]["messages"]
        print(f"=== Example {idx}: {len(msgs)} messages ===")
        for i, m in enumerate(msgs):
            role = m.get("role", "?")
            content = m.get("content", "")
            tc = m.get("tool_calls", None)
            clen = len(content) if isinstance(content, str) else f"list({len(content)})"
            print(f"  [{i}] role={role}, content_len={clen}")
            if isinstance(content, str) and content:
                print(f"       {content[:300]}")
            elif isinstance(content, list):
                for b in content[:2]:
                    print(f"       {str(b)[:300]}")
            if tc:
                print(f"       tool_calls: {str(tc)[:300]}")
        print()

    # Tools field
    print("=== Tools field (first example) ===")
    tools = s1.iloc[0].get("tools", None)
    if tools:
        for t in (tools if isinstance(tools, list) else [tools]):
            print(f"  {str(t)[:400]}")
    else:
        print("  None")
    print()

    # Char length stats
    lens = s1["messages"].apply(total_char_len)
    print("=== Char length stats (stage 1) ===")
    print(f"min={lens.min()}, max={lens.max()}, mean={lens.mean():.0f}, median={lens.median():.0f}")
    print()
    print("Distribution:")
    for threshold in [500, 1000, 2000, 4000, 8000, 16000, 25000]:
        count = (lens <= threshold).sum()
        print(f"  <= {threshold:>6}: {count:>6} ({count/len(s1)*100:.1f}%)")
    print()

    # Message count stats
    msg_counts = s1["messages"].apply(len)
    print("=== Messages per example (stage 1) ===")
    print(f"min={msg_counts.min()}, max={msg_counts.max()}, mean={msg_counts.mean():.1f}, median={msg_counts.median():.0f}")
    print()
    print("Distribution:")
    for n in [2, 4, 6, 8, 10, 12, 16, 20, 30]:
        count = (msg_counts <= n).sum()
        print(f"  <= {n:>3} msgs: {count:>6} ({count/len(s1)*100:.1f}%)")
    print()

    # Role distribution across all messages
    role_counts = {}
    for msgs in s1["messages"]:
        for m in msgs:
            r = m.get("role", "unknown")
            role_counts[r] = role_counts.get(r, 0) + 1
    print("=== Role distribution (all messages, stage 1) ===")
    for role, count in sorted(role_counts.items(), key=lambda x: -x[1]):
        print(f"  {role}: {count}")
    print()

    # Expected answer stats
    if "expected_answer" in s1.columns:
        answers = s1["expected_answer"].astype(str)
        numeric = pd.to_numeric(answers, errors="coerce")
        print("=== Expected answers (stage 1) ===")
        print(f"Total: {len(answers)}")
        print(f"Numeric: {numeric.notna().sum()}")
        print(f"Non-numeric: {numeric.isna().sum()}")
        if numeric.notna().any():
            n = numeric.dropna()
            print(f"Numeric range: {n.min():.0f} to {n.max():.0f}")
            print(f"Numeric mean: {n.mean():.1f}")
            print(f"Integer answers: {(n == n.astype(int)).sum()}")


if __name__ == "__main__":
    main()
