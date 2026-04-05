#!/usr/bin/env python3
"""
RAFT SFT Training — 8×H100 single node
Freezes MoE expert layers (MXFP4), trains only dense bf16 layers (~2.13B params).
Uses device_map="auto" to load the model across GPUs in native mixed format
(bf16 dense + MXFP4 experts) without dequantization.

Usage (from repo root — run with plain python, NOT torchrun):
  python3.12 training/raft/train_raft.py --data ./data/raft/raft_hard_all.parquet
  python3.12 training/raft/train_raft.py --data ./data/raft/raft_hard_all.parquet --hf-repo ritwikareddykancharla/gpt-oss-120b-raft

  # Curriculum RAFT
  python3.12 training/raft/train_raft.py --data ./data/raft/raft_tier1_hardest.parquet --output ./outputs/raft_r1
  python3.12 training/raft/train_raft.py --data ./data/raft/raft_tier2_hard.parquet --resume ./outputs/raft_r1 --output ./outputs/raft_r2
"""

import argparse, os, json, sys
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    DataCollatorForSeq2Seq,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class RAFTDataset(Dataset):
    """Loads RAFT parquet, tokenizes with chat template, masks non-assistant tokens."""

    def __init__(self, parquet_path, tokenizer, max_seq_len=4096):
        df = pd.read_parquet(parquet_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples = []
        skipped = 0

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Tokenizing"):
            messages = json.loads(row["messages"])
            result = self._tokenize(messages)
            if result is None:
                skipped += 1
                continue
            self.examples.append(result)

        print(f"Loaded {len(self.examples):,} examples, skipped {skipped:,} (too long or bad format)")

    def _clean_messages(self, messages):
        """Clean messages for chat template. Preserve tool_calls for TIR support."""
        clean = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content) if content is not None else ""

            # Map non-standard roles
            if role in ("tool_call", "function_call"):
                role = "assistant"
            elif role in ("tool_response", "function", "ipython"):
                role = "tool"
            elif role not in ("system", "user", "assistant", "tool", "developer"):
                role = "user"

            msg = {"role": role, "content": content}

            # Preserve tool_calls if present and non-empty
            tool_calls = m.get("tool_calls")
            if tool_calls:
                msg["tool_calls"] = tool_calls

            # Preserve tool_call_id for tool responses
            tool_call_id = m.get("tool_call_id")
            if tool_call_id and role == "tool":
                msg["tool_call_id"] = tool_call_id

            clean.append(msg)
        return clean

    def _apply_chat_template(self, messages):
        """Apply chat template and return plain list of token IDs."""
        result = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            return_tensors=None,
        )
        # apply_chat_template returns BatchEncoding — extract input_ids
        if hasattr(result, "keys") and "input_ids" in result:
            return result["input_ids"]
        # If it's already a plain list of ints, use directly
        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], int):
            return result
        # Fallback: tokenize the text output manually
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        return self.tokenizer.encode(text)

    def _tokenize(self, messages):
        """Tokenize with chat template. Mask everything except assistant responses."""
        clean = self._clean_messages(messages)

        try:
            full_ids = self._apply_chat_template(clean)
        except Exception:
            return None

        if len(full_ids) > self.max_seq_len:
            return None

        # Build labels: -100 for non-assistant tokens, train on all assistant turns
        labels = [-100] * len(full_ids)

        prefix_len = 0
        for i, msg in enumerate(clean):
            try:
                partial = self._apply_chat_template(clean[:i+1])
            except Exception:
                return None

            msg_start = prefix_len
            msg_end = len(partial)

            if msg["role"] == "assistant":
                for j in range(msg_start, min(msg_end, len(full_ids))):
                    labels[j] = full_ids[j]

            prefix_len = msg_end

        if all(l == -100 for l in labels):
            return None

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# ---------------------------------------------------------------------------
# Freeze MoE experts
# ---------------------------------------------------------------------------
def freeze_moe_experts(model):
    """Freeze MoE expert layers (MXFP4 quantized).
    Only train dense bf16 layers: attention, layernorms, embeddings, lm_head.
    """
    total_params = 0
    frozen_params = 0
    trained_params = 0
    trained_names = []

    for name, param in model.named_parameters():
        total_params += param.numel()

        if any(k in name for k in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps",
                                     "experts.", "gate_up_proj_exps", "down_proj_exps"]):
            param.requires_grad = False
            frozen_params += param.numel()
        elif "gate" in name and "router" in name.lower():
            param.requires_grad = False
            frozen_params += param.numel()
        else:
            param.requires_grad = True
            trained_params += param.numel()
            trained_names.append(name)

    print(f"\nParameter freeze summary:")
    print(f"  Total params:   {total_params/1e9:.2f}B")
    print(f"  Frozen (MoE):   {frozen_params/1e9:.2f}B")
    print(f"  Trainable:      {trained_params/1e9:.2f}B")
    print(f"\n  Trainable layer types:")

    from collections import Counter
    types = Counter()
    for n in trained_names:
        parts = n.split(".")
        layer_type = ".".join(parts[-2:]) if len(parts) >= 2 else n
        types[layer_type] += 1
    for t, c in types.most_common(20):
        print(f"    {t}: {c}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="Path to RAFT parquet")
    p.add_argument("--model", type=str, default="unsloth/gpt-oss-120b",
                   help="Model name or path (default: unsloth/gpt-oss-120b)")
    p.add_argument("--output", type=str, default="./outputs/raft",
                   help="Output directory (default: ./outputs/raft)")
    p.add_argument("--resume", type=str, default=None,
                   help="Resume from checkpoint dir (for curriculum rounds)")
    p.add_argument("--max-seq-len", type=int, default=16384)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1,
                   help="Per-GPU micro batch size (default: 1)")
    p.add_argument("--grad-accum", type=int, default=4,
                   help="Gradient accumulation steps (default: 4, effective batch = 8*1*4 = 32)")
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--hf-repo", type=str, default=None,
                   help="HuggingFace repo to push model, e.g. ritwikareddykancharla/gpt-oss-120b-raft")
    return p.parse_args()


def main():
    args = parse_args()

    # Unset conflicting env vars
    for var in ["PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"]:
        os.environ.pop(var, None)

    # Load HF token from .env
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
    hf_token = os.environ.get("HF_TOKEN")

    print(f"Loading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -----------------------------------------------------------------------
    # Load model with device_map="auto" to:
    #   1. Keep MoE experts in native MXFP4 (no dequantization needed)
    #   2. Shard layers across all available GPUs automatically
    #   3. Keep dense layers in bf16 as stored in checkpoint
    # -----------------------------------------------------------------------
    model_name_or_path = args.resume or args.model
    print(f"Loading model from {model_name_or_path} with device_map='auto'...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    freeze_moe_experts(model)
    model.gradient_checkpointing_enable()

    print(f"\nLoading dataset from {args.data}...")
    dataset = RAFTDataset(args.data, tokenizer, max_seq_len=args.max_seq_len)

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    os.makedirs(args.output, exist_ok=True)

    num_gpus = torch.cuda.device_count()
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="wandb",
        run_name=f"raft-{os.path.basename(args.data).replace('.parquet', '')}",
        dataloader_num_workers=4,
        remove_unused_columns=False,
        # No FSDP — device_map="auto" handles multi-GPU sharding
        # Push every checkpoint to HF
        push_to_hub=True if args.hf_repo else False,
        hub_model_id=args.hf_repo,
        hub_token=hf_token,
        hub_strategy="checkpoint",
        hub_private_repo=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    print(f"\nStarting training...")
    print(f"  Dataset:          {len(dataset):,} examples")
    print(f"  Epochs:           {args.epochs}")
    print(f"  Batch (per GPU):  {args.batch_size}")
    print(f"  Grad accum:       {args.grad_accum}")
    print(f"  Effective batch:  {num_gpus * args.batch_size * args.grad_accum}")
    print(f"  Learning rate:    {args.lr}")
    print(f"  Max seq len:      {args.max_seq_len}")
    print(f"  Output:           {args.output}")
    print(f"  GPUs:             {num_gpus}")
    print(f"  Strategy:         device_map='auto' (model-parallel)")

    trainer.train(resume_from_checkpoint=args.resume if args.resume and os.path.isdir(args.resume) else None)

    # Save final model
    final_dir = os.path.join(args.output, "final")
    print(f"\nSaving full model to {final_dir}...")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    if args.hf_repo:
        print(f"Pushing final model to https://huggingface.co/{args.hf_repo}")
        trainer.push_to_hub(commit_message="training complete")

    print(f"\nDone! Model saved to {final_dir}")


if __name__ == "__main__":
    main()
