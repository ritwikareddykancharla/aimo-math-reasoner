"""
Two-stage RAFT training for GPT-OSS-120B on Nemotron-Math-v2.

Stage 1: Broad hard problems (AoPS + StackExchange) with TIR
Stage 2: AoPS competition-focused with TIR

Uses DeepSpeed ZeRO-3 with CPU offload across 2 nodes (16x H100).
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)


def load_parquet_as_dataset(path: str) -> Dataset:
    """Load a parquet file as a HuggingFace Dataset."""
    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} rows from {path}")
    return Dataset.from_pandas(df, preserve_index=False)


def tokenize_messages(example, tokenizer, max_seq_len: int = 4096):
    """
    Tokenize a messages-format example using the model's chat template.

    The messages field contains the full conversation including tool calls/responses.
    We use the tokenizer's chat template to format it properly for the harmony format.
    """
    messages = example["messages"]

    # Apply chat template - this handles the harmony format automatically
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=example.get("tools", None) or None,
        )
    except Exception:
        # Fallback: try without tools
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception as e:
            print(f"Warning: Failed to apply chat template: {e}")
            # Last resort: concatenate messages manually
            parts = []
            for msg in messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if isinstance(content, str):
                    parts.append(f"<|{role}|>\n{content}")
            text = "\n".join(parts)

    # Tokenize
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_seq_len,
        padding=False,
        return_tensors=None,
    )

    # For causal LM, labels = input_ids (shifted internally by the model)
    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized


def main():
    parser = argparse.ArgumentParser(description="RAFT SFT Training for GPT-OSS-120B")
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2],
        required=True,
        help="Training stage (1=broad, 2=competition-focused)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data/models/gpt-oss-120b",
        help="Path to base model or Stage 1 checkpoint",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/data/aimo-math-reasoner/data",
        help="Directory containing prepared parquet files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for checkpoints (default: /data/checkpoints/raft_stage{N})",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=4096,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        default=1,
        help="Per-device train batch size",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Gradient accumulation steps (effective batch = per_device * num_gpus * grad_accum)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
        help="Warmup ratio",
    )
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="steps",
        help="Save strategy",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=200,
        help="Save every N steps",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log every N steps",
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="Path to DeepSpeed config JSON",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank (set by DeepSpeed launcher)",
    )

    args = parser.parse_args()

    # --- Resolve paths ---
    if args.output_dir is None:
        args.output_dir = f"/data/checkpoints/raft_stage{args.stage}"

    if args.stage == 1:
        data_path = os.path.join(args.data_dir, "raft_stage1_all_hard_tir.parquet")
    else:
        data_path = os.path.join(args.data_dir, "raft_stage2_aops_hard_tir.parquet")

    # --- Unset conflicting env vars (critical for MoE models) ---
    for var in ["PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"]:
        os.environ.pop(var, None)

    # --- Load tokenizer ---
    print(f"Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load and tokenize data ---
    print(f"Loading data from {data_path}")
    dataset = load_parquet_as_dataset(data_path)

    print("Tokenizing dataset...")
    tokenized = dataset.map(
        lambda ex: tokenize_messages(ex, tokenizer, args.max_seq_len),
        remove_columns=dataset.column_names,
        desc="Tokenizing",
        num_proc=32,
    )

    # Filter out empty sequences
    tokenized = tokenized.filter(lambda x: len(x["input_ids"]) > 10)
    print(f"Tokenized dataset size: {len(tokenized)}")

    # Print token stats
    lengths = [len(x["input_ids"]) for x in tokenized]
    print(f"Token lengths - min: {min(lengths)}, max: {max(lengths)}, "
          f"mean: {sum(lengths)/len(lengths):.0f}, "
          f"median: {sorted(lengths)[len(lengths)//2]}")

    # --- Load model ---
    print(f"Loading model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False  # Required for gradient checkpointing

    # --- Training arguments ---
    # Effective batch size = per_device * 16 gpus * grad_accum
    # = 1 * 16 * 2 = 32
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.1,
        bf16=True,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=3,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        deepspeed=args.deepspeed,
        report_to="wandb",
        run_name=f"raft-stage{args.stage}-gptoss120b",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        ddp_timeout=7200,  # 2 hours - generous for multi-node
    )

    # --- Data collator ---
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_length=args.max_seq_len,
        pad_to_multiple_of=8,
    )

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # --- Train ---
    print(f"\n{'='*60}")
    print(f"Starting RAFT Stage {args.stage} Training")
    print(f"  Model: {args.model_path}")
    print(f"  Data: {data_path} ({len(tokenized)} examples)")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Effective batch size: {args.per_device_batch_size * 16 * args.gradient_accumulation_steps}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Max seq len: {args.max_seq_len}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*60}\n")

    trainer.train()

    # --- Save final model ---
    print("Saving final model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
