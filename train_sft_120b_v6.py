"""
AIMO3: Curriculum SFT on GPT-OSS-120B
======================================
Stage 1: Train on acc=0.375 (easiest hard) → merge → upload
Stage 2: Load stage1 model → train on acc=0.25 → merge → upload
Stage 3: Load stage2 model → train on acc=0.125 (hardest) → merge → upload

Usage:
    # Run all 3 stages
    python3 train_curriculum.py

    # Run specific stage
    python3 train_curriculum.py --stage 1
    python3 train_curriculum.py --stage 2 --base_model merged-stage1
    python3 train_curriculum.py --stage 3 --base_model merged-stage2
"""

import unsloth  # noqa: F401 — must be first

import os, sys, time, gc, shutil, glob, argparse
from dotenv import load_dotenv
load_dotenv()

KAGGLE_USERNAME = os.environ["KAGGLE_USERNAME"]
DATA_DIR = os.environ.get("DATA_DIR", "./data/nemotron-sft-high-only/hf_dataset")

# ============================================================
# PARSE ARGS
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--stage", type=int, default=0, help="Run specific stage (1/2/3). 0=all")
parser.add_argument("--base_model", type=str, default=None, help="Path to base model for stages 2/3")
args = parser.parse_args()

# ============================================================
# STAGE CONFIG
# ============================================================
STAGES = [
    {
        "stage": 1,
        "accuracy": 0.125,
        "label": "hardest",
        "lr": 2e-5,
        "epochs": 2,        # 2 epochs — smallest set, needs more exposure
    },
    {
        "stage": 2,
        "accuracy": 0.25,
        "label": "hard",
        "lr": 1e-5,
        "epochs": 1,
    },
    {
        "stage": 3,
        "accuracy": 0.375,
        "label": "easiest-hard",
        "lr": 5e-6,
        "epochs": 1,
    },
]

# ============================================================
# W&B
# ============================================================
WANDB_KEY = os.environ.get("WANDB_API_KEY", "")
if WANDB_KEY:
    try:
        import wandb
        wandb.login(key=WANDB_KEY)
        os.environ["WANDB_PROJECT"] = "aimo3-curriculum"
        REPORT_TO = "wandb"
        print("W&B: enabled")
    except Exception:
        REPORT_TO = "none"
else:
    REPORT_TO = "none"

# ============================================================
# IMPORTS
# ============================================================
from datasets import load_from_disk
from transformers import set_seed, TrainerCallback
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
import torch

set_seed(42)

MAX_SEQ_LEN = 4096

# ============================================================
# HELPER: Run one stage
# ============================================================

def run_stage(stage_config, base_model_path):
    stage_num = stage_config["stage"]
    acc_target = stage_config["accuracy"]
    lr = stage_config["lr"]
    epochs = stage_config["epochs"]
    label = stage_config["label"]

    output_dir = f"sft-stage{stage_num}"
    merged_dir = f"merged-stage{stage_num}"

    print("\n" + "=" * 60)
    print(f"  STAGE {stage_num}: acc={acc_target} ({label})")
    print(f"  Base model: {base_model_path}")
    print(f"  LR: {lr} | Epochs: {epochs}")
    print("=" * 60)

    # ── Load model ──
    print(f"\n  Loading model from {base_model_path}...")
    t0 = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_path,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        fast_inference=False,
        max_lora_rank=32,
    )
    print(f"  Loaded in {time.time()-t0:.0f}s | GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    model = FastLanguageModel.get_peft_model(
        model, r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        use_gradient_checkpointing="unsloth",
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {trainable/1e6:.0f}M params")

    # ── Load + filter dataset ──
    print(f"\n  Loading dataset and filtering acc={acc_target}, tools only...")
    ds = load_from_disk(DATA_DIR)
    ds = ds.filter(lambda x: x['accuracy'] == acc_target and x['has_tools'] == True)
    ds = ds.shuffle(seed=42)
    print(f"  Filtered: {len(ds)} rows")

    def format_for_sft(example):
        msgs = example['messages']
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False,
        )
        return {"text": text}

    print(f"  Formatting...")
    sft_data = ds.map(format_for_sft, num_proc=8, remove_columns=ds.column_names)
    sft_data = sft_data.filter(lambda x: len(x['text']) > 100)
    print(f"  Ready: {len(sft_data)} samples")

    # Token length check
    sample_lens = []
    for i in range(min(500, len(sft_data))):
        sample_lens.append(len(tokenizer.encode(sft_data[i]['text'])))
    sample_lens.sort()
    n = len(sample_lens)
    print(f"  Tokens: median={sample_lens[n//2]} p90={sample_lens[9*n//10]} max={sample_lens[-1]}")

    # ── Train ──
    print(f"\n  Training stage {stage_num}...")
    FastLanguageModel.for_training(model)

    trainer = SFTTrainer(
        model=model,
        train_dataset=sft_data,
        args=SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=32,
            gradient_accumulation_steps=1,
            num_train_epochs=epochs,
            learning_rate=lr,
            bf16=True,
            max_seq_length=MAX_SEQ_LEN,
            lr_scheduler_type="cosine",
            warmup_ratio=0,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=2,
            logging_steps=1,
            eval_strategy="no",
            max_grad_norm=0.5,
            report_to=REPORT_TO,
            run_name=f"stage{stage_num}-acc{acc_target}",
            dataset_text_field="text",
            packing=True,
        ),
        processing_class=tokenizer,
    )

    t0 = time.time()
    # Resume if checkpoint exists
    checkpoints = sorted(glob.glob(f"{output_dir}/checkpoint-*"))
    if checkpoints:
        print(f"  Resuming from {checkpoints[-1]}...")
        trainer.train(resume_from_checkpoint=checkpoints[-1])
    else:
        trainer.train()

    train_time = time.time() - t0
    final_step = trainer.state.global_step
    final_loss = trainer.state.log_history[-1].get('loss', '?') if trainer.state.log_history else '?'
    print(f"\n  Stage {stage_num} done in {train_time/3600:.1f}h | step={final_step} | loss={final_loss}")

    # ── Merge ──
    print(f"\n  Merging to 4bit → {merged_dir}...")
    t0 = time.time()
    model.save_pretrained_merged(
        merged_dir,
        tokenizer,
        save_method="merged_4bit_forced",
    )
    merge_mins = (time.time() - t0) / 60
    print(f"  ✅ Merge done in {merge_mins:.1f} mins")

    # Write readme
    readme = (
        f"# GPT-OSS-120B AIMO3 — Stage {stage_num} ({label})\n\n"
        f"| Key | Value |\n"
        f"|-----|-------|\n"
        f"| Stage | {stage_num} |\n"
        f"| Accuracy | {acc_target} |\n"
        f"| Step | {final_step} |\n"
        f"| Loss | {final_loss} |\n"
        f"| LR | {lr} |\n"
        f"| Epochs | {epochs} |\n"
        f"| Train samples | {len(sft_data)} |\n"
        f"| Base model | {base_model_path} |\n"
        f"| Train time | {train_time/3600:.1f}h |\n"
    )
    with open(os.path.join(merged_dir, "README.md"), "w") as f:
        f.write(readme)

    # ── Upload to Kaggle ──
    print(f"\n  Uploading to Kaggle...")
    try:
        import kagglehub
        handle = f"{KAGGLE_USERNAME}/gpt-oss-120b-aimo3-sft-v3/transformers/default"
        t0 = time.time()
        kagglehub.model_upload(
            handle, merged_dir,
            version_notes=f"Stage {stage_num} ({label}) | acc={acc_target} | loss={final_loss} | {len(sft_data)} samples",
            license_name="Apache 2.0",
        )
        upload_mins = (time.time() - t0) / 60
        print(f"  ✅ Upload done in {upload_mins:.1f} mins")
    except Exception as e:
        print(f"  ⚠️  Upload failed: {e}")
        print(f"  Merged model at: {merged_dir}")

    # ── Cleanup ──
    del model, tokenizer, trainer, sft_data
    gc.collect()
    torch.cuda.empty_cache()

    # Clean trainer checkpoints (we have the merged model)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"  🗑  Cleaned: {output_dir}")

    print(f"\n  Stage {stage_num} complete. Merged model: {merged_dir}")
    return merged_dir

# ============================================================
# MAIN
# ============================================================

if args.stage > 0:
    # Run single stage
    stage_config = STAGES[args.stage - 1]
    base = args.base_model or "unsloth/gpt-oss-120b-unsloth-bnb-4bit"
    run_stage(stage_config, base)

else:
    # Run all 3 stages sequentially
    base_model = "unsloth/gpt-oss-120b-unsloth-bnb-4bit"

    for stage_config in STAGES:
        merged_dir = run_stage(stage_config, base_model)

        # Next stage loads from this merged model
        base_model = merged_dir

        # Clean up previous stage's merged model (except last)
        if stage_config["stage"] < 3:
            prev = f"merged-stage{stage_config['stage'] - 1}" if stage_config["stage"] > 1 else None
            if prev and os.path.exists(prev):
                shutil.rmtree(prev)
                print(f"  🗑  Cleaned previous: {prev}")

    print("\n" + "=" * 60)
    print("  ALL STAGES COMPLETE")
    print("=" * 60)
    print(f"  Final model: merged-stage3")
    print(f"  Kaggle: {KAGGLE_USERNAME}/gpt-oss-120b-aimo3-sft-v3")
    print("=" * 60)

if WANDB_KEY:
    try:
        import wandb
        wandb.finish()
    except Exception:
        pass
