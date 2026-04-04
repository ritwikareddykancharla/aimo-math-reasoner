#!/bin/bash
# =============================================================================
# RAFT Two-Stage Training for GPT-OSS-120B
# Run this ONLY on Node 1 (head node): 172.31.110.230
# =============================================================================
set -e

# --- Config ---
export REPO_DIR="/data/aimo-math-reasoner"
export MODEL_DIR="/data/models/gpt-oss-120b"
export DATA_DIR="${REPO_DIR}/data"
export SCRIPT_DIR="${REPO_DIR}/training/sft"
export CHECKPOINT_DIR="/data/checkpoints"
export HF_HOME="/data/hf_cache"
export HF_DATASETS_CACHE="/data/hf_cache/datasets"
export WANDB_PROJECT="aimo3-raft"

# Unset conflicting env vars (critical for MoE)
unset PYTORCH_CUDA_ALLOC_CONF
unset PYTORCH_ALLOC_CONF

# Node IPs (update if changed)
HEAD_IP="172.31.110.230"
WORKER_IP="172.31.106.192"
NUM_NODES=2
NUM_GPUS_PER_NODE=8
TOTAL_GPUS=$((NUM_NODES * NUM_GPUS_PER_NODE))

echo "============================================"
echo "RAFT Training Pipeline for AIMO3"
echo "  Head node:   ${HEAD_IP}"
echo "  Worker node: ${WORKER_IP}"
echo "  Total GPUs:  ${TOTAL_GPUS}"
echo "============================================"

# =============================================================================
# Step 0: Prepare data (runs locally, no multi-node needed)
# =============================================================================
echo ""
echo "[Step 0] Preparing RAFT training data..."
if [ ! -f "${DATA_DIR}/raft_stage1_all_hard_tir.parquet" ]; then
    python3.12 ${REPO_DIR}/training/data/prepare_raft_data.py \
        --output_dir "${DATA_DIR}" \
        --difficulty_threshold 0.5 \
        --cache_dir "${HF_HOME}"
    echo "[Step 0] Data preparation complete."
else
    echo "[Step 0] Data already exists, skipping."
fi

# =============================================================================
# Step 1: Stage 1 - Broad SFT on all hard problems with TIR
# =============================================================================
echo ""
echo "============================================"
echo "[Step 1] Stage 1: Broad SFT (all hard + TIR)"
echo "============================================"

STAGE1_OUTPUT="${CHECKPOINT_DIR}/raft_stage1"

if [ ! -f "${STAGE1_OUTPUT}/config.json" ]; then
    deepspeed \
        --hostfile "${SCRIPT_DIR}/hostfile" \
        --master_addr "${HEAD_IP}" \
        --master_port 29500 \
        --num_nodes ${NUM_NODES} \
        --num_gpus ${NUM_GPUS_PER_NODE} \
        ${SCRIPT_DIR}/train_raft.py \
            --stage 1 \
            --model_path "${MODEL_DIR}" \
            --data_dir "${DATA_DIR}" \
            --output_dir "${STAGE1_OUTPUT}" \
            --max_seq_len 4096 \
            --per_device_batch_size 1 \
            --gradient_accumulation_steps 2 \
            --learning_rate 2e-5 \
            --num_epochs 1 \
            --warmup_ratio 0.03 \
            --save_steps 200 \
            --logging_steps 10 \
            --deepspeed "${SCRIPT_DIR}/ds_config_z3_offload.json"

    echo "[Step 1] Stage 1 complete. Checkpoint saved to ${STAGE1_OUTPUT}"
else
    echo "[Step 1] Stage 1 checkpoint exists, skipping."
fi

# =============================================================================
# Step 2: Stage 2 - Competition-focused SFT on AoPS hard problems
# =============================================================================
echo ""
echo "============================================"
echo "[Step 2] Stage 2: Competition SFT (AoPS hard + TIR)"
echo "============================================"

STAGE2_OUTPUT="${CHECKPOINT_DIR}/raft_stage2"

deepspeed \
    --hostfile "${SCRIPT_DIR}/hostfile" \
    --master_addr "${HEAD_IP}" \
    --master_port 29500 \
    --num_nodes ${NUM_NODES} \
    --num_gpus ${NUM_GPUS_PER_NODE} \
    ${SCRIPT_DIR}/train_raft.py \
        --stage 2 \
        --model_path "${STAGE1_OUTPUT}" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${STAGE2_OUTPUT}" \
        --max_seq_len 4096 \
        --per_device_batch_size 1 \
        --gradient_accumulation_steps 2 \
        --learning_rate 5e-6 \
        --num_epochs 1 \
        --warmup_ratio 0.05 \
        --save_steps 100 \
        --logging_steps 5 \
        --deepspeed "${SCRIPT_DIR}/ds_config_z3_offload.json"

echo "[Step 2] Stage 2 complete. Final model saved to ${STAGE2_OUTPUT}"

# =============================================================================
# Done
# =============================================================================
echo ""
echo "============================================"
echo "RAFT Training Complete!"
echo "  Stage 1 checkpoint: ${STAGE1_OUTPUT}"
echo "  Stage 2 checkpoint: ${STAGE2_OUTPUT}"
echo ""
echo "Next steps:"
echo "  1. Eval on AMC/AIME validation sets"
echo "  2. Sweep temperature [0.6, 0.7, 0.8, 0.9, 1.0]"
echo "  3. Use majority voting with 8-16 samples"
echo "============================================"
