#!/bin/bash
# =============================================================================
# Run this on BOTH nodes before training
# Sets up SSH, env, and verifies GPU access
# =============================================================================
set -e

echo "=== Node Setup for RAFT Training ==="
echo "Hostname: $(hostname)"
echo "IP: $(hostname -I | awk '{print $1}')"

# --- 1. Unset conflicting env vars ---
echo "unset PYTORCH_CUDA_ALLOC_CONF" >> ~/.bashrc
echo "unset PYTORCH_ALLOC_CONF" >> ~/.bashrc
echo 'export HF_HOME="/data/hf_cache"' >> ~/.bashrc
echo 'export WANDB_PROJECT="aimo3-raft"' >> ~/.bashrc
source ~/.bashrc

# --- 2. Verify GPUs ---
echo ""
echo "=== GPU Check ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "Total GPUs: ${NUM_GPUS}"

if [ "$NUM_GPUS" -ne 8 ]; then
    echo "WARNING: Expected 8 GPUs, found ${NUM_GPUS}"
fi

# --- 3. Verify Python packages ---
echo ""
echo "=== Package Check ==="
python3.12 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
python3.12 -c "import deepspeed; print(f'DeepSpeed: {deepspeed.__version__}')"
python3.12 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python3.12 -c "import datasets; print(f'Datasets: {datasets.__version__}')"
python3.12 -c "import wandb; print(f'WandB: {wandb.__version__}')"

# --- 4. Verify model exists ---
echo ""
echo "=== Model Check ==="
if [ -d "/data/models/gpt-oss-120b" ]; then
    MODEL_SIZE=$(du -sh /data/models/gpt-oss-120b | awk '{print $1}')
    echo "Model found: /data/models/gpt-oss-120b (${MODEL_SIZE})"
else
    echo "ERROR: Model not found at /data/models/gpt-oss-120b"
    echo "Download it with:"
    echo '  python3.12 -c "from huggingface_hub import snapshot_download; snapshot_download(\"unsloth/gpt-oss-120b\", local_dir=\"/data/models/gpt-oss-120b\", max_workers=8)"'
    exit 1
fi

# --- 5. Verify NCCL ---
echo ""
echo "=== NCCL Check ==="
python3.12 -c "
import torch.distributed as dist
import os
# Just check NCCL is available
print(f'NCCL available: {dist.is_nccl_available()}')
print(f'NCCL_IB_DISABLE: {os.environ.get(\"NCCL_IB_DISABLE\", \"not set\")}')
"

# --- 6. Verify SSH connectivity (run on head node only) ---
echo ""
echo "=== SSH Check ==="
echo "If this is the HEAD node, verify you can SSH to the worker:"
echo "  ssh 172.31.106.192 hostname"
echo "If this is the WORKER node, verify you can SSH to the head:"
echo "  ssh 172.31.110.230 hostname"

echo ""
echo "=== Setup Complete ==="
