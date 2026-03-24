#!/bin/bash
"""
Setup script for AWS 8× H100 instance
Run this first after SSH-ing in.
"""

echo "================================"
echo "Setting up 8× H100 for AIMO3"
echo "================================"

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers trl datasets accelerate deepspeed
pip install flash-attn --no-build-isolation
pip install wandb kagglehub python-dotenv

# Verify GPUs
echo ""
echo "GPUs detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""
python3 -c "import torch; print(f'PyTorch sees {torch.cuda.device_count()} GPUs')"

# Verify DeepSpeed
echo ""
ds_report

# Create .env if not exists
if [ ! -f .env ]; then
    echo "Creating .env template..."
    cat > .env << 'EOF'
HF_TOKEN=hf_xxx
KAGGLE_USERNAME=xxx
KAGGLE_API_TOKEN=xxx
WANDB_API_KEY=xxx
EOF
    echo "Edit .env with your tokens!"
fi

echo ""
echo "================================"
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Edit .env with your tokens"
echo "  2. Copy your dataset: scp data.parquet user@host:~/aimo3/data/"
echo "  3. Run: accelerate launch --config_file accelerate_config.yaml train_full_8xh100.py"
echo "================================"
