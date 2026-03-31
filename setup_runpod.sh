#!/bin/bash
# ─────────────────────────────────────────────────────────────
# RunPod Setup Script for Qwen3.5-122B-A10B Fine-Tuning
# ─────────────────────────────────────────────────────────────
#
# Run this ONCE after spinning up a 4x A100 80GB SXM pod.
#
# Pod setup:
#   1. Go to RunPod > Pods > Deploy
#   2. Select: 4x A100 80GB SXM
#   3. Template: RunPod PyTorch 2.4 (or latest)
#   4. Volume: 500GB network storage (for model weights + checkpoints)
#   5. Deploy, then SSH in or open terminal
#   6. Run: bash setup_runpod.sh
#
# ─────────────────────────────────────────────────────────────

set -e

echo "=================================================="
echo "  RunPod Fine-Tuning Setup"
echo "=================================================="

# ─── 1. System check ──────────────────────────────────────
echo ""
echo "[1/5] Checking GPUs..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Found $GPU_COUNT GPU(s)"

if [ "$GPU_COUNT" -lt 4 ]; then
    echo "WARNING: Expected 4 GPUs but found $GPU_COUNT. Training may not fit in VRAM."
    echo "Continue anyway? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then exit 1; fi
fi

# ─── 2. Install Unsloth + dependencies ───────────────────
echo ""
echo "[2/5] Installing Unsloth and dependencies..."
pip install --upgrade pip
pip install "unsloth[cu124-ampere-torch250] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes xformers
pip install datasets wandb

echo "Unsloth installed successfully."

# ─── 3. Download model weights ──────────────────────────
echo ""
echo "[3/5] Downloading Qwen3.5-122B-A10B base weights..."
echo "This will download ~240GB. Make sure you have enough storage."

# Pre-download to HuggingFace cache (avoids timeout during training)
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'Qwen/Qwen3.5-122B-A10B',
    local_dir='/workspace/models/Qwen3.5-122B-A10B',
    resume_download=True,
)
print('Download complete.')
"

# ─── 4. Upload training data ────────────────────────────
echo ""
echo "[4/5] Checking for training data..."
if [ -f "/workspace/maya_training_data.jsonl" ]; then
    echo "Training data found at /workspace/maya_training_data.jsonl"
else
    echo "Training data not found."
    echo ""
    echo "Upload it with one of these methods:"
    echo "  Option A: scp maya_training_data.jsonl root@<pod-ip>:/workspace/"
    echo "  Option B: Use RunPod file manager in the web UI"
    echo "  Option C: wget from a URL you host temporarily"
    echo ""
    echo "Then re-run this script or proceed to training directly."
fi

# ─── 5. Verify setup ────────────────────────────────────
echo ""
echo "[5/5] Verifying setup..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    mem = torch.cuda.get_device_properties(i).total_mem / 1024**3
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({mem:.0f}GB)')

import unsloth
print(f'Unsloth: OK')

from trl import SFTTrainer
print(f'TRL: OK')

print()
print('All checks passed.')
"

echo ""
echo "=================================================="
echo "  Setup Complete"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. Upload maya_training_data.jsonl to /workspace/"
echo "  2. Upload finetune.py to /workspace/"
echo "  3. Dry run:  python3 finetune.py --dry-run"
echo "  4. Train:    python3 finetune.py --merge"
echo "  5. Export:   python3 finetune.py --gguf  (if needed)"
echo ""
echo "Estimated training time: 2-4 hours for 250 examples x 3 epochs"
echo "Estimated cost: ~\$15-25 at \$5.56/hr"
echo ""
