#!/bin/bash
# =============================================================================
# Quick Fix Script for vLLM ABI Mismatch on Blackwell GPU
# =============================================================================

set -e

echo "=================================================="
echo "vLLM Quick Fix for Blackwell GPU"
echo "=================================================="

# Check virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ Error: Please activate your virtual environment first"
    echo "   Run: source venv/bin/activate"
    exit 1
fi

echo ""
echo "Current PyTorch version:"
python -c "import torch; print(f'  PyTorch: {torch.__version__}')" 2>/dev/null || echo "  PyTorch not installed"

# Step 1: Upgrade PyTorch
echo ""
echo "Step 1: Upgrading PyTorch to 2.6.0+ with CUDA 12.8 support..."
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

echo ""
echo "New PyTorch version:"
python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')"

# Step 2: Rebuild vLLM
if [ ! -d "/workspace/vllm" ]; then
    echo ""
    echo "❌ Error: vLLM source not found at /workspace/vllm"
    echo "   Please clone it first: git clone https://github.com/vllm-project/vllm.git /workspace/vllm"
    exit 1
fi

echo ""
echo "Step 2: Rebuilding vLLM with correct PyTorch ABI..."
cd /workspace/vllm

# Clean build artifacts
rm -rf build/ .deps/ dist/ *.egg-info
find . -name "*.so" -type f -delete 2>/dev/null || true

# Set environment variables for Blackwell
export TORCH_CUDA_ARCH_LIST="9.0;10.0;12.0"
export VLLM_TARGET_DEVICE="cuda"
export CUDA_HOME="/usr/local/cuda"
export MAX_JOBS=8
export VLLM_INSTALL_PUNICA_KERNELS=0
export VLLM_USE_V1=1

echo "  Building vLLM (this takes 10-20 minutes)..."
pip install -e . --no-build-isolation --force-reinstall --no-deps -v

# Step 3: Verify
echo ""
echo "Step 3: Verifying installation..."
python -c "
import torch
import vllm
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ vLLM: {vllm.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
    cap = torch.cuda.get_device_capability(0)
    print(f'✅ Compute capability: {cap[0]}.{cap[1]}')
" || {
    echo ""
    echo "❌ Verification failed"
    exit 1
}

echo ""
echo "=================================================="
echo "✅ Fix complete!"
echo "=================================================="
echo ""
echo "You can now start vLLM servers:"
echo ""
echo "vllm serve Qwen/Qwen2.5-VL-3B-Instruct --port 8006 --gpu-memory-utilization 0.3"
echo ""
