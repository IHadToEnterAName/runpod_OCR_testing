# Installation Guide for Blackwell GPU Setup

## The Problem

You're encountering an **ABI mismatch** error:
```
ImportError: undefined symbol: _ZN3c108ListType3getE...
```

This happens because vLLM was compiled against a different PyTorch version than what's currently installed.

**Root Cause:**
- Your system has PyTorch 2.5.1
- vLLM was built expecting PyTorch 2.9.1 (or different ABI)
- Blackwell GPU (SM 12.0) requires special compilation flags

---

## Solution: Clean Installation

### Step 1: Clean Your Environment

```bash
cd /workspace

# Deactivate and remove old venv
deactivate 2>/dev/null || true
rm -rf venv/

# Create fresh virtual environment
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install PyTorch with CUDA 12.8 Support

```bash
# Install PyTorch that supports Blackwell
pip install torch>=2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Step 3: Build vLLM from Source

```bash
cd /workspace

# Clean any existing vLLM
if [ -d "vllm" ]; then
    cd vllm
    rm -rf build/ .deps/ dist/ *.egg-info
    find . -name "*.so" -delete
    git pull  # Get latest
else
    git clone https://github.com/vllm-project/vllm.git
    cd vllm
fi

# Set Blackwell-specific environment variables
export TORCH_CUDA_ARCH_LIST="9.0;10.0;12.0"  # Hopper, Ada, Blackwell
export VLLM_TARGET_DEVICE="cuda"
export CUDA_HOME="/usr/local/cuda"
export MAX_JOBS=8
export VLLM_INSTALL_PUNICA_KERNELS=0
export VLLM_USE_V1=1

# Build vLLM (takes 10-20 minutes)
pip install -e . --no-build-isolation -v
```

### Step 4: Install RAG Application Dependencies

```bash
cd /home/nasih/runpod_testing/OCR_LLM/vllm_inference_llama3

# Install remaining dependencies
pip install -r Scripts/requirements.txt
```

### Step 5: Verify Everything Works

```bash
# Test vLLM import
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"

# Test PyTorch GPU
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Compute capability: {torch.cuda.get_device_capability(0)}')
"
```

---

## Quick Fix for Current Installation

If you don't want to rebuild everything, try this quick fix:

```bash
source /workspace/venv/bin/activate

# Upgrade PyTorch to match what vLLM expects
pip install --upgrade torch>=2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Rebuild vLLM C++ extensions
cd /workspace/vllm
rm -rf build/ .deps/
export TORCH_CUDA_ARCH_LIST="9.0;10.0;12.0"
export VLLM_INSTALL_PUNICA_KERNELS=0
pip install -e . --no-build-isolation --force-reinstall --no-deps
```

---

## Start vLLM Servers

Once everything is installed:

```bash
# Terminal 1 - Vision Model
source /workspace/venv/bin/activate
vllm serve Qwen/Qwen2.5-VL-3B-Instruct \
    --port 8006 \
    --gpu-memory-utilization 0.3 \
    --max-model-len 4096 \
    --limit-mm-per-prompt '{"image":12}' \
    --enforce-eager \
    --trust-remote-code

# Terminal 2 - Reasoning Model
source /workspace/venv/bin/activate
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --port 8005 \
    --gpu-memory-utilization 0.54 \
    --max-model-len 16384 \
    --enforce-eager \
    --enable-prefix-caching
```

---

## Common Issues

### Issue 1: "Unknown CUDA Architecture Name 12.0"
**Solution:** Use numeric compute capability instead:
```bash
export TORCH_CUDA_ARCH_LIST="12.0"  # Numeric, not "Blackwell"
```

### Issue 2: "PyTorch version mismatch"
**Solution:** Ensure vLLM is built against the installed PyTorch:
```bash
pip list | grep torch
cd /workspace/vllm && rm -rf build/ && pip install -e . --no-build-isolation
```

### Issue 3: "No compatible archs found"
**Solution:** This is OK for some kernels. vLLM will use fallback implementations.

### Issue 4: Out of Memory during compilation
**Solution:** Limit parallel jobs:
```bash
export MAX_JOBS=4
```

---

## Verify Installation Checklist

- [ ] PyTorch 2.6.0+ installed
- [ ] `torch.cuda.is_available()` returns `True`
- [ ] `import vllm` works without errors
- [ ] GPU compute capability shows `(12, 0)` for Blackwell
- [ ] vLLM servers start without import errors
- [ ] RAG app dependencies installed

---

## What Changed in requirements.txt

The updated requirements.txt:
1. **Removed pinned PyTorch versions** - PyTorch is now installed separately with CUDA support
2. **Uses flexible version constraints** (`>=`) instead of exact pins
3. **Added installation instructions** at the top
4. **Commented out optional dependencies** (Airflow, PostgreSQL)
5. **Compatible with Blackwell GPU** requirements

This prevents version conflicts and ensures proper CUDA support.
