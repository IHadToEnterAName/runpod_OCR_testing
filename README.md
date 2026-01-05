# 🚀 RTX 5090 Corporate AI Assistant: Master Documentation

This guide provides the full consolidated workflow for managing your dual-model (Reasoning + Vision) system on an **RTX 5090**.

---

## 1. Local Machine: Connection & Deployment

To connect to your Pod and transfer files, follow the official configuration steps.

### A. SSH Setup

For instructions on generating SSH keys, adding them to your RunPod account, and establishing a secure connection, refer to the official documentation:
👉 **[RunPod SSH Configuration Guide](https://docs.runpod.io/pods/configuration/use-ssh)**

### B. Deployment Command (Rsync)

Edit and Run this command from your **local terminal** to synchronize your project folder to the server. Edit according the address, ports, home and destintation folder.

```bash
# Sync project while skipping large model/env folders to save time
rsync -avz --exclude 'venv' --exclude 'huggingface' --exclude '.git' \
-e "ssh -p 53834 -i ~/.ssh/id_ed25519" \
/home/nasih/runpod_testing/OCR_LLM/vllm_inference_llama3/ \
root@157.157.221.29:/workspace/vllm_inference_llama3

```

---

## 2. Remote Server: Storage & Environment Prep

Run these commands inside your RunPod terminal once to prepare the persistent 400TB volume.

```bash
# 1. Initialize persistent storage paths
mkdir -p /workspace/huggingface
mkdir -p /workspace/vllm_cache
mkdir -p /workspace/data

# 2. Set persistent environment variables
export HF_HOME="/workspace/huggingface"
export VLLM_CACHE_ROOT="/workspace/vllm_cache"
export HF_HUB_ENABLE_HF_TRANSFER=1

# 3. Virtual Environment Setup
python3 -m venv /workspace/venv
source /workspace/venv/bin/activate
pip install --upgrade pip setuptools wheel

```

---

## 3. Remote Server: Installation (Blackwell Optimized)

The RTX 5090 requires specific builds to unlock the Blackwell architecture (`sm_120`).

```bash
# 1. Install Nightly PyTorch (CUDA 12.8+ support)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# 2. Build vLLM from Source
cd /workspace
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -r requirements-build.txt
export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS=12 
pip install -e . --no-build-isolation

# 3. Install remaining UI & Processing tools
pip install hf_transfer tiktoken sentence-transformers uvloop chainlit openai

```

---

## 4. Launching the Inference Servers

Open two separate terminals (or use `screen`/`tmux`) to run these backends.

### Terminal 1: Vision Model (Port 8006)

**Launch this first.** Handles OCR and image analysis.

```bash
source /workspace/venv/bin/activate
VLLM_USE_V1=0 vllm serve "Qwen/Qwen2.5-VL-3B-Instruct" \
    --port 8006 \
    --gpu-memory-utilization 0.35 \
    --max-model-len 8192 \
    --limit-mm-per-prompt '{"image":12}' \
    --enforce-eager \
    --trust-remote-code

```

### Terminal 2: Reasoning Model (Port 8005)

Handles deep document analysis and complex logic.

```bash
source /workspace/venv/bin/activate
VLLM_USE_V1=0 vllm serve "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
    --port 8005 \
    --gpu-memory-utilization 0.30 \
    --max-model-len 32768 \
    --enforce-eager \
    --enable-prefix-caching

```

---

## 5. Running the Chat Interface

Once both servers show `Uvicorn running`, launch your application UI.

```bash
source /workspace/venv/bin/activate
cd /workspace/vllm_inference_llama3
chainlit run chat.py --host 0.0.0.0 --port 8000

```

---

## 📈 System Maintenance

* **VRAM Monitor:** `watch -n 1 nvidia-smi`
* **Hard Reset (Clean all AI processes):**
```bash
pkill -9 python && pkill -9 vllm && rm -rf /tmp/vllm_*

```


* **Wipe Kernels Cache:** `rm -rf /workspace/vllm_cache/*`

---
