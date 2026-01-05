
---

# Corporate Document Assistant: RTX 5090 (vLLM + RAG + Vision)

This repository contains a high-performance AI assistant optimized for an **RTX 5090 (32GB VRAM)**. It utilizes dual **vLLM** servers for reasoning and vision, alongside **SentenceTransformers** for local RAG, designed to process massive corporate datasets with ultra-low latency.

## 🚀 Optimized Model Stack

* **Reasoning Model:** `DeepSeek-R1-Distill-Qwen-1.5B` (Running on Port 8005)
* **Vision Model:** `Qwen2.5-VL-3B-Instruct` (Running on Port 8006)
* **Embedding Model:** `nomic-ai/nomic-embed-text-v1.5` (Local GPU-accelerated)

---

## 🛠️ Installation & Setup

### 1. Environment Configuration

Ensure your Python environment is ready and your workspace (400TB volume) is targeted for model storage.

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Set model storage to the large workspace volume
export HF_HOME="/workspace/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=1

```

### 2. Dependency Installation

Install the Blackwell-optimized stack, including vision and token-trimming libraries.

```bash
pip install torchvision hf_transfer tiktoken sentence-transformers uvloop chainlit openai

```

---

## 🛰️ Launching the Inference Servers

Because the RTX 5090 is on the Blackwell architecture, we use specific flags to ensure stability and prevent Out-of-Memory (OOM) errors during dual-model execution.

### Server 1: Vision-Language Model (Port 8006)

This handles OCR and image analysis. **Launch this first.**

```bash
VLLM_USE_V1=0 vllm serve "Qwen/Qwen2.5-VL-3B-Instruct" \
    --port 8006 \
    --gpu-memory-utilization 0.45 \
    --max-model-len 8192 \
    --limit-mm-per-prompt '{"image":12}' \
    --enforce-eager \
    --trust-remote-code

```

### Server 2: Reasoning Model (Port 8005)

This handles document analysis and complex logic.

```bash
VLLM_USE_V1=0 vllm serve "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
    --port 8005 \
    --gpu-memory-utilization 0.30 \
    --max-model-len 8192 \
    --enforce-eager \
    --enable-prefix-caching

```

---

## 🖥️ Running the Application

### Deploying Code (Rsync)

If working from a local machine, sync your project to the RunPod instance (replace with your current port/IP):

```bash
rsync -avz --exclude 'venv' --exclude 'huggingface' -e "ssh -p 53834 -i ~/.ssh/id_ed25519" ./project_folder root@157.157.221.29:/workspace/

```

### Starting the Chat UI

Launch the Chainlit interface once both vLLM servers show `Uvicorn running`.

```bash
chainlit run chat.py --host 0.0.0.0 --port 8000

```

---

## 📈 System Health & Management

### Port Cleanup

If you encounter "Address already in use" errors:

```bash
pkill -9 python
pkill -9 vllm
rm -rf /tmp/vllm_*

```

### Resource Monitoring

Monitor your 32GB VRAM allocation across the two vLLM processes:

```bash
watch -n 1 nvidia-smi

```

## 📂 Project Structure

```text
├── chat.py             # Logic for Port 8005/8006 routing & RAG
├── requirements.txt    # Updated with tiktoken and torchvision
├── venv/               # Local virtual environment
└── huggingface/        # 400TB storage for model weights

```

0. Storage & Cache Initialization
Run these commands first to prepare the persistent volume. This prevents the small root partition from filling up and ensures you don't have to re-download models if the pod restarts.

Bash

# Create the directory for HuggingFace model weights
mkdir -p /workspace/huggingface

# Create the directory for vLLM's internal compilation & kernel cache
mkdir -p /workspace/vllm_cache

# Create the directory for your uploaded documents and processed images
mkdir -p /workspace/data
1. Linking Directories to the System
We use environment variables to tell the AI libraries to use these specific folders on the large disk.

Bash

# Point HuggingFace to the 400TB volume
export HF_HOME="/workspace/huggingface"

# Point vLLM's compiler to the persistent cache folder
# This speeds up subsequent launches of the Vision model
export VLLM_CACHE_ROOT="/workspace/vllm_cache"

---

