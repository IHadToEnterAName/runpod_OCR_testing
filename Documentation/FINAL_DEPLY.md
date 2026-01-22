This is a clean, professionally formatted version of your instructions. I have used a clear hierarchy with bolding and code blocks to ensure it is easy to read and copy directly into a Google Doc.

---

# **RAG System: Deployment & Execution Guide**

This document outlines the steps required to run the RAG system using either a **Native (Direct)** installation or **Docker**.

---

## **Method 1: Running WITHOUT Docker (Native)**

### **Prerequisites**

* **Python:** 3.10+
* **Hardware:** NVIDIA GPU with CUDA 12.x
* **VRAM:** ~24GB+ (Single GPU setup)
* **Redis:** Redis server installed locally

### **Step 1: Start vLLM Servers (2 Terminals)**

**Terminal 1 — Vision Model (OCR)**

* **Model:** Qwen2.5-VL-3B-Instruct
* **Port:** 8006

```bash
cd /home/nasih/runpod_testing/OCR_LLM/vllm_inference_llama3
source /workspace/venv/bin/activate

VLLM_USE_V1=0 vllm serve "Qwen/Qwen2.5-VL-3B-Instruct" \
    --port 8006 \
    --gpu-memory-utilization 0.3 \
    --max-model-len 4096 \
    --limit-mm-per-prompt '{"image":12}' \
    --enforce-eager \
    --trust-remote-code

```

**Terminal 2 — Reasoning Model**

* **Model:** DeepSeek-R1-Distill-Qwen-7B
* **Port:** 8005

```bash
source /workspace/venv/bin/activate

VLLM_USE_V1=0 vllm serve "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
    --port 8005 \
    --gpu-memory-utilization 0.54 \
    --max-model-len 16384 \
    --enforce-eager \
    --enable-prefix-caching

```

### **Step 2: Start Redis (Terminal 3)**

```bash
redis-server

```

### **Step 3: Install Dependencies**

```bash
cd /home/nasih/runpod_testing/OCR_LLM/vllm_inference_llama3
pip install -r Scripts/requirements.txt

```

### **Step 4: Run the Application (Terminal 4)**

```bash
cd /home/nasih/runpod_testing/OCR_LLM/vllm_inference_llama3

# Set environment
export PYTHONPATH=$PWD/src
export HF_HOME=/workspace/huggingface
export CUDA_VISIBLE_DEVICES=0

# Run Chainlit app
python -m chainlit run src/app.py --host 0.0.0.0 --port 8000

```

---

## **Method 2: Running WITH Docker**

### **Option A: Interactive Script (Easiest)**

```bash
cd /home/nasih/runpod_testing/OCR_LLM/vllm_inference_llama3/Docker
chmod +x docker-start.sh
./docker-start.sh

```

> *Select **Option 1** for first-time setup (Build + Start).*

### **Option B: Manual Docker Commands**

*Note: Ensure vLLM servers are already running on the host (see Method 1).*

```bash
cd /home/nasih/runpod_testing/OCR_LLM/vllm_inference_llama3/Docker

# Build and start services
docker compose up -d --build

# Check status & logs
docker compose ps
docker compose logs -f rag_app

```

### **Option C: Fully Containerized (vLLM inside Docker)**

```bash
cd /home/nasih/runpod_testing/OCR_LLM/vllm_inference_llama3/Docker

# Uses complete compose file (includes vLLM containers)
docker compose -f docker-compose.complete.yml up -d --build

```

---

## **Reference Tables**

### **Service Access URLs**

| Service | URL | Credentials |
| --- | --- | --- |
| **RAG Application** | `http://localhost:8000` | - |
| **ChromaDB** | `http://localhost:8003` | - |
| **Redis Insight** | `http://localhost:8001` | - |
| **Airflow UI** | `http://localhost:8080` | `admin / admin` |

### **Common Docker Commands**

| Action | Command |
| --- | --- |
| **Stop Services** | `docker compose down` |
| **View App Logs** | `docker compose logs -f rag_app` |
| **View Redis Logs** | `docker compose logs -f redis` |
| **Full Cleanup** | `docker compose down -v` (removes volumes) |

---

**Would you like me to add a troubleshooting section for common GPU/VRAM errors in case she runs into issues during the test?**