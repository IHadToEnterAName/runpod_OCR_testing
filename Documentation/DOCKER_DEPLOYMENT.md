# Complete Docker Deployment Guide

## 🎯 Overview

This setup runs **everything in Docker containers**:
- ✅ vLLM Vision Server (Qwen2.5-VL-3B)
- ✅ vLLM Reasoning Server (DeepSeek-R1-1.5B)
- ✅ RAG Application (Chainlit)
- ✅ ChromaDB Vector Database
- ✅ Redis Cache

**Result**: Run the entire system on any machine with just `docker compose up`!

## 📦 What You Get

### 1. **Dockerfile.complete**
- CUDA 12.6 base (Blackwell support)
- PyTorch nightly (CUDA 12.8)
- vLLM built from source with `sm_120` support
- All dependencies pre-installed
- Optimized for RTX 5090

### 2. **docker-compose.complete.yml**
- 5 containerized services
- GPU allocation configured
- Volume persistence
- Health checks
- Auto-restart policies

### 3. **docker-start.sh**
- Interactive control panel
- One-command deployment
- Status monitoring
- Log viewing
- Easy cleanup

## 🚀 Quick Start (5 Commands)

```bash
# 1. Check prerequisites
./docker-start.sh
# Select: 1) First-time setup

# 2. Wait for build (10-20 minutes first time)
# Wait for all services to start (~5 minutes)

# 3. Access application
open http://localhost:8000

# 4. Test APIs
curl http://localhost:8005/v1/models  # Reasoning
curl http://localhost:8006/v1/models  # Vision

# 5. Upload documents and start chatting!
```

## 📋 Prerequisites

### Hardware
- **GPU**: NVIDIA GPU with 16GB+ VRAM (tested on RTX 5090)
- **RAM**: 32GB+ system memory recommended
- **Storage**: 50GB+ free space (models + cache)

### Software
- **Docker**: 24.0+
- **Docker Compose**: V2
- **NVIDIA Container Toolkit**: Latest
- **Operating System**: Ubuntu 22.04+ or similar

### Install NVIDIA Container Toolkit

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verify
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

## 📁 Project Structure

```
./
├── Dockerfile.complete              # Main image definition
├── docker-compose.complete.yml      # Service orchestration
├── docker-start.sh                  # Control script
├── src_refactored/                  # Application code
│   ├── app.py
│   ├── config/
│   ├── rag/
│   ├── processing/
│   ├── storage/
│   └── utils/
├── data/                            # Document storage
│   ├── uploads/
│   ├── processed/
│   └── failed/
└── volumes/                         # Persistent data
    ├── huggingface/                 # Model cache
    ├── vllm_cache/                  # vLLM kernel cache
    └── models/                      # Additional models
```

## 🔧 Configuration

### GPU Allocation

**Dual GPU (Default - RTX 5090 x2)**:
```yaml
vllm_vision:
  device_ids: ['0']  # GPU 0, 0.3 utilization

vllm_reasoning:
  device_ids: ['1']  # GPU 1, 0.53 utilization

rag_app:
  device_ids: ['0']  # GPU 0 (embeddings)
```

**Single GPU**:
Change all `device_ids: ['0']` and reduce memory:
```yaml
vllm_vision:
  --gpu-memory-utilization 0.25

vllm_reasoning:
  --gpu-memory-utilization 0.45
```

### Memory Settings

Edit `docker-compose.complete.yml`:

```yaml
# Vision server
command: >
  vllm serve "Qwen/Qwen2.5-VL-3B-Instruct"
  --gpu-memory-utilization 0.3  # Adjust this

# Reasoning server  
command: >
  vllm serve "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
  --gpu-memory-utilization 0.53  # Adjust this
```

## 🎮 Using the Control Script

### First Time Setup
```bash
./docker-start.sh
# Select: 1) First-time setup (build + start)
```

This will:
1. Check prerequisites (Docker, GPU, etc.)
2. Create directory structure
3. Build Docker images (~10-20 minutes)
4. Start all services
5. Wait for initialization
6. Show service URLs

### Daily Usage
```bash
./docker-start.sh
# Select: 2) Start services
```

### View Logs
```bash
./docker-start.sh
# Select: 5) View logs
# Choose which service
```

### Check Status
```bash
./docker-start.sh
# Select: 6) Check status
```

### Stop Services
```bash
./docker-start.sh
# Select: 3) Stop services
```

## 🖥️ Manual Commands

### Build
```bash
docker compose -f docker-compose.complete.yml build
```

### Start
```bash
docker compose -f docker-compose.complete.yml up -d
```

### Stop
```bash
docker compose -f docker-compose.complete.yml down
```

### Logs
```bash
# All services
docker compose -f docker-compose.complete.yml logs -f

# Specific service
docker compose -f docker-compose.complete.yml logs -f rag_app
docker compose -f docker-compose.complete.yml logs -f vllm_vision
docker compose -f docker-compose.complete.yml logs -f vllm_reasoning
```

### Status
```bash
docker compose -f docker-compose.complete.yml ps
```

### Restart Service
```bash
docker compose -f docker-compose.complete.yml restart rag_app
```

## 🌐 Accessing Services

Once running:

| Service | URL | Purpose |
|---------|-----|---------|
| **RAG Application** | http://localhost:8000 | Chat interface |
| **Vision API** | http://localhost:8006/v1 | Image analysis |
| **Reasoning API** | http://localhost:8005/v1 | Text generation |
| **ChromaDB** | http://localhost:8003 | Vector database |
| **RedisInsight** | http://localhost:8001 | Cache UI |

### Test APIs
```bash
# Test vision model
curl http://localhost:8006/v1/models

# Test reasoning model
curl http://localhost:8005/v1/models

# Test ChromaDB
curl http://localhost:8003/api/v1/heartbeat
```

## 📊 Monitoring

### GPU Usage
```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Container GPU usage
docker stats
```

### Container Health
```bash
# Check health status
docker compose -f docker-compose.complete.yml ps

# Inspect specific container
docker inspect rag_vllm_vision | grep -A 10 Health
```

### Model Loading Progress
```bash
# Vision model loading
docker compose -f docker-compose.complete.yml logs -f vllm_vision

# Look for: "Finished loading model in X seconds"
```

## 🐛 Troubleshooting

### Build Fails

**Issue**: vLLM compilation errors
```bash
# Check Docker BuildKit
export DOCKER_BUILDKIT=1

# Increase build memory
docker compose -f docker-compose.complete.yml build --memory=8g
```

### Models Won't Load

**Issue**: OOM or CUDA errors
```bash
# Reduce GPU memory utilization
# Edit docker-compose.complete.yml:
--gpu-memory-utilization 0.2  # Vision
--gpu-memory-utilization 0.4  # Reasoning
```

### GPU Not Detected

**Issue**: `nvidia-smi` works but Docker can't see GPU
```bash
# Restart Docker
sudo systemctl restart docker

# Check runtime
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi

# Verify toolkit
nvidia-ctk --version
```

### Container Crashes

**Issue**: Container exits immediately
```bash
# Check logs
docker compose -f docker-compose.complete.yml logs vllm_vision

# Common issues:
# - Out of memory: Reduce gpu-memory-utilization
# - CUDA version mismatch: Check nvidia-smi driver version
# - Model download failed: Check internet connection
```

### Slow Model Downloads

**Issue**: First start takes forever
```bash
# Models download from HuggingFace (~20GB)
# Speed it up with HF_HUB_ENABLE_HF_TRANSFER=1 (already set)

# Or pre-download models:
docker run --rm -v $(pwd)/volumes/huggingface:/cache \
  -e HF_HOME=/cache \
  huggingface/transformers-pytorch-gpu \
  python -c "from transformers import AutoModel; \
  AutoModel.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct')"
```

### Port Already in Use

**Issue**: Port 8005/8006/7860 already bound
```bash
# Check what's using the port
sudo lsof -i :8005

# Kill the process or change ports in docker-compose.complete.yml
ports:
  - "8015:8005"  # External:Internal
```

## 🔄 Updates and Maintenance

### Update Application Code
```bash
# Code is mounted as volume, changes reflect immediately
# Edit files in src_refactored/
# Restart app container
docker compose -f docker-compose.complete.yml restart rag_app
```

### Update Models
```bash
# Models cached in volumes/huggingface/
# To force re-download:
rm -rf volumes/huggingface/*
docker compose -f docker-compose.complete.yml up -d
```

### Update Docker Images
```bash
# Rebuild with latest code/dependencies
docker compose -f docker-compose.complete.yml build --no-cache
docker compose -f docker-compose.complete.yml up -d
```

### Clean Up Old Images
```bash
# Remove unused images
docker image prune -a

# Remove unused volumes
docker volume prune
```

## 💾 Backup and Restore

### Backup
```bash
# Backup volumes
tar -czf rag-backup-$(date +%Y%m%d).tar.gz \
  volumes/ \
  data/

# Backup just models (large)
tar -czf models-backup.tar.gz volumes/huggingface/
```

### Restore
```bash
# Extract backup
tar -xzf rag-backup-YYYYMMDD.tar.gz

# Start services
docker compose -f docker-compose.complete.yml up -d
```

## 🚀 Production Deployment

### Environment Variables
Create `.env` file:
```bash
# GPU settings
GPU_MEMORY_VISION=0.3
GPU_MEMORY_REASONING=0.53

# Model settings
VISION_MODEL=Qwen/Qwen2.5-VL-3B-Instruct
REASONING_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

# Network
EXTERNAL_PORT_APP=7860
EXTERNAL_PORT_VISION=8006
EXTERNAL_PORT_REASONING=8005
```

### HTTPS/SSL
Use reverse proxy (Nginx/Traefik):
```yaml
# Add to docker-compose.complete.yml
nginx:
  image: nginx:alpine
  ports:
    - "443:443"
  volumes:
    - ./nginx.conf:/etc/nginx/nginx.conf
    - ./ssl:/etc/nginx/ssl
```

### Resource Limits
```yaml
# Add to each service in docker-compose.complete.yml
deploy:
  resources:
    limits:
      memory: 8G
      cpus: '4.0'
```

## 📈 Performance Tuning

### Optimize vLLM
```yaml
# Add to command in docker-compose.complete.yml
--max-num-seqs 256           # Batch size
--max-num-batched-tokens 8192  # Token batching
--enable-chunked-prefill     # Better throughput
```

### Optimize ChromaDB
```yaml
# Add environment variables
environment:
  - CHROMA_SERVER_AUTHN_CREDENTIALS_FILE=/chroma/auth.yaml
  - CHROMA_SERVER_AUTHN_PROVIDER=chromadb.auth.token.TokenAuthenticationServerProvider
```

## 🎯 Success Checklist

After deployment, verify:

- [ ] All 5 containers running (`docker compose ps`)
- [ ] No errors in logs (`docker compose logs`)
- [ ] GPU visible in containers (`nvidia-smi` inside container)
- [ ] Vision API responds (`curl localhost:8006/v1/models`)
- [ ] Reasoning API responds (`curl localhost:8005/v1/models`)
- [ ] ChromaDB healthy (`curl localhost:8003/api/v1/heartbeat`)
- [ ] Web interface loads (`open http://localhost:8000`)
- [ ] Can upload documents
- [ ] Can ask questions
- [ ] Models generate responses

## 📞 Support

### Logs to Check
1. **Build issues**: Docker build output
2. **Startup issues**: `docker compose logs`
3. **Runtime issues**: Individual service logs
4. **GPU issues**: `nvidia-smi` and container logs

### Common Solutions
1. **Restart Docker**: `sudo systemctl restart docker`
2. **Restart services**: `docker compose restart`
3. **Rebuild images**: `docker compose build --no-cache`
4. **Check GPU**: `nvidia-smi`
5. **Check ports**: `netstat -tlnp`

---

## 🎉 You're All Set!

Your complete RAG system is now:
- ✅ Fully containerized
- ✅ Portable (run anywhere with Docker + GPU)
- ✅ Reproducible (same environment everywhere)
- ✅ Production-ready (health checks, restart policies)
- ✅ Easy to manage (one script controls everything)

**Run it**: `./docker-start.sh` → Select "1" → Wait → Chat!
