# How to Run - Quick Reference

## 🚀 First Time Setup (5 minutes)

### 1. Prerequisites Check
```bash
# Verify Docker and NVIDIA runtime
docker --version
docker compose version
nvidia-smi
```

### 2. Environment Configuration
```bash
# Copy and edit environment file
cp .env.example .env
nano .env

# Key settings to update:
# - GPU memory allocation (VISION_GPU_MEMORY_UTILIZATION, REASONING_GPU_MEMORY_UTILIZATION)
# - Cache parameters (CACHE_SIMILARITY_THRESHOLD, CACHE_TTL_SECONDS)
# - Security keys (change defaults for production!)
```

### 3. Start vLLM Servers on Host

**Terminal 1 - Vision Model (GPU 0)**:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --port 8006 \
  --gpu-memory-utilization 0.8 \
  --tensor-parallel-size 1 \
  --device cuda:0
```

**Terminal 2 - Reasoning Model (GPU 1)**:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --port 8005 \
  --gpu-memory-utilization 0.85 \
  --tensor-parallel-size 1 \
  --device cuda:1
```

### 4. Launch the Stack
```bash
# Use the convenience script
./start.sh

# Or manually
docker compose up -d
docker compose logs -f rag_app
```

### 5. Access Services

Once all services are green:

- **RAG Chat**: http://localhost:8000
- **Airflow**: http://localhost:8080 (admin/admin)
- **Redis UI**: http://localhost:8001
- **ChromaDB**: http://localhost:8003

---

## 📥 Document Ingestion Workflow

### Method 1: Automated (Production)

1. Place PDF/DOCX/TXT files in `./data/uploads/`

2. Trigger Airflow DAG:
   - Go to http://localhost:8080
   - Login: admin/admin
   - Find DAG: `document_ingestion_pipeline`
   - Click "Trigger DAG"

3. Monitor progress in Airflow UI

4. Processed files move to `./data/processed/`

### Method 2: Quick Test (Development)

Upload directly through Chainlit UI at http://localhost:8000

---

## 💬 Using the Chat Interface

### Basic Usage

```
You: What is the main topic of the document?
Agent: [Retrieves, grades, generates, verifies]

You: Summarize page 5
Agent: [Page-specific retrieval]

You: What are the key findings?
Agent: [Semantic search across document]
```

### Commands

- `/stats` - View cache hit rate and performance
- `/clear` - Clear semantic cache
- `/help` - Show welcome message

### How It Works

1. **Cache Check**: Query checked against semantic cache (~5ms)
2. **On Cache Miss**:
   - Retrieve documents from ChromaDB
   - Grade documents for relevance (LLM)
   - If irrelevant → Transform query → Retry
   - Generate answer from relevant docs
   - Verify answer against context
   - If hallucination → Retry with stronger grounding
3. **Cache Result**: Store for future similar queries

---

## 🔧 Common Operations

### View Logs
```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f rag_app
docker compose logs -f airflow_scheduler
docker compose logs -f redis
```

### Check Status
```bash
docker compose ps
```

### Restart Service
```bash
docker compose restart rag_app
```

### Stop Everything
```bash
docker compose down
```

### Full Cleanup (Remove Volumes)
```bash
docker compose down -v
```

### Update Code
```bash
# Pull changes
git pull

# Rebuild and restart
docker compose up -d --build
```

---

## 🐛 Troubleshooting

### GPU Not Detected
```bash
# Test NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# If fails, install nvidia-container-toolkit:
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

### vLLM Not Accessible from Docker
```bash
# Test from host
curl http://localhost:8005/v1/models

# Test from container
docker exec -it rag_app curl http://host.docker.internal:8005/v1/models

# If fails, check firewall and Docker network settings
```

### Out of Memory
```bash
# Reduce batch sizes in .env
EMBEDDING_BATCH_SIZE=32
RETRIEVAL_TOP_K=4

# Reduce vLLM memory
VISION_GPU_MEMORY_UTILIZATION=0.6
REASONING_GPU_MEMORY_UTILIZATION=0.7
```

### ChromaDB Connection Failed
```bash
# Check health
curl http://localhost:8003/api/v1/heartbeat

# Restart
docker compose restart chromadb

# Check logs
docker compose logs chromadb
```

### Airflow DAG Not Running
```bash
# Check scheduler
docker compose logs airflow_scheduler

# Manually trigger
docker exec -it rag_airflow_webserver \
  airflow dags trigger document_ingestion_pipeline

# Unpause DAG if paused
docker exec -it rag_airflow_webserver \
  airflow dags unpause document_ingestion_pipeline
```

---

## 📊 Performance Tuning

### For Dual RTX 5090 (64GB Total VRAM)

**Optimal Settings**:
```bash
# Vision (3B params, ~6GB VRAM)
VISION_GPU_MEMORY_UTILIZATION=0.8

# Reasoning (7B params, ~14GB VRAM)
REASONING_GPU_MEMORY_UTILIZATION=0.85

# Embeddings share GPU 0 (~2GB overhead)
EMBEDDING_BATCH_SIZE=64
```

### For Single GPU
```bash
# Share GPU between models (not recommended)
# Or use CPU for embeddings:
# In src/agent/langgraph_agent.py:
# self.embedding_model.to('cpu')
```

### High Concurrency
```bash
# Increase limits
LLM_CONCURRENT_LIMIT=4
VISION_CONCURRENT_LIMIT=2

# Increase cache
CACHE_MAX_SIZE=20000
```

---

## 🔒 Security Checklist

Before deploying to production:

- [ ] Change `AIRFLOW__CORE__FERNET_KEY` in .env
- [ ] Change `AIRFLOW__WEBSERVER__SECRET_KEY` in .env
- [ ] Change Airflow admin password (admin/admin)
- [ ] Enable Redis authentication
- [ ] Set up HTTPS/TLS for web interfaces
- [ ] Restrict network access (firewall rules)
- [ ] Set up backup strategy for volumes
- [ ] Enable audit logging
- [ ] Review and harden Docker configurations

---

## 📈 Monitoring

### Cache Performance
```bash
# In chat: /stats
# Or check Redis directly:
docker exec -it rag_redis redis-cli
> FT.INFO query_cache_idx
```

### GPU Usage
```bash
# From host
nvidia-smi

# Watch continuously
watch -n 1 nvidia-smi
```

### Container Resources
```bash
docker stats
```

### Vector Store Size
```bash
# From Python
docker exec -it rag_app python -c "
import chromadb
client = chromadb.HttpClient(host='chromadb', port=8003)
col = client.get_collection('documents')
print(f'Total chunks: {col.count()}')
"
```

---

## 🎯 Quick Test

After startup, test the system:

1. **Upload a document** through Airflow or Chainlit
2. **Ask a question**: "What is this document about?"
3. **Check cache**: Use `/stats` command
4. **Ask similar question**: Should see cache hit
5. **View logs**: `docker compose logs -f rag_app`

---

## 📞 Getting Help

- Check logs: `docker compose logs -f`
- View service status: `docker compose ps`
- Test endpoints: `curl http://localhost:PORT/health`
- Review README.md for detailed documentation
- Open GitHub issue for bugs

---

**Tips**:
- Start vLLM servers first (they take time to load models)
- Wait for "Service ready" messages in logs
- First run downloads models (~20GB) - be patient
- Use `/stats` frequently to verify cache is working
- Monitor GPU memory with `nvidia-smi`
