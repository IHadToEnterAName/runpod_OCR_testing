# 🚀 5-Minute Quick Start Guide

## Prerequisites Check ✓

```bash
# Verify your system
docker --version          # ✓ Docker 24.0+
docker compose version    # ✓ Docker Compose V2
nvidia-smi               # ✓ NVIDIA GPU detected
```

## Step 1: Download & Configure (2 min)

```bash
# Get the code
git clone <your-repo-url>
cd production-rag

# Configure environment
cp .env.example .env
nano .env  # Quick edits:
           # - GPU memory (VISION_GPU_MEMORY_UTILIZATION=0.8)
           # - Cache threshold (CACHE_SIMILARITY_THRESHOLD=0.95)
```

## Step 2: Start vLLM Servers (1 min)

**Terminal 1 - Vision Model:**
```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --port 8006 \
  --gpu-memory-utilization 0.8 \
  --device cuda:0
```

**Terminal 2 - Reasoning Model:**
```bash
python -m vllm.entrypoints.openai.api_server \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --port 8005 \
  --gpu-memory-utilization 0.85 \
  --device cuda:1
```

⏳ Wait for "Application startup complete" messages (~30-60 seconds for model loading)

## Step 3: Launch Stack (1 min)

**Terminal 3:**
```bash
./start.sh
# Select: 1) Start all services
```

⏳ Wait for services to initialize (~1-2 minutes)

## Step 4: Verify (1 min)

```bash
# Run health check
./test_system.py

# Expected output:
# ✅ RAG App              - OK (200)
# ✅ Airflow              - OK (200)
# ✅ Redis                - OK
# ✅ ChromaDB             - OK (200)
# ✅ vLLM Vision          - OK (200)
# ✅ vLLM Reasoning       - OK (200)
```

## Step 5: Use It! 🎉

### Access Services

Open in browser:
- **Chat Interface**: http://localhost:8000
- **Airflow Dashboard**: http://localhost:8080 (admin/admin)
- **Redis UI**: http://localhost:8001

### First Document Upload

**Option A - Via Airflow (Production):**
```bash
# 1. Copy document
cp your_document.pdf data/uploads/

# 2. Go to http://localhost:8080
# 3. Find DAG: document_ingestion_pipeline
# 4. Click "Trigger DAG" button
# 5. Wait ~60 seconds
```

**Option B - Via Chat (Quick Test):**
```
1. Go to http://localhost:8000
2. Drag & drop PDF file
3. Wait for processing
```

### Ask Questions

```
You: What is this document about?
Agent: [Retrieves, grades docs, generates, verifies]
      📊 Agent Metadata:
      • Grading: relevant
      • Verification: verified
      • Retries: 0
      • Retrieved Docs: 5
      
      [Answer based on document...]

You: Summarize page 3
Agent: [Page-specific retrieval]
      [Answer for page 3...]

You: /stats
Agent: 📊 Cache Statistics
      Hit Rate: 50.0%
      Total Requests: 2
      Cache Hits: 1
      Cache Misses: 1
```

## What Just Happened?

### Behind the Scenes

```
Your Query
    ↓
1. Cache Check (5ms)
    ↓ MISS
2. Retrieve from ChromaDB (200ms)
    ↓
3. Grade Documents (800ms)
    ↓ RELEVANT
4. Generate Answer (2s)
    ↓
5. Verify Answer (400ms)
    ↓ VERIFIED
6. Cache Result (5ms)
    ↓
Return Answer (Total: ~3.4s first time)

Same Query Again
    ↓
1. Cache Check (5ms)
    ↓ HIT!
Return Answer (Total: 5ms)
```

## Common Commands

```bash
# View logs
docker compose logs -f rag_app

# Check service status
docker compose ps

# Restart a service
docker compose restart rag_app

# Stop everything
docker compose down

# Full cleanup (removes data)
docker compose down -v

# Health check
./test_system.py

# Interactive control
./start.sh
```

## Chat Commands

In the chat interface:
- `/stats` - View cache performance
- `/clear` - Clear semantic cache
- `/help` - Show help message

## Troubleshooting Quick Fixes

### GPU Not Detected
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Service Won't Start
```bash
docker compose logs [service_name]
docker compose restart [service_name]
```

### Out of Memory
Edit `.env`:
```bash
VISION_GPU_MEMORY_UTILIZATION=0.6    # Lower from 0.8
REASONING_GPU_MEMORY_UTILIZATION=0.7 # Lower from 0.85
EMBEDDING_BATCH_SIZE=32              # Lower from 64
```

### Cache Not Working
```bash
# Check Redis
docker exec -it rag_redis redis-cli ping

# Check stats
# Use /stats in chat interface
```

## Performance Expectations

### First Query (No Cache)
- Latency: 2-5 seconds
- GPU Usage: 60-80%
- Cache: MISS

### Subsequent Similar Queries
- Latency: 5-10ms
- GPU Usage: <5%
- Cache: HIT

### After 100 Queries
- Cache Hit Rate: 70-80%
- Average Latency: 200-500ms
- GPU Usage: 30-50% (balanced)

## Architecture At a Glance

```
┌───────────────────────────────────────┐
│  You → Chat Interface (Chainlit)      │
└────────────────┬──────────────────────┘
                 ↓
┌────────────────┴──────────────────────┐
│  Semantic Cache (Redis) - 80% hits    │
└────────────────┬──────────────────────┘
                 ↓ (20% miss)
┌────────────────┴──────────────────────┐
│  LangGraph Agent                      │
│  • Retrieve → Grade → Generate        │
│  • Verify → Cache                     │
└─────┬──────────────────┬──────────────┘
      ↓                  ↓
┌─────────┐      ┌───────────────┐
│ChromaDB │      │  vLLM Servers │
│ Vectors │      │  GPU 0 & GPU 1│
└─────────┘      └───────────────┘
      ↑
      │
┌─────────┐
│ Airflow │
│   ETL   │
└─────────┘
```

## Next Steps

1. ✅ Upload more documents
2. ✅ Ask various questions
3. ✅ Monitor cache hit rate
4. ✅ Review logs
5. ✅ Read full documentation (README.md)
6. ✅ Customize for your use case

## Support

- **Documentation**: README.md (comprehensive)
- **Quick Ops**: HOWTO.md (cheat sheet)
- **System Design**: ARCHITECTURE.md (deep dive)
- **Deployment**: DEPLOYMENT_CHECKLIST.md (production)
- **Health Check**: `./test_system.py`
- **Logs**: `docker compose logs -f`

---

**🎉 Congratulations!** You now have a production-ready agentic RAG system running!

For more details, see:
- **PROJECT_SUMMARY.md** - Complete overview
- **README.md** - Full documentation
- **ARCHITECTURE.md** - System design
