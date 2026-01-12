# Production Agentic RAG System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-required-blue.svg)](https://www.docker.com/)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)

A production-ready Retrieval-Augmented Generation (RAG) system with agentic reasoning capabilities, optimized for dual NVIDIA RTX 5090 GPUs.

## 🎯 Key Features

### Agentic RAG with LangGraph
- **Corrective RAG**: Automatic document grading and query transformation
- **Self-Reflective RAG**: Answer verification to prevent hallucinations
- **Stateful Conversations**: Redis-backed state persistence across sessions

### Production-Grade Infrastructure
- **Semantic Caching**: Redis-powered cache with embedding similarity matching
- **Automated Ingestion**: Airflow DAG for document processing pipeline
- **Vector Storage**: ChromaDB for efficient similarity search
- **Dockerized Stack**: Complete multi-container setup with GPU support

### Performance Optimizations
- **Dual GPU Support**: Separate GPUs for vision and reasoning models
- **Smart Chunking**: Page-aware document splitting (800 chars)
- **Batch Processing**: Optimized embedding generation
- **Concurrent Limits**: Semaphore-based rate limiting

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Interface                          │
│                    (Chainlit Web App)                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Semantic Cache Layer                        │
│              (Redis with Vector Search)                      │
└────────────────────────┬────────────────────────────────────┘
                         │ Cache Miss
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   LangGraph Agent                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  1. Retrieve    →  2. Grade Documents               │   │
│  │                                                       │   │
│  │  3. Generate    →  4. Verify Answer                 │   │
│  │         │                    │                        │   │
│  │         └─→ Transform Query ─┘ (if needed)          │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
    ┌─────────┐   ┌──────────┐   ┌──────────┐
    │ ChromaDB│   │  vLLM    │   │  vLLM    │
    │ Vectors │   │  Vision  │   │ Reasoning│
    │         │   │ (GPU 0)  │   │ (GPU 1)  │
    └─────────┘   └──────────┘   └──────────┘
         ▲
         │
┌────────┴────────┐
│     Airflow     │
│  Ingestion DAG  │
│  (Automated)    │
└─────────────────┘
```

## 📋 Prerequisites

- **Hardware**: NVIDIA GPU with 16GB+ VRAM (tested on RTX 5090)
- **Software**:
  - Docker 24.0+ with Docker Compose
  - NVIDIA Docker Runtime
  - CUDA 12.1+
  - 32GB+ System RAM

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone repository
git clone <your-repo-url>
cd production-rag

# Copy environment template
cp .env.example .env

# Edit .env with your settings
nano .env
```

### 2. Start vLLM Servers (Host)

Start your vLLM inference servers on the host machine:

```bash
# Terminal 1 - Vision Model (GPU 0)
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --port 8006 \
  --gpu-memory-utilization 0.8 \
  --tensor-parallel-size 1 \
  --device cuda:0

# Terminal 2 - Reasoning Model (GPU 1)  
python -m vllm.entrypoints.openai.api_server \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --port 8005 \
  --gpu-memory-utilization 0.85 \
  --tensor-parallel-size 1 \
  --device cuda:1
```

### 3. Launch Stack

```bash
# Build and start all services
docker compose up -d

# View logs
docker compose logs -f rag_app

# Check service health
docker compose ps
```

### 4. Access Services

- **RAG Application**: http://localhost:7860
- **Airflow UI**: http://localhost:8080 (admin/admin)
- **RedisInsight**: http://localhost:8001
- **ChromaDB**: http://localhost:8000

## 📂 Project Structure

```
.
├── src/
│   ├── agent/
│   │   └── langgraph_agent.py      # LangGraph agentic RAG
│   ├── cache/
│   │   └── semantic_cache.py       # Redis semantic cache
│   ├── airflow/
│   │   └── dags/
│   │       └── document_ingestion.py  # Airflow ETL DAG
│   └── app.py                      # Main Chainlit application
├── data/
│   ├── uploads/                    # Place documents here
│   ├── processed/                  # Processed documents
│   └── failed/                     # Failed processing
├── Dockerfile                      # Application container
├── docker-compose.yml              # Multi-service orchestration
├── requirements.txt                # Python dependencies
└── .env.example                    # Environment template
```

## 🔧 Configuration

### GPU Memory Allocation

Edit `.env` to configure VRAM usage:

```bash
# Vision model (lighter)
VISION_GPU_MEMORY_UTILIZATION=0.8

# Reasoning model (heavier)
REASONING_GPU_MEMORY_UTILIZATION=0.85
```

### Cache Settings

Adjust semantic cache behavior:

```bash
# Similarity threshold (0.0-1.0)
CACHE_SIMILARITY_THRESHOLD=0.95

# Cache TTL in seconds
CACHE_TTL_SECONDS=3600

# Max cached queries
CACHE_MAX_SIZE=10000
```

### RAG Parameters

Fine-tune retrieval and generation:

```bash
# Retrieval
RETRIEVAL_TOP_K=8
CHUNK_SIZE=800
CHUNK_OVERLAP=100

# Generation
TEMPERATURE=0.3
MAX_TOKENS=2048
MAX_RETRIES=2
```

## 📥 Document Ingestion

### Method 1: Automated (Recommended)

Place documents in `data/uploads/` and trigger the Airflow DAG:

```bash
# Access Airflow UI
open http://localhost:8080

# Trigger DAG: "document_ingestion_pipeline"
# Or schedule it (runs daily at 2 AM by default)
```

### Method 2: Manual Upload

Upload files directly through the Chainlit UI (for testing only).

### Supported Formats

- PDF (with text and image extraction)
- DOCX (Microsoft Word)
- TXT (Plain text)
- PNG/JPG (Images)

## 💬 Usage

### Basic Queries

```
User: What is discussed on page 5?
Agent: [Retrieves page 5, grades documents, generates answer]

User: Summarize the main findings
Agent: [Retrieves relevant sections, verifies answer]
```

### Commands

- `/stats` - View cache performance statistics
- `/clear` - Clear cache and reset statistics
- `/help` - Show welcome message

### Agent Behavior

1. **Query Processing**: 
   - Checks semantic cache first
   - On miss, proceeds to agent

2. **Document Grading**:
   - LLM grades each retrieved document
   - If irrelevant, transforms query and retries

3. **Answer Generation**:
   - Generates answer from relevant docs
   - Verifies against context
   - Retries if hallucination detected

4. **Caching**:
   - Stores result with metadata
   - Future similar queries return instantly

## 🔍 Monitoring

### Application Logs

```bash
# RAG app
docker compose logs -f rag_app

# Airflow scheduler
docker compose logs -f airflow_scheduler

# Redis
docker compose logs -f redis
```

### Cache Statistics

Use the `/stats` command in the chat or check RedisInsight at http://localhost:8001

### Vector Store

```python
# Query ChromaDB directly
import chromadb
client = chromadb.HttpClient(host='localhost', port=8000)
collection = client.get_collection('documents')
print(f"Total chunks: {collection.count()}")
```

## 🧪 Testing

### Test Semantic Cache

```bash
# Connect to Redis container
docker exec -it rag_redis redis-cli

# Check cache keys
KEYS cache:*

# View cache stats
FT.INFO query_cache_idx
```

### Test Retrieval

Use the `/test` command in the chat interface to verify vector search.

### Test Airflow DAG

```bash
# Trigger DAG manually
docker exec -it rag_airflow_webserver \
  airflow dags trigger document_ingestion_pipeline

# Check status
docker exec -it rag_airflow_webserver \
  airflow dags list-runs -d document_ingestion_pipeline
```

## 🐛 Troubleshooting

### GPU Not Detected

```bash
# Verify NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Check compose GPU config
docker compose config | grep -A 5 "deploy:"
```

### vLLM Connection Failed

```bash
# Test endpoints
curl http://localhost:8005/v1/models
curl http://localhost:8006/v1/models

# Check if accessible from container
docker exec -it rag_app curl http://host.docker.internal:8005/v1/models
```

### ChromaDB Connection Issues

```bash
# Restart ChromaDB
docker compose restart chromadb

# Check health
curl http://localhost:8000/api/v1/heartbeat
```

### Out of Memory

Reduce batch sizes in `.env`:

```bash
EMBEDDING_BATCH_SIZE=32  # Down from 64
RETRIEVAL_TOP_K=4        # Down from 8
```

## 🔒 Security

### Production Checklist

- [ ] Change all default passwords in `.env`
- [ ] Set strong `AIRFLOW__CORE__FERNET_KEY`
- [ ] Enable Redis authentication
- [ ] Use HTTPS for web interfaces
- [ ] Restrict network access with firewall
- [ ] Set up backup strategy for volumes
- [ ] Monitor resource usage
- [ ] Enable audit logging

### Environment Variables

Never commit `.env` to version control:

```bash
# Add to .gitignore
echo ".env" >> .gitignore
```

## 📊 Performance Tuning

### For Dual RTX 5090 (64GB Total)

```bash
# Vision model (3B params): ~6GB VRAM
VISION_GPU_MEMORY_UTILIZATION=0.8

# Reasoning model (7B params): ~14GB VRAM
REASONING_GPU_MEMORY_UTILIZATION=0.85

# Embedding model shares GPU with minimal overhead
```

### Scaling Up

To handle more concurrent users:

1. Increase Redis memory limit
2. Scale Airflow workers horizontally
3. Add read replicas for ChromaDB
4. Implement load balancing for web app

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Open a Pull Request

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- LangChain & LangGraph for agentic framework
- ChromaDB for vector database
- Anthropic for inspiration on RAG patterns
- vLLM for high-performance inference

## 📧 Support

For issues and questions:
- Open a GitHub issue
- Check existing issues for solutions
- Review logs with `docker compose logs`

---

**Built with ❤️ for production AI systems**
