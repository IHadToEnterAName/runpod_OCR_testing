# Deployment Checklist

## Pre-Deployment

### Hardware Requirements
- [ ] NVIDIA GPU(s) with 16GB+ VRAM (tested on RTX 5090)
- [ ] 32GB+ System RAM
- [ ] 100GB+ Free disk space
- [ ] CUDA 12.1+ installed
- [ ] NVIDIA Docker runtime configured

### Software Requirements
- [ ] Docker 24.0+ installed
- [ ] Docker Compose V2 installed
- [ ] NVIDIA Container Toolkit installed
- [ ] Git installed

### Network Requirements
- [ ] Ports available: 6379, 8000, 8001, 8002, 8003, 8005, 8006, 8080
- [ ] Internet connection for model downloads (~20GB on first run)
- [ ] Firewall rules configured (if applicable)

## Installation Steps

### 1. Clone and Configure
```bash
# Clone repository
git clone <your-repo-url>
cd production-rag

# Copy environment template
cp .env.example .env

# Edit .env with your settings
nano .env
```

**Critical .env settings to update:**
- [ ] GPU memory allocation (VISION_GPU_MEMORY_UTILIZATION, REASONING_GPU_MEMORY_UTILIZATION)
- [ ] Security keys (AIRFLOW__CORE__FERNET_KEY, AIRFLOW__WEBSERVER__SECRET_KEY)
- [ ] Admin passwords (change from defaults!)
- [ ] Cache parameters (CACHE_SIMILARITY_THRESHOLD, CACHE_TTL_SECONDS)

### 2. Start vLLM Servers

**Terminal 1 - Vision Model (GPU 0):**
```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --port 8006 \
  --gpu-memory-utilization 0.8 \
  --tensor-parallel-size 1 \
  --device cuda:0
```
- [ ] Vision server started
- [ ] Models downloaded
- [ ] Server responding at http://localhost:8006/v1/models

**Terminal 2 - Reasoning Model (GPU 1):**
```bash
python -m vllm.entrypoints.openai.api_server \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --port 8005 \
  --gpu-memory-utilization 0.85 \
  --tensor-parallel-size 1 \
  --device cuda:1
```
- [ ] Reasoning server started
- [ ] Models downloaded
- [ ] Server responding at http://localhost:8005/v1/models

### 3. Launch Docker Stack

```bash
# Build and start all services
docker compose up -d

# Check service status
docker compose ps

# Wait for initialization (~2-3 minutes)
./test_system.py
```

- [ ] All containers started
- [ ] No error messages in logs
- [ ] Health checks passing
- [ ] Test script shows all services healthy

### 4. Verify Services

- [ ] RAG App: http://localhost:8000 (loads chat interface)
- [ ] Airflow: http://localhost:8080 (login: admin/admin)
- [ ] RedisInsight: http://localhost:8001 (Redis UI)
- [ ] ChromaDB: http://localhost:8003 (API responding)

### 5. Initialize Data

**Option A - Upload via Airflow (Recommended):**
```bash
# Place documents in uploads directory
cp your_documents.pdf data/uploads/

# Trigger DAG via Airflow UI or CLI
docker exec -it rag_airflow_webserver \
  airflow dags trigger document_ingestion_pipeline
```
- [ ] Documents placed in data/uploads/
- [ ] DAG triggered successfully
- [ ] Documents processed (check Airflow logs)
- [ ] Chunks visible in ChromaDB

**Option B - Quick test via UI:**
- [ ] Upload document through Chainlit interface
- [ ] Document processed successfully
- [ ] Query returns relevant results

### 6. Test Functionality

```bash
# Run system test
./test_system.py
```

Expected results:
- [ ] All services healthy
- [ ] ChromaDB contains collections
- [ ] Redis connected and operational
- [ ] vLLM inference working

**Manual test queries:**
- [ ] "What is this document about?" (should retrieve and answer)
- [ ] Ask same question again (should hit cache - check `/stats`)
- [ ] "Summarize page 5" (should use page filtering)
- [ ] Complex multi-part question (should trigger agent reasoning)

## Post-Deployment

### Security Hardening
- [ ] Changed all default passwords
- [ ] Generated new secret keys
- [ ] Enabled Redis authentication (if production)
- [ ] Configured firewall rules
- [ ] Set up HTTPS/TLS for web interfaces
- [ ] Implemented backup strategy
- [ ] Enabled audit logging
- [ ] Restricted network access to necessary ports only

### Monitoring Setup
- [ ] GPU monitoring (nvidia-smi, Grafana)
- [ ] Container resource monitoring (Docker stats, cAdvisor)
- [ ] Application logging configured
- [ ] Alert thresholds set
- [ ] Health check endpoints monitored
- [ ] Cache performance tracking

### Backup Configuration
- [ ] Redis data backup scheduled
- [ ] ChromaDB volume backup scheduled
- [ ] Airflow metadata backup scheduled
- [ ] Application logs archived
- [ ] Model cache documented

### Documentation
- [ ] System architecture documented
- [ ] API endpoints documented
- [ ] Troubleshooting guide prepared
- [ ] Runbook created for operations team
- [ ] Disaster recovery plan documented

## Production Readiness

### Performance Baseline
Record initial performance metrics:
- [ ] Average query latency: _____ seconds
- [ ] Cache hit rate: _____ %
- [ ] GPU utilization: _____ %
- [ ] Concurrent users supported: _____
- [ ] Documents per minute (ingestion): _____

### Load Testing
- [ ] Tested with expected user load
- [ ] Tested with peak load (2x expected)
- [ ] Memory usage acceptable under load
- [ ] No memory leaks detected
- [ ] Graceful degradation under stress

### Failure Testing
- [ ] Tested Redis failure (should degrade gracefully)
- [ ] Tested ChromaDB failure (should error gracefully)
- [ ] Tested vLLM failure (should show error message)
- [ ] Tested disk full scenario
- [ ] Tested network partition

## Maintenance

### Daily
- [ ] Check service health (`docker compose ps`)
- [ ] Review error logs
- [ ] Monitor cache hit rate (`/stats`)
- [ ] Check GPU utilization (`nvidia-smi`)

### Weekly
- [ ] Review and clear old processed files
- [ ] Check disk space usage
- [ ] Review Airflow DAG execution logs
- [ ] Update any failed ingestion jobs

### Monthly
- [ ] Update Docker images (if available)
- [ ] Review and optimize cache settings
- [ ] Audit system security
- [ ] Backup verification test
- [ ] Performance benchmarking

## Troubleshooting Quick Reference

### Service Won't Start
```bash
# Check logs
docker compose logs [service_name]

# Restart specific service
docker compose restart [service_name]

# Full restart
docker compose down && docker compose up -d
```

### Out of Memory
```bash
# Check GPU memory
nvidia-smi

# Reduce batch sizes in .env
# Reduce vLLM memory utilization
# Restart services
```

### Cache Not Working
```bash
# Check Redis connection
docker exec -it rag_redis redis-cli ping

# Check cache stats in app
# Use /stats command in chat

# Clear cache if needed
# Use /clear command in chat
```

### Poor Performance
```bash
# Check GPU utilization
nvidia-smi

# Check cache hit rate
# Should be >70% after warmup

# Check ChromaDB query time
# Should be <300ms

# Review system resources
docker stats
```

## Support

### Get Help
- Review logs: `docker compose logs -f`
- Check README.md for detailed documentation
- Run health check: `./test_system.py`
- Review ARCHITECTURE.md for system design
- Check HOWTO.md for common operations

### Report Issues
When reporting issues, include:
1. Output of `docker compose ps`
2. Relevant logs from `docker compose logs`
3. Output of `nvidia-smi`
4. Output of `./test_system.py`
5. Steps to reproduce the issue

---

**Deployment Date:** __________
**Deployed By:** __________
**Environment:** [ ] Development [ ] Staging [ ] Production
**Notes:** ___________________________________________________
