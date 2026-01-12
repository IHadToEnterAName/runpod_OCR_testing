# System Architecture

## Component Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                                 │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │           Chainlit Web Interface (Port 7860)                │   │
│  │                                                               │   │
│  │  • Chat UI                                                   │   │
│  │  • File Upload                                               │   │
│  │  • Real-time Streaming                                       │   │
│  │  • Command Interface (/stats, /clear, /help)                │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                       │
└──────────────────────────────┼───────────────────────────────────────┘
                               │
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      APPLICATION LAYER                               │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │           Semantic Cache (Redis Stack)                   │      │
│  │                                                            │      │
│  │  • Vector Search Index                                   │      │
│  │  • Similarity Matching (Cosine)                          │      │
│  │  • TTL & Eviction                                         │      │
│  │  • Hit Rate: ~80% (typical)                              │      │
│  └──────────────────────────────────────────────────────────┘      │
│                              ↓ (on miss)                             │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │         LangGraph Agentic RAG Pipeline                    │      │
│  │                                                            │      │
│  │   ┌─────────────────────────────────────────────────┐   │      │
│  │   │  1. RETRIEVE NODE                                │   │      │
│  │   │     • Query ChromaDB (semantic search)          │   │      │
│  │   │     • Top-K retrieval (default: 8)              │   │      │
│  │   │     • Page filtering support                    │   │      │
│  │   └─────────────────────────────────────────────────┘   │      │
│  │                      ↓                                    │      │
│  │   ┌─────────────────────────────────────────────────┐   │      │
│  │   │  2. GRADE DOCUMENTS NODE (Corrective RAG)       │   │      │
│  │   │     • LLM grades each document                  │   │      │
│  │   │     • Relevance scoring                         │   │      │
│  │   │     • Decision: relevant / partial / irrelevant│   │      │
│  │   └─────────────────────────────────────────────────┘   │      │
│  │           ↓ relevant            ↓ irrelevant             │      │
│  │   ┌─────────────────┐   ┌────────────────────────┐      │      │
│  │   │  3A. GENERATE   │   │ 3B. TRANSFORM QUERY    │      │      │
│  │   │      NODE       │   │      NODE              │      │      │
│  │   │                 │   │                        │      │      │
│  │   │  • Context from │   │  • LLM rewrites query  │      │      │
│  │   │    relevant docs│   │  • Retry retrieval     │──┐   │      │
│  │   │  • Answer gen   │   │  • Max 2 retries       │  │   │      │
│  │   └─────────────────┘   └────────────────────────┘  │   │      │
│  │           ↓                                          │   │      │
│  │           └──────────────────────────────────────────┘   │      │
│  │                      ↓                                    │      │
│  │   ┌─────────────────────────────────────────────────┐   │      │
│  │   │  4. VERIFY ANSWER NODE (Self-Reflective RAG)    │   │      │
│  │   │     • Check answer vs context                   │   │      │
│  │   │     • Detect hallucinations                     │   │      │
│  │   │     • Result: verified / hallucination          │   │      │
│  │   └─────────────────────────────────────────────────┘   │      │
│  │           ↓ verified            ↓ hallucination          │      │
│  │   ┌─────────────────┐   ┌────────────────────────┐      │      │
│  │   │  5A. END        │   │ 5B. RETRY GENERATION   │      │      │
│  │   │                 │   │     • Stronger grounding│──┐   │      │
│  │   │  • Cache result │   │     • Back to generate │  │   │      │
│  │   │  • Return answer│   └────────────────────────┘  │   │      │
│  │   └─────────────────┘                               │   │      │
│  │                                                      │   │      │
│  │           └──────────────────────────────────────────┘   │      │
│  └──────────────────────────────────────────────────────────┘      │
│                                                                       │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                       INFERENCE LAYER                                │
│                                                                       │
│  ┌──────────────────────┐      ┌──────────────────────┐            │
│  │   vLLM Vision Server │      │  vLLM Reasoning      │            │
│  │   (GPU 0, Port 8006) │      │  Server (GPU 1, 8005)│            │
│  │                       │      │                       │            │
│  │  • Qwen2.5-VL-3B     │      │  • DeepSeek-R1-7B    │            │
│  │  • Image analysis    │      │  • RAG generation     │            │
│  │  • ~6GB VRAM         │      │  • ~14GB VRAM         │            │
│  └──────────────────────┘      └──────────────────────┘            │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                   │
│                                                                       │
│  ┌──────────────────────┐      ┌──────────────────────┐            │
│  │   ChromaDB           │      │  Redis Stack          │            │
│  │   (Port 8000)        │      │  (Port 6379)          │            │
│  │                       │      │                       │            │
│  │  • Document vectors  │      │  • Semantic cache     │            │
│  │  • Cosine similarity │      │  • State persistence  │            │
│  │  • Metadata filtering│      │  • Session storage    │            │
│  └──────────────────────┘      └──────────────────────┘            │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
                               ↑
┌─────────────────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE (Airflow)                      │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                   DAG: document_ingestion_pipeline            │  │
│  │                                                                │  │
│  │   1. SCAN       →   2. EXTRACT    →   3. CHUNK               │  │
│  │   (uploads dir)     (PDF/DOCX/TXT)    (800 chars)            │  │
│  │                                                                │  │
│  │   4. EMBED      →   5. STORE      →   6. CLEANUP             │  │
│  │   (SentenceTransf)  (ChromaDB)        (processed dir)         │  │
│  │                                                                │  │
│  │   Schedule: Daily 2 AM | Manual Trigger                       │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  Airflow UI: http://localhost:8080                                  │
└───────────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Query Processing Flow

```
User Query
    ↓
┌───────────────────────────────────────┐
│ 1. Semantic Cache Lookup               │
│    • Embed query                        │
│    • Search Redis vector index          │
│    • Similarity threshold: 0.95         │
└───────────────────────────────────────┘
    ↓
    ├─ HIT (80% typical) → Return cached answer (5-10ms)
    │
    └─ MISS (20% typical) ↓
┌───────────────────────────────────────┐
│ 2. Agentic RAG Pipeline                 │
│                                          │
│   a. Retrieve from ChromaDB             │
│      • Embed query                       │
│      • Top-K similarity search          │
│      • Retrieve 8 chunks                │
│                                          │
│   b. Grade Documents (Corrective)       │
│      • LLM judges relevance             │
│      • Filter irrelevant docs           │
│      • Transform query if needed        │
│                                          │
│   c. Generate Answer                    │
│      • Build context window             │
│      • LLM generates response           │
│                                          │
│   d. Verify (Self-Reflective)          │
│      • Check for hallucinations         │
│      • Retry if needed                  │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│ 3. Cache & Return                       │
│    • Store in Redis cache               │
│    • Return to user                     │
│    • Total time: 2-5 seconds           │
└───────────────────────────────────────┘
```

### Document Ingestion Flow

```
PDF/DOCX/TXT File → data/uploads/
    ↓
┌───────────────────────────────────────┐
│ Airflow DAG Execution                   │
│                                          │
│ 1. Scan: Find new documents             │
│    ↓                                     │
│ 2. Extract: Text + Images               │
│    • PDF: PyMuPDF (text + images)      │
│    • DOCX: python-docx                  │
│    • TXT: plain read                    │
│    ↓                                     │
│ 3. Chunk: Smart splitting               │
│    • Size: 800 chars                    │
│    • Overlap: 100 chars                 │
│    • Preserve page numbers              │
│    ↓                                     │
│ 4. Embed: Generate vectors              │
│    • Model: nomic-embed-text-v1.5      │
│    • Batch size: 64                     │
│    • Dimension: 768                     │
│    ↓                                     │
│ 5. Store: Save to ChromaDB              │
│    • Upsert chunks                      │
│    • With metadata                      │
│    • Deduplication by hash              │
│    ↓                                     │
│ 6. Cleanup: Move to processed/          │
└───────────────────────────────────────┘
    ↓
Document ready for queries
```

## State Management

### LangGraph State (Redis-backed)

```python
AgentState {
    messages: List[Message],          # Conversation history
    query: str,                        # Current query
    retrieved_documents: List[Doc],   # Raw retrieval results
    relevant_documents: List[Doc],    # After grading
    generation: str,                   # Generated answer
    retry_count: int,                  # Retry attempts
    grading_decision: str,             # relevant/irrelevant/partial
    verification_result: str           # verified/hallucination/uncertain
}
```

State persists across:
- Multiple conversation turns
- Query transformations
- Retry attempts
- Session boundaries (via Redis checkpointer)

## Performance Characteristics

### Latency

| Operation | Typical Time | Notes |
|-----------|-------------|-------|
| Cache Hit | 5-10ms | Semantic similarity lookup |
| Cache Miss + Full RAG | 2-5s | Depends on document size |
| Document Retrieval | 100-300ms | ChromaDB vector search |
| LLM Document Grading | 500-1000ms | Per top-5 documents |
| Answer Generation | 1-2s | Depends on output length |
| Answer Verification | 300-500ms | Quick LLM check |

### Throughput

| Component | Capacity | Bottleneck |
|-----------|----------|------------|
| Cache Lookups | 10,000+ req/s | Redis I/O |
| Vector Search | 100-500 req/s | ChromaDB HNSW |
| LLM Inference | 5-10 req/s | GPU memory |
| Concurrent Users | 20-50 | GPU utilization |

### GPU Utilization

```
┌──────────────────────────────────┐
│  RTX 5090 GPU 0 (32GB)           │
│  ┌────────────────────────────┐  │
│  │  vLLM Vision (6GB)         │  │
│  │  Qwen2.5-VL-3B             │  │
│  └────────────────────────────┘  │
│  ┌────────────────────────────┐  │
│  │  SentenceTransformer (2GB) │  │
│  │  nomic-embed-text          │  │
│  └────────────────────────────┘  │
│  Available: 24GB               │  │
└──────────────────────────────────┘

┌──────────────────────────────────┐
│  RTX 5090 GPU 1 (32GB)           │
│  ┌────────────────────────────┐  │
│  │  vLLM Reasoning (14GB)     │  │
│  │  DeepSeek-R1-7B            │  │
│  └────────────────────────────┘  │
│  Available: 18GB               │  │
└──────────────────────────────────┘
```

## Scaling Considerations

### Horizontal Scaling

- **Cache**: Redis Cluster (multi-node)
- **Vector DB**: ChromaDB with read replicas
- **Web App**: Multiple Chainlit instances behind load balancer
- **Airflow**: Celery executor with worker pool

### Vertical Scaling

- **More GPUs**: Add tensor parallelism to vLLM
- **More RAM**: Increase batch sizes
- **Faster Storage**: NVMe for vector DB persistence

## Security Architecture

```
┌──────────────────────────────────────────┐
│  External (Internet/VPN)                  │
│                                            │
│  ┌──────────────────────────────────┐    │
│  │  Reverse Proxy (nginx)           │    │
│  │  • HTTPS/TLS                      │    │
│  │  • Rate limiting                  │    │
│  │  • WAF rules                      │    │
│  └──────────────────────────────────┘    │
└──────────────────┬───────────────────────┘
                   │
┌──────────────────┼───────────────────────┐
│  Docker Network (rag_network)            │
│                  │                        │
│  ┌───────────────┼──────────────────┐    │
│  │  Application Layer                │    │
│  │  • No external exposure           │    │
│  │  • Internal DNS resolution        │    │
│  │  • Service-to-service auth        │    │
│  └───────────────────────────────────┘    │
│                                            │
│  ┌──────────────────────────────────┐    │
│  │  Data Layer                       │    │
│  │  • Encrypted volumes              │    │
│  │  • No direct external access      │    │
│  │  • Backup encryption              │    │
│  └──────────────────────────────────┘    │
└────────────────────────────────────────────┘
```

## Monitoring Points

1. **Application Metrics**
   - Cache hit rate (target: >80%)
   - Query latency (p50, p95, p99)
   - Error rate
   - Active sessions

2. **Infrastructure Metrics**
   - GPU utilization (target: 60-80%)
   - GPU memory usage
   - Container CPU/RAM
   - Disk I/O

3. **Business Metrics**
   - Queries per minute
   - Documents processed
   - User satisfaction (thumbs up/down)
   - Average session length

4. **Health Checks**
   - Service availability (5 services)
   - vLLM endpoint responsiveness
   - Vector DB query speed
   - Cache hit rate trends
