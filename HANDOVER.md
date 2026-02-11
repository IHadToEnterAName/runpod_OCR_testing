# Visual RAG Document Assistant - Handover Documentation

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Directory Structure](#3-directory-structure)
4. [Getting Started](#4-getting-started)
5. [Configuration Reference](#5-configuration-reference)
6. [Module-by-Module Breakdown](#6-module-by-module-breakdown)
7. [Data Flow & Request Lifecycle](#7-data-flow--request-lifecycle)
8. [Query Routing System](#8-query-routing-system)
9. [Key Design Decisions](#9-key-design-decisions)
10. [Infrastructure & Deployment](#10-infrastructure--deployment)
11. [Troubleshooting](#11-troubleshooting)
12. [Known Limitations & Future Considerations](#12-known-limitations--future-considerations)

---

## 1. Project Overview

### What This Is

A **Visual Retrieval-Augmented Generation (RAG)** document assistant that lets users upload documents (PDF, images, DOCX, TXT) and ask questions about them through a web chat interface.

### The Key Innovation

Unlike traditional text-based RAG systems that extract text from documents, split it into chunks, and embed the chunks, this system **indexes documents as page images**. The retrieval model (ColQwen2) creates visual embeddings of entire pages, and the generation model (Qwen3-VL) reads the actual page images to answer questions. This means:

- Tables, charts, diagrams, and layouts are preserved exactly as they appear
- No information is lost through text extraction
- Handwritten notes and visual elements are naturally supported

### Core Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Retrieval** | Byaldi (ColQwen2 - `vidore/colqwen2-v0.1`) | Visual document indexing & similarity search |
| **Generation** | Qwen3-VL (via vLLM) | Vision-language model that reads page images and generates answers |
| **Web UI** | Chainlit | Chat interface with file upload, streaming responses |
| **Caching** | Redis | Caches search results (1h) and LLM responses (30m) |
| **Containerization** | Docker Compose | Redis + RAG App (vLLM runs on host) |

---

## 2. Architecture

```
                         User (Browser)
                        http://localhost:8080
                              |
                     Chainlit WebSocket
                              |
                     +--------v--------+
                     |    app.py       |
                     | (Entry Point)   |
                     +---+----+---+----+
                         |    |   |
             +-----------+    |   +----------+
             |                |              |
    +--------v------+  +-----v-----+  +-----v--------+
    |file_processor |  |  router   |  |   memory     |
    |(Upload/Index) |  | (Intent)  |  |  (History)   |
    +--------+------+  +-----+-----+  +-----+--------+
             |               |               |
             +-------+-------+-------+-------+
                     |               |
            +--------v------+  +-----v--------+
            | visual_store  |  |   cache      |
            | (Byaldi/      |  |  (Redis)     |
            |  ColQwen2)    |  +--------------+
            +--------+------+
                     |
            +--------v--------+
            |   pipeline      |
            | (Build prompt,  |
            |  stream output) |
            +--------+--------+
                     |
            +--------v--------+
            |  vLLM Server    |
            |  (Qwen3-VL)    |
            |  port 8005      |
            +-----------------+
                     |
            Stream tokens back
              to user's browser
```

### Service Layout

- **vLLM Server** runs directly on the host machine (not in Docker) for optimal GPU access
- **Redis + RAG App** run inside Docker containers, orchestrated by Docker Compose
- The RAG App container talks to vLLM via `host.docker.internal:8005`

---

## 3. Directory Structure

```
vllm_inference_llama3/
├── src/                              # All application source code
│   ├── app.py                        # Main entry point (Chainlit handlers)
│   ├── config/
│   │   └── settings.py               # All configuration (dataclasses + env vars)
│   ├── rag/                          # Core RAG pipeline modules
│   │   ├── pipeline.py               # Main generation pipeline (search → stream)
│   │   ├── router.py                 # Query intent classification & parameter tuning
│   │   ├── cache.py                  # Redis caching layer
│   │   ├── memory.py                 # Per-session conversation history
│   │   ├── traffic_controller.py     # Rate limiting, circuit breaker, health checks
│   │   └── visual_reranker.py        # Optional Qwen3-VL-based page reranking
│   ├── storage/
│   │   └── visual_store.py           # Byaldi index management (create, search, delete)
│   ├── processing/
│   │   ├── file_processor.py         # File upload handling & format conversion
│   │   └── page_screenshotter.py     # PDF → image conversion utility (PyMuPDF)
│   └── utils/
│       ├── helpers.py                # Token counting, page number extraction
│       └── grounding.py              # Bounding box parsing & drawing
├── Docker/
│   ├── Dockerfile                    # NVIDIA CUDA 12.1 + Python 3.11
│   ├── docker-compose.yml            # Redis + RAG App services
│   ├── start.sh                      # Production startup script
│   └── .env                          # Environment variables (ports, models, etc.)
├── Scripts/
│   └── requirements.txt              # Python dependencies
├── .chainlit/
│   └── config.toml                   # Chainlit UI configuration
├── public/                           # Static web assets (JS, CSS)
├── logs/                             # Runtime logs (vllm.log)
└── persistent/                       # Docker volume mounts
    ├── redis/                        # Redis RDB snapshots
    ├── indexes/                      # Byaldi document indexes
    ├── huggingface/                  # Downloaded model weights
    └── data/                         # Uploads and processed files
```

---

## 4. Getting Started

### Prerequisites

- **NVIDIA GPU** with sufficient VRAM (the .env defaults to `VLLM_GPU_MEMORY=0.22`)
- **NVIDIA Container Toolkit** (`nvidia-ctk`) for Docker GPU access
- **Docker** and **Docker Compose**
- **vLLM** installed on the host (`pip install vllm`)
- **Hugging Face token** (set `HF_TOKEN` in `Docker/.env`)

### Quick Start

```bash
# 1. Clone/navigate to the project
cd vllm_inference_llama3

# 2. Configure environment
#    Edit Docker/.env to set your HF_TOKEN, model, GPU memory, etc.

# 3. Start everything (vLLM + Docker stack)
cd Docker
chmod +x start.sh
./start.sh

# 4. Open browser
#    http://localhost:8080
```

### Start Script Options

```bash
./start.sh              # Start vLLM + Docker stack
./start.sh --no-vllm    # Skip vLLM (if already running separately)
./start.sh --build      # Force rebuild Docker images
./start.sh --logs       # Follow logs after startup
./start.sh --stop       # Stop all services
```

### What Happens on Startup

1. `start.sh` checks for Docker, Docker Compose, NVIDIA GPU
2. Configures NVIDIA Docker runtime if needed
3. Creates `persistent/` directories for data persistence
4. Starts vLLM server on host (port 8005) and waits for it to load (up to 10 min)
5. Starts Docker stack: Redis (port 6379) + RAG App (port 8080)
6. RAG App pre-loads the ColQwen2 model at startup before accepting users

### Install Dependencies (Without Docker)

If you want to run the app without Docker:

```bash
# Install PyTorch with CUDA first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install vLLM
pip install vllm

# Install app dependencies
pip install -r Scripts/requirements.txt

# Start vLLM server manually
vllm serve "Qwen/Qwen3-VL-8B-Instruct-FP8" \
    --port 8005 \
    --gpu-memory-utilization 0.22 \
    --max-model-len 32768 \
    --limit-mm-per-prompt '{"image":8}' \
    --trust-remote-code \
    --enable-prefix-caching

# Start Redis
redis-server

# Start the app
cd src
PYTHONPATH=. python -m chainlit run app.py --host 0.0.0.0 --port 8000
```

---

## 5. Configuration Reference

All configuration lives in `src/config/settings.py` as Python dataclasses. Values can be overridden via environment variables (set in `Docker/.env`).

### Model Configuration (`ModelConfig`)

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `base_url` | `VLLM_URL` | `http://localhost:8005/v1` | vLLM OpenAI-compatible endpoint |
| `model_name` | `VLLM_MODEL` | `QuantTrio/Qwen3-VL-32B-Instruct-AWQ` | Model served by vLLM |

### Byaldi Configuration (`ByaldiConfig`)

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `model_name` | `BYALDI_MODEL` | `vidore/colqwen2-v0.1` | ColQwen2 visual embedding model |
| `index_path` | `BYALDI_INDEX_PATH` | `/workspace/data/indexes` | Where indexes are stored on disk |
| `store_collection_with_index` | - | `True` | Store page images inside the index (required for retrieval) |

### Visual RAG Configuration (`VisualRAGConfig`)

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `search_top_k` | `SEARCH_TOP_K` | `5` | Default number of pages to retrieve per query |
| `enable_grounding` | `ENABLE_GROUNDING` | `true` | Ask model to use visual grounding for accuracy |
| `image_dpi` | `IMAGE_DPI` | `200` | DPI for PDF page screenshots |

### Generation Configuration (`GenerationConfig`)

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `temperature` | `TEMPERATURE` | `0.3` | LLM sampling temperature |
| `top_p` | `TOP_P` | `0.85` | Nucleus sampling threshold |
| `max_output_tokens` | `MAX_TOKENS` | `4096` | Max tokens per response |
| `enable_auto_continuation` | - | `True` | Auto-continue if response hits token limit |
| `max_continuations` | - | `5` | Max number of auto-continuations |

### Cache Configuration (`CacheConfig`)

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `enabled` | `CACHE_ENABLED` | `true` | Enable/disable Redis caching |
| `query_ttl` | - | `3600` (1h) | TTL for cached search results |
| `response_ttl` | - | `1800` (30m) | TTL for cached LLM responses |

### Routing Configuration (`RoutingConfig`)

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `enabled` | `ROUTING_ENABLED` | `true` | Enable intelligent query routing |
| `page_specific_search_k` | - | `3` | Pages to retrieve for page-specific queries |
| `summarization_search_k` | - | `8` | Pages to retrieve for summarization |
| `factual_search_k` | - | `3` | Pages to retrieve for factual lookups |
| `document_specific_search_k` | - | `15` | Pages to retrieve for document-specific queries |

### Traffic Configuration (`TrafficConfig`)

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `model_rpm` | `MODEL_RPM` | `60` | Rate limit (requests per minute) |
| `failure_threshold` | `MODEL_FAILURE_THRESHOLD` | `5` | Failures before circuit breaker opens |
| `recovery_timeout` | `MODEL_RECOVERY_TIMEOUT` | `20.0` | Seconds before circuit breaker retries |
| `model_timeout` | `MODEL_TIMEOUT` | `120.0` | Request timeout in seconds |
| `health_check_interval` | - | `30.0` | Seconds between health checks |

### Performance Configuration (`PerformanceConfig`)

| Setting | Default | Description |
|---------|---------|-------------|
| `model_concurrent_limit` | `2` | Max concurrent requests to vLLM |

---

## 6. Module-by-Module Breakdown

### 6.1 `src/app.py` — Entry Point

**Purpose:** Chainlit web application. Handles session lifecycle, commands, file uploads, and message routing.

**Key handlers:**

| Handler | Trigger | What It Does |
|---------|---------|--------------|
| `@cl.on_chat_start` | New session | Loads existing index from disk if available, initializes session state (memory, file list) |
| `@cl.on_message` | User message | Routes to commands, file upload, or RAG generation |
| `@cl.on_stop` | Stop button | Sets cancellation flag so streaming stops |
| `@cl.on_chat_end` | Session closes | Logs that the index is preserved on disk |

**Commands:**

| Command | Action |
|---------|--------|
| `/clear` | Deletes the entire index, clears cache and memory |
| `/files` | Lists all uploaded files with page counts |
| `/stats` | Shows index stats (pages, documents) and cache memory usage |
| `/debug` | Shows index name, page count, and document names |
| `/health` | Pings vLLM server and lists available models |

**Session state** (stored in `cl.user_session`):

| Key | Type | Description |
|-----|------|-------------|
| `index_name` | `str` | Always `"documents"` (shared persistent index) |
| `files` | `list[dict]` | List of uploaded files with name, pages, type |
| `memory` | `ConversationMemory` | Conversation history for this session |
| `cancelled` | `bool` | Flag to stop streaming |
| `has_documents` | `bool` | Whether any documents are indexed |

**Important:** All sessions share a **single persistent index** called `"documents"`. The index survives container restarts because it's stored on a Docker volume mount (`persistent/indexes/`).

---

### 6.2 `src/config/settings.py` — Configuration

**Purpose:** Centralized configuration using Python dataclasses. All settings in one place.

**Pattern:** Singleton via `get_config()`. First call creates the `Config` instance, subsequent calls return the same one.

**System prompt** is defined inline (lines 159-201). It instructs the model to:
- Analyze page images strictly (no outside knowledge)
- Cite sources as `[Page X, DocumentName]`
- Handle page-specific and document-specific requests
- Format responses directly without filler phrases

---

### 6.3 `src/rag/pipeline.py` — Core RAG Pipeline

**Purpose:** The main response generation flow. This is where everything comes together.

**Function: `generate_response(query, index_name, memory, msg)`**

Step-by-step flow:

1. **Route the query** — Calls `route_query()` to classify intent and get parameters
2. **Determine search strategy** — Adjusts `search_top_k` based on intent type
3. **Retrieve pages** — Three paths:
   - **Page-specific** (e.g., "What's on page 5?"): Direct page lookup via `store.get_page_by_number()`
   - **Document-specific** (e.g., "What's in report.pdf?"): Gets all pages from that document via `store.get_document_pages()`, capped at 8
   - **General**: Semantic search via `store.search(query, top_k=...)`
4. **Build messages** — Constructs OpenAI-format messages with page images, conversation history, and grounding instructions
5. **Stream response** — Sends to Qwen3-VL via vLLM, streams tokens back to the UI
6. **Auto-continuation** — If the model hits the token limit (`finish_reason="length"`), automatically continues up to 5 times
7. **Save to memory** — Stores the clean response (box tags stripped) in conversation history

**Class: `BoxTagFilter`**

A real-time streaming filter that strips `<box>...</box>` tags from the model's output before displaying to the user. The model uses these internally for visual grounding but they shouldn't appear in the response.

How it works:
- Maintains a buffer of incoming tokens
- When it encounters `<box>`, it enters "in tag" mode and suppresses output
- When it encounters `</box>`, it exits and resumes normal output
- Handles partial tag matches across chunk boundaries

**Function: `build_visual_messages()`**

Constructs the message array sent to Qwen3-VL:

```
[
  { role: "system", content: system_prompt },
  { role: "user", content: [
      "[Page 1 from report.pdf]",
      <image: base64 PNG>,
      "[Page 3 from report.pdf]",
      <image: base64 PNG>,
      "Recent conversation:\n...",
      "Question: <user's query>",
      "Use visual grounding internally..."
  ]}
]
```

Conversation history is text-only (last 2 turns, assistant responses truncated to 200 chars).

---

### 6.4 `src/rag/router.py` — Query Router

**Purpose:** Classifies user queries by intent and determines optimal retrieval/generation parameters.

**How it works:** Pattern-based regex matching. No LLM calls — it's pure regex for speed.

**Query intents (in priority order):**

| Intent | Example | search_top_k | Temperature | Behavior |
|--------|---------|:---:|:---:|----------|
| `PAGE_SPECIFIC` | "What's on page 5?" | 3 | 0.2 | Direct page lookup |
| `DOCUMENT_SPECIFIC` | "What's in OIA.pdf?" | 15 | 0.3 | Filter by document name |
| `VISUAL_CONTENT` | "Describe the chart" | 8 | 0.3 | Visual analysis focus |
| `SUMMARIZATION` | "Summarize the document" | 8 | 0.3 | Broad retrieval |
| `COMPARISON` | "Compare X and Y" | 16 | 0.2 | Side-by-side comparison |
| `LIST_EXTRACTION` | "List all items" | 15 | 0.1 | Structured list format |
| `MULTI_HOP` | "First X, then Y" | 10 | 0.3 | Step-by-step reasoning |
| `ANALYTICAL` | "Why did X happen?" | 12 | 0.4 | Detailed reasoning |
| `DEFINITION` | "What is X?" | 5 | 0.2 | Focused retrieval |
| `FACTUAL_LOOKUP` | Short queries (<=8 words) | 6 | 0.1 | Direct answer |
| `GENERAL` | Everything else | default | 0.3 | Fallback |

**`RoutingDecision` dataclass fields:**

- `intent` — The classified QueryIntent
- `retrieval_strategy` — How to search (PAGE_LOOKUP, SEMANTIC_SEARCH, BROAD_RETRIEVAL, etc.)
- `generation_strategy` — How to format output (DIRECT_ANSWER, SUMMARY, COMPARISON_TABLE, etc.)
- `top_k` — Pages to retrieve
- `temperature` — Generation temperature
- `max_tokens` — Max response length
- `system_prompt_modifier` — Extra instructions appended to the system prompt
- `page_filter` — Extracted page number (for PAGE_SPECIFIC)
- `document_filter` — Extracted document name (for DOCUMENT_SPECIFIC)
- `confidence` — 0.0-1.0 confidence score

**`build_enhanced_prompt()`** — Appends `system_prompt_modifier` from routing decision to the base system prompt.

---

### 6.5 `src/storage/visual_store.py` — Byaldi Index Manager

**Purpose:** Wraps the Byaldi library to manage visual document indexes.

**Class: `VisualStore`** (singleton via `get_visual_store()`)

**Core methods:**

| Method | Description |
|--------|-------------|
| `initialize()` | Pre-loads ColQwen2 model at startup |
| `create_index(index_name, file_path, file_name)` | Creates a new index from the first document |
| `add_to_index(index_name, file_path, file_name)` | Adds a document to an existing index |
| `search(index_name, query, top_k, document_filter)` | Semantic visual search, returns `SearchResults` |
| `get_page_by_number(index_name, page_number)` | Direct page lookup by 1-indexed page number |
| `get_document_pages(index_name, doc_name)` | Get all pages from a specific document |
| `delete_index(index_name)` | Deletes index from disk and reloads model |
| `load_existing_index(index_name)` | Loads a persisted index from disk |
| `save_file_metadata(index_name, file_list)` | Saves file list as JSON alongside the index |
| `load_file_metadata(index_name)` | Loads file list from JSON |

**Data structures:**

```python
@dataclass
class PageResult:
    page_number: int      # 1-indexed
    document_name: str    # e.g., "report.pdf"
    score: float          # Similarity score from ColQwen2
    image_base64: str     # Page image as base64-encoded PNG

@dataclass
class SearchResults:
    results: List[PageResult]
    query: str
    total_pages_searched: int
```

**How document name resolution works (`_get_document_name`):**

Byaldi assigns each document a numeric `doc_id` (0, 1, 2...) in the order they were indexed. The `_active_indexes` dict maintains a list of document names in the same order, so `doc_id` is used as an index into that list.

**Index persistence:**
- Byaldi stores its index files at `{index_path}/{index_name}/`
- File metadata is saved as `{index_path}/{index_name}/file_metadata.json`
- On session start, both are loaded from disk

---

### 6.6 `src/processing/file_processor.py` — File Upload Handler

**Purpose:** Processes uploaded files and indexes them into Byaldi.

**Function: `process_files(files, index_name, file_list, is_first_upload)`**

| File Type | Processing |
|-----------|-----------|
| **PDF** | Indexed directly by Byaldi (it screenshots pages internally) |
| **Images** (PNG, JPG, JPEG, WebP) | Indexed as single-page documents |
| **DOCX** | Converted to PDF via `_convert_docx_to_pdf()`, then indexed |
| **TXT** | Rendered to PDF via `_convert_txt_to_pdf()`, then indexed |

**First upload vs subsequent:** The first document uses `store.create_index()` which creates a new Byaldi index. All subsequent documents use `store.add_to_index()` which appends to the existing index.

**Format converters:**
- `_convert_docx_to_pdf()` — Extracts text from DOCX paragraphs, renders to PDF
- `_convert_txt_to_pdf()` — Reads text file, renders to PDF
- `_text_to_pdf()` — Shared helper that renders text into a multi-page PDF using PyMuPDF with A4 dimensions, 11pt font, word wrapping at 80 chars

---

### 6.7 `src/rag/cache.py` — Redis Caching

**Purpose:** Caches search results and LLM responses to avoid redundant computation.

**Class: `RedisCache`** (singleton via `get_cache()`)

**Cache keys and TTLs:**

| Cache Type | Key Pattern | TTL | What's Cached |
|-----------|-------------|-----|---------------|
| Query results | `rag:query:{index_name}:{sha256(query)[:16]}` | 1 hour | Byaldi search results (serialized as JSON) |
| LLM responses | `rag:response:{sha256(query+context)[:16]}` | 30 min | Generated text responses |

**Note:** The query cache is not currently used in the pipeline flow (`pipeline.py` does not call `cache.get_query_result()`). It's available for future use.

**Methods:**

| Method | Description |
|--------|-------------|
| `get_query_result(query, index_name)` | Look up cached search results |
| `set_query_result(query, index_name, results)` | Cache search results |
| `get_response(query, context_hash)` | Look up cached LLM response |
| `set_response(query, context_hash, response)` | Cache LLM response |
| `clear_index_cache(index_name)` | Delete all cache entries for an index |
| `get_stats()` | Redis memory usage, key count, hit/miss ratio |

**Connection handling:** Connects on initialization. If Redis is unavailable, caching is silently disabled (`self._connected = False`) and all get/set methods return None/False.

---

### 6.8 `src/rag/memory.py` — Conversation History

**Purpose:** Maintains per-session conversation history.

**Class: `ConversationMemory`**

- Stores last 10 `(user_query, assistant_response)` turns
- Tracks mentioned page numbers (regex-extracted from user messages)
- `get_history(n)` returns the last `n` turns (used with `n=2` in the pipeline)
- `clear()` resets everything

**In-memory only.** History is lost when the session ends or the server restarts.

---

### 6.9 `src/rag/traffic_controller.py` — Traffic Management

**Purpose:** Rate limiting, circuit breaker, and health monitoring for the vLLM server.

**Note:** This module is defined and available but is **not actively used** in the current pipeline flow. The pipeline uses its own simple semaphore for concurrency control. This module is available for integration if needed.

**Features:**

| Feature | Description |
|---------|-------------|
| **Rate limiting** | 60 requests/minute sliding window |
| **Circuit breaker** | Opens after 5 consecutive failures, auto-recovers after 20 seconds |
| **Health monitoring** | Periodic health checks via `GET /v1/models` |
| **Concurrency control** | Asyncio semaphore (configurable limit) |

**Server health states:**

| State | Meaning |
|-------|---------|
| `HEALTHY` | Server responding normally |
| `DEGRADED` | Server responding but with errors |
| `UNHEALTHY` | Server not responding |
| `UNKNOWN` | Not checked yet |

---

### 6.10 `src/rag/visual_reranker.py` — Visual Reranking

**Purpose:** Optional reranking step that uses Qwen3-VL to assess page relevance.

**Note:** This module is defined but **not called** in the current pipeline. It's available for integration.

**How it would work:**
1. Send all candidate page images to Qwen3-VL with a reranking prompt
2. Ask the model to rank pages by relevance to the query
3. Parse the ranking response (e.g., `RANKING: [Page 3], [Page 1], [Page 5]`)
4. Return the top-k reranked results

**Fallback:** If reranking fails or returns too few results, falls back to original search scores.

---

### 6.11 `src/processing/page_screenshotter.py` — PDF to Image

**Purpose:** Utility for converting PDF pages to PIL Images using PyMuPDF.

**Functions:**

| Function | Description |
|----------|-------------|
| `screenshot_pdf_pages(file_path, dpi)` | Convert all pages to images. Returns `[(page_num, PIL.Image)]` |
| `screenshot_single_page(file_path, page_number, dpi)` | Convert a single page (1-indexed) |
| `get_pdf_page_count(file_path)` | Get number of pages in a PDF |

**Note:** Byaldi handles its own page screenshots during indexing. This module is used by `file_processor.py` to get page counts and is available as a utility.

---

### 6.12 `src/utils/helpers.py` — Utilities

| Function | Description |
|----------|-------------|
| `count_tokens(text)` | Count tokens using tiktoken (`cl100k_base`). Fallback: `len(text) // 4` |
| `extract_page_numbers(text)` | Extract page numbers from text like "page 5" |

---

### 6.13 `src/utils/grounding.py` — Bounding Box Utilities

**Purpose:** Parse and draw bounding boxes from Qwen3-VL's visual grounding output.

Qwen3-VL uses a 0-1000 normalized coordinate system: `[ymin, xmin, ymax, xmax]`.

| Function | Description |
|----------|-------------|
| `parse_bounding_boxes(text)` | Extract bbox coordinates from model output (supports `<box>`, `bbox_2d`, and bare array formats) |
| `draw_bounding_boxes(image, boxes, labels, color, width)` | Draw boxes on a PIL Image, scaling from 0-1000 to pixel coords |
| `extract_grounded_response(text)` | Split model response into clean text + bounding boxes |

**Note:** Currently, bounding boxes are stripped from the output (not drawn). The grounding module is available if you want to render visual annotations on page images.

---

## 7. Data Flow & Request Lifecycle

### Document Upload Flow

```
User uploads file(s) via web UI
         |
    app.py: on_message (detects message.elements)
         |
    file_processor.process_files()
         |
         +-- PDF?  → store.create_index() or store.add_to_index()
         |           (Byaldi screenshots pages + creates ColQwen2 embeddings)
         |
         +-- Image? → store.create_index() or store.add_to_index()
         |           (Single page, direct indexing)
         |
         +-- DOCX?  → _convert_docx_to_pdf() → index the PDF
         |
         +-- TXT?   → _convert_txt_to_pdf() → index the PDF
         |
    store.save_file_metadata()  (persist file list as JSON)
         |
    Confirmation message sent to user
```

### Query Flow

```
User sends a text query
         |
    app.py: on_message
         |
    generate_response(query, index_name, memory, msg)
         |
    1. route_query(query)
         |   → Regex pattern matching
         |   → Returns RoutingDecision (intent, top_k, temperature, etc.)
         |
    2. Visual Search
         |   PAGE_SPECIFIC?  → store.get_page_by_number()
         |   DOC_SPECIFIC?   → store.get_document_pages() (capped at 8)
         |   Otherwise?      → store.search(query, top_k)
         |
    3. build_visual_messages()
         |   → System prompt + page images + conversation history + query
         |
    4. Stream from vLLM
         |   → AsyncOpenAI client, streaming=True
         |   → BoxTagFilter strips <box>...</box> tags in real-time
         |   → Tokens streamed to user via msg.stream_token()
         |
    5. Auto-continuation (if finish_reason == "length")
         |   → Builds text-only continuation messages (no images re-sent)
         |   → Up to 5 continuations
         |
    6. memory.add(query, clean_response)
```

---

## 8. Query Routing System

The router uses **regex pattern matching** (no LLM calls) to classify queries. Each intent type has associated regex patterns defined in `router.py`.

### Pattern Examples

**Page-specific** (`page\s*(\d+)`, `p\.\s*(\d+)`, etc.):
- "What's on page 5?" → `PAGE_SPECIFIC`, extracts page_filter=5
- "Summarize p. 10" → `PAGE_SPECIFIC`, extracts page_filter=10

**Document-specific** (`[\w\-\.]+\.(?:pdf|docx|...)`)
- "What's in report.pdf?" → `DOCUMENT_SPECIFIC`, extracts document_filter="report.pdf"

**Summarization** (`summar(y|ize)`, `overview`, `key points`, `tl;dr`):
- "Give me an overview" → `SUMMARIZATION`

**Visual content** (`chart`, `graph`, `table`, `figure`, `diagram`):
- "What does the chart show?" → `VISUAL_CONTENT`

### How Routing Affects Behavior

Each intent sets three things:
1. **Retrieval parameters** — How many pages to search, whether to filter by page/document
2. **Generation parameters** — Temperature, max tokens
3. **System prompt modifier** — Extra instructions like "Provide a direct, factual answer"

The modifiers are appended to the base system prompt under a `QUERY-SPECIFIC INSTRUCTIONS:` header.

---

## 9. Key Design Decisions

### Visual-First RAG (No Text Extraction)

Documents are never text-extracted. ColQwen2 creates embeddings from page images, and Qwen3-VL reads page images directly. This preserves all visual information but means:
- Search is based on visual similarity, not keyword matching
- The system requires more GPU memory (images are larger than text)
- It works great for visually rich documents but may be less precise for pure-text queries

### Single Persistent Index

All documents across all sessions share one index called `"documents"`. This means:
- Any user's uploads are visible to all users
- The index survives container restarts
- `/clear` deletes everything for everyone

### vLLM Runs on Host

vLLM runs directly on the host machine (not in Docker) because:
- Direct GPU access without Docker GPU overhead
- Easier to manage GPU memory allocation
- The RAG app container connects to it via `host.docker.internal:8005`

### 8-Page Image Limit

vLLM is started with `--limit-mm-per-prompt '{"image":8}'`, meaning a maximum of 8 images per request. Document-specific queries cap at 8 pages. This is a balance between context and memory.

### Auto-Continuation

When the model hits its max token limit, the pipeline automatically sends a continuation request (up to 5 times). Continuations use text-only messages (no images re-sent) to save bandwidth and processing.

### Box Tag Filtering

Qwen3-VL outputs `<box>` tags for visual grounding. These are stripped in real-time during streaming so users never see raw coordinates, but the model uses them internally for accuracy.

---

## 10. Infrastructure & Deployment

### Docker Compose Services

| Service | Container | Image | Port | Purpose |
|---------|-----------|-------|------|---------|
| `redis` | `rag_redis` | `redis:7-alpine` | 6379 | Caching (2GB max, LRU eviction) |
| `rag_app` | `rag_app` | Custom (Dockerfile) | 8080→8000 | Chainlit web app + Byaldi |

### Docker Volumes (Bind Mounts)

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `persistent/data` | `/workspace/data` | Uploads and processed files |
| `persistent/indexes` | `/workspace/data/indexes` | Byaldi index files |
| `persistent/huggingface` | `/workspace/huggingface` | Model weight cache |
| `persistent/redis` | `/data` | Redis RDB snapshots |
| `src/` | `/workspace/src:ro` | Source code (read-only, dev mode) |
| `.chainlit/` | `/workspace/.chainlit:ro` | Chainlit config |
| `public/` | `/workspace/public:ro` | Static assets |

### Resource Limits

- RAG App container: 16GB memory limit
- Redis: 2GB max memory with LRU eviction
- GPU: All NVIDIA GPUs passed to RAG App container (for ColQwen2)

### Dockerfile Layers

The Dockerfile uses layered `pip install` for optimal caching:

1. **PyTorch + CUDA 12.1** (largest, changes least)
2. **Byaldi + ColPali** (visual retrieval)
3. **ML/Tokenization** (transformers, tiktoken)
4. **Redis client**
5. **Document processing** (PyMuPDF, python-docx, Pillow)
6. **Web framework** (Chainlit, FastAPI, OpenAI client)
7. **Utilities** (numpy, pydantic, etc.)

ColQwen2 model weights are **pre-downloaded** during Docker build (`snapshot_download('vidore/colqwen2-v0.1')`).

### Chainlit Configuration

Key settings in `.chainlit/config.toml`:
- Session timeout: 1 hour
- File upload: enabled, max 20 files, max 500MB each, all file types accepted
- HTML rendering: disabled (security)
- LaTeX: disabled
- Allow origins: `["*"]` (no CORS restrictions)

---

## 11. Troubleshooting

### NVIDIA Runtime Not Found (most common)

**Error:**
```
✘ Container rag_app  Error response from daemon: unknown or invalid runtime name: nvidia
```

**Cause:** Docker doesn't know about the NVIDIA container runtime. This happens on a fresh machine or after a Docker reinstall.

**Fix:**
```bash
# 1. Configure the NVIDIA runtime for Docker
sudo nvidia-ctk runtime configure --runtime=docker

# 2. Restart Docker
sudo systemctl restart docker        # Linux (systemd)
# OR
sudo service docker restart           # WSL2 / older systems

# 3. Verify it worked
docker info | grep -i nvidia
# Should show "nvidia" in the Runtimes list

# 4. Re-run the stack
cd Docker
docker compose up -d
```

If `nvidia-ctk` is not installed:
```bash
# Install NVIDIA Container Toolkit first
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

### vLLM Server Won't Start

```bash
# Check logs
tail -f logs/vllm.log

# Common issues:
# - Not enough GPU memory: reduce VLLM_GPU_MEMORY in .env
# - Model not found: check VLLM_MODEL name, ensure HF_TOKEN is set
# - Port in use: change VLLM_PORT in .env
```

### RAG App Can't Connect to vLLM

```bash
# From inside the container, vLLM is at host.docker.internal:8005
# Test from host:
curl http://localhost:8005/v1/models

# If using WSL2, ensure Docker Desktop has "host.docker.internal" support
```

### ColQwen2 Model Loading Slow

First load downloads ~2GB of model weights. Subsequent loads use the cache at `persistent/huggingface/`. If the container was rebuilt, the Dockerfile pre-downloads the model.

### Index Not Found After Restart

Check that `persistent/indexes/documents/` exists and is not empty. The `file_metadata.json` file must also be present for document names to work.

### Redis Connection Failed

```bash
# Check Redis container
docker logs rag_redis

# Redis is optional - if it fails, caching is silently disabled
```

### Out of GPU Memory

- Reduce `VLLM_GPU_MEMORY` in `.env` (e.g., from `0.80` to `0.60`)
- Reduce `SEARCH_TOP_K` to send fewer images per request
- The 8-image limit per request is hardcoded in the vLLM startup flag

---

## 12. Known Limitations & Future Considerations

### Current Limitations

1. **No authentication** — Any visitor can access the web UI and all uploaded documents
2. **Single shared index** — All users share one index; no per-user isolation
3. **In-memory conversation history** — Lost on session end or restart
4. **HF_TOKEN in plaintext** — Stored in `.env` file
5. **No text search fallback** — Pure visual search may miss keyword-based queries
6. **8-page limit per request** — Large documents may not fit in a single query context
7. **Cache not integrated in pipeline** — `cache.py` is defined but not called from `pipeline.py`
8. **Traffic controller not integrated** — `traffic_controller.py` is defined but the pipeline uses its own semaphore
9. **Visual reranker not integrated** — `visual_reranker.py` is defined but not called from the pipeline

### Modules Ready for Integration

These modules are fully implemented and can be plugged into the pipeline:

- **`cache.py`** — Add cache lookups before search and before generation in `pipeline.py`
- **`traffic_controller.py`** — Replace the pipeline's simple semaphore with the full traffic controller
- **`visual_reranker.py`** — Add reranking after Byaldi search and before generation
