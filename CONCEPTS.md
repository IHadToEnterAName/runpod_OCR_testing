# Beginner's Guide to the Concepts Behind This Project

This document explains the core AI/ML concepts, tools, and techniques used in this codebase. If you're new to Python, machine learning, or building AI applications, start here before diving into the code.

---

## Table of Contents

1. [What Is an LLM?](#1-what-is-an-llm)
2. [Vision-Language Models (VLMs)](#2-vision-language-models-vlms)
3. [What Is vLLM?](#3-what-is-vllm)
4. [What Is RAG?](#4-what-is-rag)
5. [Visual RAG (What This Project Does)](#5-visual-rag-what-this-project-does)
6. [Embeddings & Vector Search](#6-embeddings--vector-search)
7. [ColQwen2 & Byaldi (Visual Retrieval)](#7-colqwen2--byaldi-visual-retrieval)
8. [Reranking](#8-reranking)
9. [Query Routing](#9-query-routing)
10. [Streaming (Token-by-Token Output)](#10-streaming-token-by-token-output)
11. [Map-Reduce Summarization](#11-map-reduce-summarization)
12. [OCR (Optical Character Recognition)](#12-ocr-optical-character-recognition)
13. [Caching (Redis)](#13-caching-redis)
14. [Circuit Breaker & Traffic Control](#14-circuit-breaker--traffic-control)
15. [Tokenization](#15-tokenization)
16. [Docker & Containerization](#16-docker--containerization)
17. [Async Programming (asyncio)](#17-async-programming-asyncio)
18. [How It All Fits Together](#18-how-it-all-fits-together)

---

## 1. What Is an LLM?

**LLM** stands for **Large Language Model**. Think of it as an extremely advanced autocomplete — it predicts the most likely next word given all the previous words, but at a scale so large that it can write essays, answer questions, summarize documents, and reason about problems.

### Key terms

- **Model**: The trained neural network (billions of parameters/numbers that were learned from text data)
- **Inference**: Using a trained model to generate output (as opposed to training it)
- **Prompt**: The input text you give the model
- **Tokens**: The pieces the model breaks text into (roughly ~4 characters per token in English). "Hello world" = 2 tokens
- **Context window**: The maximum number of tokens the model can "see" at once (input + output combined). Our model (Qwen3-VL) has a 32,768 token context window
- **Temperature**: Controls randomness. `0.0` = always pick the most likely word (deterministic), `1.0` = more creative/random. We use `0.3` (mostly factual, slightly varied)
- **max_tokens**: The maximum number of tokens the model will generate in its response

### Example

```
Prompt:  "What is the capital of France?"
Model:   "The capital of France is Paris."
```

The model doesn't "know" facts — it learned statistical patterns from training data that make it very good at producing correct-sounding answers.

---

## 2. Vision-Language Models (VLMs)

A regular LLM only understands text. A **Vision-Language Model** (VLM) understands both **text and images**.

This project uses **Qwen3-VL** — a VLM made by Alibaba. You can send it a photo of a document page along with a question, and it can:
- Read the text in the image
- Understand charts, tables, and diagrams
- Answer questions about what it sees
- Point to specific regions (bounding boxes)

### Why not just extract text?

Text extraction (OCR) loses formatting, tables, charts, and visual layout. A VLM "sees" the page exactly like a human would — tables stay as tables, charts stay as charts, and spatial relationships are preserved.

### In this project

When you ask a question about an uploaded PDF, the system sends **page images** (not extracted text) to Qwen3-VL. The model literally looks at screenshots of the PDF pages to answer your question.

---

## 3. What Is vLLM?

**vLLM** is an open-source **inference server** for LLMs. Think of it as the engine that actually runs the model.

### Why not just load the model in Python?

You could do `model = load("Qwen3-VL")` in Python, but:
- It would block your entire application while generating
- It can't handle multiple users at once
- It doesn't optimize memory or batching

vLLM solves all of this:
- **PagedAttention**: A memory management technique that reduces GPU memory waste (like virtual memory for GPUs)
- **Continuous batching**: Handles multiple requests simultaneously by batching them efficiently
- **OpenAI-compatible API**: Exposes the model as a REST API that looks like OpenAI's API, so you can use the same `openai` Python library

### In this project

vLLM runs on the **host machine** (with direct GPU access), and the web app inside Docker calls it over HTTP:

```
[Web App (Docker)] --HTTP--> [vLLM Server (Host)] --GPU--> [Qwen3-VL Model]
```

The app uses the `openai` Python library to talk to vLLM, because vLLM mimics the OpenAI API format. This means any code written for OpenAI's GPT-4 would work with our local Qwen3-VL just by changing the URL.

### Key vLLM flags we use

```bash
--gpu-memory-utilization 0.22    # Only use 22% of GPU memory
--max-model-len 32768            # Max context window
--limit-mm-per-prompt '{"image":8}'  # Max 8 images per request (hardware limit)
```

---

## 4. What Is RAG?

**RAG** stands for **Retrieval-Augmented Generation**. It's a technique that makes LLMs smarter by giving them relevant information before asking them to answer.

### The problem RAG solves

LLMs only know what they learned during training. They don't know about:
- Your private documents
- Recent events after their training cutoff
- Company-specific data

### How RAG works

```
Without RAG:
  User: "What were Q3 earnings?"
  LLM:  "I don't have access to your financial data." ❌

With RAG:
  1. User: "What were Q3 earnings?"
  2. System RETRIEVES relevant pages from the uploaded earnings report
  3. System sends: "Here are pages from the report: [pages]. Based on this, what were Q3 earnings?"
  4. LLM: "According to the Q3 report, earnings were $4.2B..." ✅
```

### The RAG pipeline

```
User Query → RETRIEVE relevant documents → AUGMENT the prompt with them → GENERATE answer
              ^^^^^^^^                      ^^^^^^^                        ^^^^^^^^
              (search step)                 (add context)                  (LLM answers)
```

---

## 5. Visual RAG (What This Project Does)

Traditional RAG extracts **text** from documents, then searches through text. This project does **Visual RAG** — it works with **page images** instead of text.

### Traditional RAG vs Visual RAG

| Step | Traditional RAG | Visual RAG (this project) |
|------|----------------|--------------------------|
| **Index** | Extract text → chunk → embed text | Screenshot pages → embed page images |
| **Search** | Text similarity search | Visual similarity search |
| **Answer** | Send retrieved text to LLM | Send retrieved page images to VLM |

### Why Visual RAG?

- **No text extraction errors** — OCR makes mistakes, especially with complex layouts
- **Tables and charts preserved** — A VLM can read a table from an image; text extraction often garbles tables
- **Layout matters** — The spatial arrangement of information on a page carries meaning
- **Works on any document** — Even scanned handwritten notes

### Trade-off

Visual RAG is more accurate but more expensive (images use many more tokens than text). That's why we limit to 8 images per request.

---

## 6. Embeddings & Vector Search

**Embeddings** are the foundation of how search works in RAG. An embedding converts something (text, an image, a document page) into a list of numbers (a "vector") that captures its meaning.

### Intuition

Imagine you could place every document page on a map where:
- Pages about "financial results" are clustered together
- Pages about "employee policies" are in another cluster
- Pages with charts about revenue are near the "financial results" cluster

An embedding model creates this "map" by converting each page into coordinates (a vector of ~128 numbers).

### How search works

```
1. At upload time:
   Each page → Embedding model → Vector [0.23, -0.71, 0.15, ...]
   Store all vectors in an index

2. At query time:
   "What were Q3 earnings?" → Embedding model → Vector [0.21, -0.68, 0.19, ...]
   Find the stored vectors closest to this query vector
   → Returns: Page 14, Page 15, Page 23 (the most relevant pages)
```

The "closeness" is measured by **cosine similarity** — how similar the angle between two vectors is.

---

## 7. ColQwen2 & Byaldi (Visual Retrieval)

### ColQwen2

**ColQwen2** is the embedding model we use. Unlike text embedding models (which only work on text), ColQwen2 creates embeddings from **page images**. It's part of the "ColPali" family of models.

How it works:
1. Takes a screenshot of a PDF page
2. Processes it through a vision transformer
3. Outputs a vector that captures the visual and textual content of the page

When a user asks a question, ColQwen2 also embeds the query text, then finds which page images have the most similar vectors.

### Byaldi

**Byaldi** is a Python library that wraps ColQwen2 and handles:
- Converting PDFs to page images automatically
- Creating and managing the vector index
- Running similarity search
- Persisting indexes to disk

### In this project (`src/storage/visual_store.py`)

```python
# At upload time:
store.create_index("session_123", "report.pdf", "report.pdf")
# Byaldi screenshots each page → ColQwen2 embeds them → stored in index

# At query time:
results = store.search("session_123", "Q3 earnings", top_k=3)
# ColQwen2 embeds the query → finds 3 most similar page images
# Returns: page numbers + similarity scores + base64 page images
```

---

## 8. Reranking

**Reranking** is a second-pass relevance check that improves search quality.

### Why rerank?

The initial search (ColQwen2) is fast but approximate. It's like doing a quick scan of a library shelf. Reranking is like actually opening the top results and reading them more carefully to confirm which are truly relevant.

### How our visual reranker works (`src/rag/visual_reranker.py`)

```
1. ColQwen2 search returns top 5 pages (fast, approximate)
2. Visual reranker sends all 5 page images + the query to Qwen3-VL
3. Qwen3-VL (the full VLM) looks at each page and ranks them by relevance
4. Pages are re-sorted by the VLM's ranking
5. Top 3 (or however many) are sent for the final answer
```

This is more expensive than the initial search (it uses the full VLM), but much more accurate. The VLM can understand nuance — for example, a page might contain the word "earnings" but be about a different quarter.

### Traditional reranking vs our approach

| Traditional | Ours |
|------------|------|
| Cross-encoder (text model) re-scores text chunks | Qwen3-VL (vision model) re-scores page images |
| Only understands text | Understands tables, charts, layout |

---

## 9. Query Routing

**Query routing** is like a receptionist that looks at your question and decides the best way to handle it.

### Why route queries? (`src/rag/router.py`)

Different questions need different strategies:
- "What's on page 5?" → Just look up page 5 directly (no need for search)
- "Summarize the document" → Need to look at many pages, not just 1-2
- "What is GDP?" → Factual lookup, needs precision
- "Compare section A and section B" → Needs content from multiple places

### How it works

The router uses **regex pattern matching** (not an LLM call — it's instant and free) to classify queries:

```python
# Simplified example of what the router does:
if "summarize" in query or "overview" in query:
    intent = SUMMARIZATION
    search_top_k = 8      # Look at more pages
    temperature = 0.3      # Stay factual
elif "page" in query and has_number:
    intent = PAGE_SPECIFIC
    search_top_k = 1       # Just get that page
    temperature = 0.1      # Very precise
elif "compare" in query:
    intent = COMPARISON
    search_top_k = 6       # Need content from both sides
```

### The routing decision

For each query, the router returns:
- **Intent**: What type of question is this?
- **Retrieval strategy**: How should we search? (page lookup, semantic search, broad retrieval)
- **Generation strategy**: How should the answer be structured? (direct answer, table, list)
- **Parameters**: How many pages to retrieve, what temperature to use

---

## 10. Streaming (Token-by-Token Output)

When you chat with ChatGPT and see words appearing one at a time, that's **streaming**. Without streaming, you'd wait for the entire response to be generated before seeing anything.

### How it works

```
Without streaming:
  User asks question → [waiting 15 seconds...] → Full response appears at once

With streaming:
  User asks question → "The" → "capital" → "of" → "France" → "is" → "Paris."
  (each token appears as it's generated, ~50ms apart)
```

### In this project

vLLM supports streaming natively. The app uses the OpenAI client's streaming mode:

```python
stream = await client.chat.completions.create(
    model="Qwen3-VL",
    messages=[...],
    stream=True,  # ← This enables streaming
)

async for chunk in stream:
    token = chunk.choices[0].delta.content
    await msg.stream_token(token)  # Send to user's browser immediately
```

### BoxTagFilter

Qwen3-VL sometimes outputs bounding box tags like `<box>(123,456),(789,012)</box>` mixed into its text response. The `BoxTagFilter` is a streaming filter that strips these tags in real-time so the user sees clean text, while the system can still use the coordinates for visual grounding.

---

## 11. Map-Reduce Summarization

### The problem

The VLM can only see **8 images at a time** (vLLM hardware limit). For a 1000-page document, that's less than 1% coverage — not enough for a good summary.

### The solution: Map-Reduce

This is a pattern borrowed from big data (Google's MapReduce paper). The idea: break a huge task into small pieces, process each piece independently, then combine the results.

```
MAP PHASE:
  Pages 1-30    → Chunk 1 → "Summary of chunk 1"
  Pages 31-60   → Chunk 2 → "Summary of chunk 2"
  Pages 61-90   → Chunk 3 → "Summary of chunk 3"
  ...           → ...     → ...
  Pages 970-1000 → Chunk 34 → "Summary of chunk 34"

  (Each chunk is summarized by a TEXT-ONLY vLLM call — no images needed,
   so the 8-image limit doesn't apply)

REDUCE PHASE:
  All 34 partial summaries → Combined into one prompt → Final comprehensive summary
  (Streamed to the user token-by-token)
```

### Why text, not images?

For summarization of large docs, we extract the **text** from PDF pages instead of sending images. This is because:
1. We need to cover ALL pages, not just 8
2. Text-only calls have no image limit
3. Text extraction is good enough for summarization (you don't need to "see" a chart to summarize the document's main points)

### When does this activate?

Only when **both** conditions are true:
1. The query router classifies the question as **SUMMARIZATION**
2. The document has more than **50 pages** (configurable)

For short documents (<=50 pages) or non-summarization queries, the visual pipeline is used as normal.

---

## 12. OCR (Optical Character Recognition)

**OCR** converts images of text into actual text characters. It's how computers "read" scanned documents.

### PyMuPDF vs pytesseract

This project uses two text extraction methods:

| Method | When used | How it works |
|--------|-----------|-------------|
| **PyMuPDF** (`fitz`) | First attempt for all PDFs | Reads the text layer embedded in the PDF (fast, accurate for digital PDFs) |
| **pytesseract** | Fallback only | Uses Google's Tesseract OCR engine to recognize text from a rendered image of the page (slower, needed for scanned documents) |

### When does OCR happen?

Only during **long-document summarization** (map-reduce path). The flow:

```
For each page in the PDF:
  1. Try PyMuPDF's page.get_text()
  2. If result has < 50 characters → page is probably scanned/image-based
  3. Fall back to pytesseract: render page as image → OCR → extract text
```

For regular Q&A, the system sends **page images** directly to Qwen3-VL — no text extraction needed.

---

## 13. Caching (Redis)

**Caching** stores results so you don't have to recompute them. **Redis** is a fast in-memory database commonly used for caching.

### What we cache (`src/rag/cache.py`)

| Cache | What's stored | TTL (time-to-live) | Why |
|-------|--------------|---------------------|-----|
| **Query cache** | Search results (which pages are relevant to a query) | 1 hour | Avoids re-running ColQwen2 search for the same question |
| **Response cache** | The LLM's full answer | 30 minutes | Avoids re-generating the same answer |

### How the cache key works

```python
# The query is hashed to create a unique key
key = sha256("What were Q3 earnings?" + "session_123")
# → "cache:query:a3f8b2c1..."

# If this key exists in Redis, return the cached result instantly
# If not, run the search/generation and store the result
```

### Why different TTLs?

- Query cache (1 hour): Search results don't change unless the document changes
- Response cache (30 minutes): Shorter because the user might want a slightly different answer if they ask again, or conversation context may have changed

---

## 14. Circuit Breaker & Traffic Control

### Rate Limiting (`src/rag/traffic_controller.py`)

Prevents overwhelming the vLLM server with too many requests:

```
If requests_this_minute > 60:
    Wait before sending the next request
```

### Circuit Breaker

A safety pattern from electrical engineering. If the vLLM server starts failing, stop sending requests to give it time to recover:

```
State: CLOSED (normal) → requests flow through
  ↓ (5 consecutive failures)
State: OPEN (tripped) → all requests immediately fail with a friendly error
  ↓ (wait 20 seconds)
State: HALF-OPEN (testing) → let one request through
  ↓ (if it succeeds)
State: CLOSED (recovered) → back to normal
```

Without a circuit breaker, if vLLM crashes, every user request would wait for the full timeout (120 seconds) before failing. With it, requests fail instantly with "service temporarily unavailable."

### Concurrency Control

Uses Python's `asyncio.Semaphore` to limit how many requests hit vLLM simultaneously:

```python
semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests

async with semaphore:
    # Only 5 requests can be here at once
    # Request #6 waits until one of the 5 finishes
    response = await client.chat.completions.create(...)
```

---

## 15. Tokenization

**Tokenization** is how text is split into pieces (tokens) that the model can process.

### Why not just split by words?

Models don't work with words directly. They use **subword tokens** — common words are one token, uncommon words are split into pieces:

```
"Hello"       → ["Hello"]           (1 token)
"unhappiness" → ["un", "happiness"] (2 tokens)
"Qwen3"       → ["Q", "wen", "3"]  (3 tokens)
```

### In this project (`src/utils/helpers.py`)

We use **tiktoken** (OpenAI's tokenizer) to count tokens accurately. This is important for:
- **Chunking text** for map-reduce: Each chunk must fit within the token budget
- **Checking if a response hit the limit**: If the model generated exactly `max_tokens` tokens, it probably got cut off
- **Estimating costs**: API pricing is per-token

```python
from utils.helpers import count_tokens

text = "This is a sample document..."
token_count = count_tokens(text)  # → 6
```

---

## 16. Docker & Containerization

**Docker** packages an application and all its dependencies into a "container" — a lightweight, isolated environment that runs the same way everywhere.

### Why Docker?

Without Docker:
```
"It works on my machine" → Install Python 3.11, install 30+ packages with exact versions,
configure system libraries, set up Redis, configure networking...
```

With Docker:
```
docker compose up → Everything runs in pre-configured containers
```

### This project's Docker setup

```
┌─────────────────────────────────────────────┐
│ Host Machine (your server)                  │
│                                             │
│  ┌─────────────┐     ┌─────────────────┐   │
│  │ vLLM Server │     │  Docker Network  │   │
│  │ (port 8005) │◄────┤                  │   │
│  │ [GPU]       │     │  ┌───────────┐   │   │
│  └─────────────┘     │  │  rag_app  │   │   │
│                      │  │ (Chainlit │   │   │
│                      │  │  + Byaldi)│   │   │
│                      │  └───────────┘   │   │
│                      │  ┌───────────┐   │   │
│                      │  │   Redis   │   │   │
│                      │  │  (cache)  │   │   │
│                      │  └───────────┘   │   │
│                      └─────────────────┘   │
└─────────────────────────────────────────────┘
```

- **rag_app**: The web application (Chainlit UI + Byaldi indexing + all RAG logic)
- **Redis**: The cache database
- **vLLM**: Runs directly on the host (not in Docker) for direct GPU access

### Key Docker concepts

- **Dockerfile**: Recipe for building the container image (install packages, copy code)
- **docker-compose.yml**: Defines how multiple containers work together (networking, volumes, environment)
- **Volume mounts**: Shared folders between host and container (so uploaded files persist)
- **`host.docker.internal`**: Special hostname that lets containers reach the host machine (used for vLLM)

---

## 17. Async Programming (asyncio)

**Async** programming lets Python do multiple things "at the same time" without threads. This is critical for a web server handling multiple users.

### The problem

```python
# Synchronous (blocking):
response1 = call_vllm(query1)     # Wait 5 seconds...
response2 = call_vllm(query2)     # Wait 5 more seconds...
# Total: 10 seconds, and user 2 had to wait for user 1

# Asynchronous (non-blocking):
task1 = call_vllm(query1)         # Start call 1
task2 = call_vllm(query2)         # Start call 2 immediately
response1, response2 = await asyncio.gather(task1, task2)
# Total: ~5 seconds, both ran in parallel
```

### Key async concepts used in this project

- **`async def`**: Defines a function that can be paused and resumed
- **`await`**: Pauses the function until the result is ready (while other code runs)
- **`asyncio.gather()`**: Runs multiple async tasks concurrently (used in map-reduce)
- **`asyncio.Semaphore`**: Limits how many tasks run concurrently
- **`async for`**: Iterates over a stream (used for token streaming)

### In this project

Almost everything is async because the app needs to:
- Handle multiple users simultaneously
- Stream tokens while other requests are processing
- Run multiple map-reduce chunks concurrently
- Check cache and search in parallel

---

## 18. How It All Fits Together

Here's the complete flow when a user asks a question:

```
User types: "What were the Q3 revenue figures?"
    │
    ▼
┌─ QUERY ROUTER ─────────────────────────────────┐
│ Regex pattern matching → FACTUAL_LOOKUP intent  │
│ search_top_k=3, temperature=0.1                 │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─ CACHE CHECK ───────────────────────────────────┐
│ Hash the query → check Redis                    │
│ Cache hit? → return cached result immediately   │
│ Cache miss? → continue to search                │
└─────────────────────────────────────────────────┘
    │ (cache miss)
    ▼
┌─ TRAFFIC CONTROLLER ───────────────────────────┐
│ Check rate limit (< 60 RPM?)                   │
│ Check circuit breaker (is vLLM healthy?)       │
│ Acquire semaphore (concurrency slot)           │
└────────────────────────────────────────────────┘
    │
    ▼
┌─ VISUAL SEARCH (Byaldi/ColQwen2) ─────────────┐
│ Embed query → find top 3 most similar pages    │
│ Returns: page images + similarity scores       │
└────────────────────────────────────────────────┘
    │
    ▼
┌─ VISUAL RERANKER (Qwen3-VL) ──────────────────┐
│ Send all 3 page images + query to VLM          │
│ VLM re-ranks by actual relevance               │
│ Top pages confirmed and re-ordered             │
└────────────────────────────────────────────────┘
    │
    ▼
┌─ GENERATION (Qwen3-VL via vLLM) ──────────────┐
│ System prompt + conversation history            │
│ + top page images + user query                  │
│ → Stream response token-by-token               │
│ → BoxTagFilter strips bounding box tags        │
│ → Tokens appear in user's browser in real-time │
└────────────────────────────────────────────────┘
    │
    ▼
┌─ POST-PROCESSING ─────────────────────────────┐
│ Cache the response in Redis                    │
│ Save to conversation memory                    │
│ Extract any bounding boxes for grounding       │
└────────────────────────────────────────────────┘
```

### For long-document summarization (>50 pages)

```
User types: "Summarize this document"
    │
    ▼
Router: SUMMARIZATION intent
    │
    ▼
Pipeline checks: 500 pages > 50 threshold → TEXT PATH
    │
    ▼
PyMuPDF extracts text from all 500 pages
  (pytesseract OCR fallback for scanned pages)
    │
    ▼
Text chunked into ~17 groups of ~3000 tokens each
    │
    ▼
MAP: 3 chunks summarized concurrently × 6 batches
  → 17 partial summaries
    │
    ▼
REDUCE: All partials combined → final summary streamed to user
```

---

## Glossary

| Term | Meaning |
|------|---------|
| **Async** | Non-blocking code execution (multiple tasks share one thread) |
| **Byaldi** | Python library that wraps ColQwen2 for visual document indexing |
| **Cache** | Stored results to avoid recomputing (Redis in our case) |
| **Circuit breaker** | Safety mechanism that stops requests to a failing service |
| **ColQwen2** | Visual embedding model that converts page images to vectors |
| **Context window** | Maximum tokens a model can process at once |
| **Docker** | Tool for packaging apps in isolated containers |
| **Embedding** | A list of numbers representing the meaning of text/images |
| **Inference** | Running a trained model to get predictions |
| **LLM** | Large Language Model (text AI) |
| **Map-Reduce** | Pattern: split big task into small pieces, process each, combine results |
| **OCR** | Optical Character Recognition (reading text from images) |
| **RAG** | Retrieval-Augmented Generation (search + LLM) |
| **Reranking** | Second-pass relevance scoring to improve search quality |
| **Semaphore** | Concurrency limiter (max N things at once) |
| **Streaming** | Sending output token-by-token instead of all at once |
| **Temperature** | Controls randomness of model output (0=deterministic, 1=creative) |
| **Token** | The basic unit of text for a model (~4 characters) |
| **TTL** | Time-to-live (how long before a cache entry expires) |
| **Vector** | A list of numbers (used for embeddings and similarity search) |
| **VLM** | Vision-Language Model (AI that understands both text and images) |
| **vLLM** | High-performance inference server for LLMs |
