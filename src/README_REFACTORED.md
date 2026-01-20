# Refactored RAG System - Code Organization

## 📁 Project Structure

```
src_refactored/
├── app.py                          # Main Chainlit application
├── config/
│   ├── __init__.py
│   └── settings.py                 # All configuration (models, tokens, etc.)
├── processing/
│   ├── __init__.py
│   ├── document_extractor.py       # PDF/DOCX/TXT extraction
│   ├── file_processor.py           # File processing orchestration
│   └── vision.py                   # Image analysis with vision model
├── rag/
│   ├── __init__.py
│   ├── embeddings.py               # Query & document embedding
│   ├── memory.py                   # Conversation history
│   └── pipeline.py                 # Core RAG generation logic
├── storage/
│   ├── __init__.py
│   └── vector_store.py             # ChromaDB operations
└── utils/
    ├── __init__.py
    └── helpers.py                  # Token counting, CJK filter, etc.
```

## 🎯 Design Principles

### 1. **All Original Logic Preserved**
Every function, parameter, and behavior from your original code is preserved exactly. No logic changes.

### 2. **Clean Separation of Concerns**
- **config/**: All settings in one place
- **processing/**: Document extraction and vision
- **rag/**: Core RAG logic (embeddings, generation, memory)
- **storage/**: Vector database operations
- **utils/**: Helper functions

### 3. **Easy to Navigate**
Each module has a clear responsibility. To modify something:
- Change settings → `config/settings.py`
- Fix PDF extraction → `processing/document_extractor.py`
- Adjust generation → `rag/pipeline.py`

## 🔧 Configuration System

All configuration is centralized in `config/settings.py`:

```python
from config.settings import get_config

config = get_config()

# Access settings
config.models.vision_model       # "Qwen/Qwen2.5-VL-3B-Instruct"
config.models.reasoning_model    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
config.tokens.max_input_tokens   # 13,384
config.chunking.chunk_size       # 600
config.generation.temperature    # 0.3
```

All environment variables are read here, with sensible defaults.

## 📝 Module Details

### app.py
**Main application entry point**
- Chainlit handlers (`on_chat_start`, `on_message`, etc.)
- Commands (`/clear`, `/files`, `/stats`, `/test`, `/debug`)
- Session management
- File upload handling

### config/settings.py
**Centralized configuration**
- `ModelConfig`: vLLM endpoints and model names
- `TokenConfig`: Token limits (matching your 16K context)
- `ChunkingConfig`: Chunk size (600), overlap (100)
- `PerformanceConfig`: Concurrency limits, batch sizes
- `GenerationConfig`: Temperature, top_p, thinking filter
- `DatabaseConfig`: ChromaDB and Redis settings
- `SYSTEM_PROMPT`: Your exact system prompt

### processing/document_extractor.py
**Document content extraction**
- `extract_pdf_pages()`: Text from PDF pages
- `extract_pdf_images()`: Images from PDF
- `extract_docx()`: Text from DOCX
- `extract_txt()`: Text from TXT files

All exactly as in your original code.

### processing/vision.py
**Image analysis**
- `resize_image()`: Resize to 384px
- `image_to_base64()`: Convert for API
- `analyze_image()`: Call vision model

Uses your exact vision server configuration.

### processing/file_processor.py
**File processing orchestration**
- `process_files()`: Main orchestration function
- Handles PDF, DOCX, TXT, and image files
- Chunks text, embeds, stores in ChromaDB
- Progress updates to user

All your original logic for batch processing preserved.

### rag/embeddings.py
**Text embedding**
- `embed_documents()`: For storage (adds "search_document:" prefix)
- `embed_query()`: For retrieval (adds "search_query:" prefix)

Your exact embedding logic with nomic-embed-text-v1.5.

### rag/memory.py
**Conversation memory**
- `ConversationMemory`: Class for managing history
- `add()`: Add turn to history (keeps last 10)
- `get_history()`: Get last N turns
- `clear()`: Reset history

Exactly as in your original code.

### rag/pipeline.py
**Core RAG generation**
- `generate_response()`: Main generation function
- Retrieval → Context formatting → Message building → Streaming generation
- Thinking filter integration
- CJK filtering
- Memory updates

Your complete original generation logic preserved.

### storage/vector_store.py
**ChromaDB operations**
- `retrieve_chunks()`: Semantic search with page filtering
- `create_collection()`: Create new collection
- `delete_collection()`: Delete collection
- `add_chunks_to_collection()`: Batch add chunks

All your original retrieval logic including page filtering.

### utils/helpers.py
**Utility functions**
- `count_tokens()`: Token counting with tiktoken
- `filter_cjk()`: Remove CJK characters
- `clean_thinking()`: Remove `<think>` tags
- `ThinkingFilter`: Streaming thinking filter
- `extract_page_numbers()`: Extract page numbers from queries
- `format_context()`: Format chunks for context

All helper functions from your original code.

## 🚀 Running the Refactored Code

### 1. Setup Environment

```bash
# Set Python path
export PYTHONPATH=/path/to/src_refactored:$PYTHONPATH
```

### 2. Start vLLM Servers (Your Exact Configuration)

**Terminal 1 - Vision (Port 8006):**
```bash
source /workspace/venv/bin/activate
VLLM_USE_V1=0 vllm serve "Qwen/Qwen2.5-VL-3B-Instruct" \
    --port 8006 \
    --gpu-memory-utilization 0.3 \
    --max-model-len 4096 \
    --limit-mm-per-prompt '{"image":12}' \
    --enforce-eager \
    --trust-remote-code
```

**Terminal 2 - Reasoning (Port 8005):**
```bash
source /workspace/venv/bin/activate
VLLM_USE_V1=0 vllm serve "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
    --port 8005 \
    --gpu-memory-utilization 0.54 \
    --max-model-len 16384 \
    --enforce-eager \
    --enable-prefix-caching
```

### 3. Run Application

```bash
cd src_refactored
chainlit run app.py --host 0.0.0.0 --port 8000
```

## 🔍 Key Features Preserved

### 1. Token Management
- Model max: 16,384 tokens (DeepSeek R1)
- Output max: 2,048 tokens
- Input max: 13,384 tokens (16,384 - 2,048 - 1,000 safety)
- Vision: 512 tokens per image

### 2. Chunking Strategy
- Chunk size: 600 characters
- Overlap: 100 characters
- Max context chunks: 12

### 3. Thinking Tag Filtering
- Enabled by default
- 30-second timeout
- Streaming filter implementation
- Fallback to post-processing

### 4. CJK Filtering
- Filters: Chinese, Japanese, Korean characters
- Applied to all generated text
- Your exact character ranges preserved

### 5. Page-Aware Retrieval
- Detects "page N" in queries
- Filters ChromaDB results by page
- Falls back to semantic search

### 6. Conversation Memory
- Keeps last 10 turns
- Tracks mentioned pages
- Included in context (last 2-3 turns)

### 7. Session Management
- Unique collection per session
- Automatic cleanup on end
- File tracking per session

## 📊 Commands

All your original commands preserved:

- `/clear` - Delete collection and reset
- `/files` - List uploaded files
- `/stats` - Show chunk count
- `/test` - Test retrieval (returns top 3)
- `/debug` - Show collection samples

## 🎨 Benefits of Refactored Structure

### Maintainability
- Each file has a single responsibility
- Easy to find and modify specific functionality
- Clear imports show dependencies

### Testability
- Each module can be tested independently
- Mock external dependencies easily
- Unit tests can focus on specific functions

### Readability
- Shorter files (100-200 lines vs 1000+)
- Clear module names indicate purpose
- Less scrolling to find code

### Extensibility
- Add new extractors in `processing/`
- Add new retrieval strategies in `storage/`
- Add new generation methods in `rag/`
- Easy to swap out components

## 🔄 Migration from Original

No code changes needed! The refactored version:
- Uses exact same models and ports
- Preserves all parameters and settings
- Keeps all original logic intact
- Maintains same API and behavior

Simply:
1. Copy `src_refactored/` to your server
2. Set `PYTHONPATH`
3. Run `chainlit run app.py`

## 🐛 Debugging

Each module prints debug info:
```
config/settings.py     → "✅ Embedding model: nomic-ai/..."
rag/embeddings.py      → "🔄 Loading embedding model..."
storage/vector_store.py → "✅ ChromaDB at /workspace/..."
app.py                 → "📚 Created collection: docs_abc123"
```

Check logs to see which module is active.

## 📝 Environment Variables

All configurable via environment:
```bash
# Models
export VISION_MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
export REASONING_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
export EMBEDDING_MODEL="nomic-ai/nomic-embed-text-v1.5"

# Endpoints
export VISION_URL="http://localhost:8006/v1"
export REASONING_URL="http://localhost:8005/v1"

# Storage
export CHROMA_HOST="localhost"
export CHROMA_PORT="8000"
export REDIS_HOST="localhost"
export REDIS_PORT="6379"
```

## 🎯 Summary

This refactored version:
✅ Preserves ALL original logic
✅ Uses your exact vLLM configuration
✅ Organizes code into logical modules
✅ Makes code easier to maintain and extend
✅ Keeps same API and behavior
✅ Improves readability

No functionality changed - just better organized!
