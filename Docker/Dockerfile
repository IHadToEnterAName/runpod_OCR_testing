# =============================================================================
# Production RAG Application Dockerfile
# Optimized for NVIDIA A100 (Ampere SM 8.0) on Azure
# =============================================================================

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

LABEL maintainer="RAG Document Assistant"
LABEL version="2.0"
LABEL description="Production RAG application for A100 GPU"

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /workspace

# =============================================================================
# System Dependencies
# =============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    build-essential \
    git \
    curl \
    wget \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && python -m ensurepip --upgrade \
    && python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# =============================================================================
# Python Dependencies - Layered for better caching
# =============================================================================

# Layer 1: PyTorch with CUDA 12.1 support (optimized for A100)
RUN pip install --no-cache-dir \
    torch==2.3.1 \
    torchvision==0.18.1 \
    torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Layer 2: ML/Transformers
RUN pip install --no-cache-dir \
    transformers>=4.40.0 \
    sentence-transformers>=3.0.0 \
    tiktoken>=0.7.0 \
    einops

# Layer 3: LangChain stack
RUN pip install --no-cache-dir \
    langchain>=0.2.0 \
    langchain-community>=0.2.0 \
    langchain-core>=0.2.0 \
    langchain-text-splitters>=0.2.0 \
    langgraph>=0.2.0

# Layer 4: Vector store and caching
RUN pip install --no-cache-dir \
    chromadb>=0.5.0 \
    redis>=5.0.0 \
    hiredis>=2.3.0

# Layer 5: Document processing
RUN pip install --no-cache-dir \
    PyMuPDF>=1.24.0 \
    python-docx>=1.1.0 \
    Pillow>=10.0.0

# Layer 6: Web framework and API
RUN pip install --no-cache-dir \
    chainlit>=1.1.0 \
    fastapi>=0.110.0 \
    "uvicorn[standard]>=0.29.0" \
    openai>=1.30.0 \
    httpx>=0.27.0

# Layer 7: Utilities
RUN pip install --no-cache-dir \
    "numpy>=1.26.0,<2.0.0" \
    pandas>=2.2.0 \
    pydantic>=2.7.0 \
    python-dotenv>=1.0.0 \
    pyyaml>=6.0.0 \
    hf_transfer \
    uvloop \
    typing-extensions>=4.10.0

# =============================================================================
# Application Setup
# =============================================================================

# Create directories
RUN mkdir -p /workspace/src \
    /workspace/data/uploads \
    /workspace/data/processed \
    /workspace/data/failed \
    /workspace/huggingface \
    /workspace/chroma_db \
    /workspace/logs

# Environment variables
ENV PYTHONPATH=/workspace/src
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/huggingface
ENV TRANSFORMERS_CACHE=/workspace/huggingface
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV CUDA_VISIBLE_DEVICES=0

# Pre-download embedding and reranker models so container doesn't need internet on startup
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True); \
from sentence_transformers import CrossEncoder; \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
print('Models downloaded successfully')"

# Copy application code
COPY src/ /workspace/src/

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "-m", "chainlit", "run", "/workspace/src/app.py", "--host", "0.0.0.0", "--port", "8000"]
