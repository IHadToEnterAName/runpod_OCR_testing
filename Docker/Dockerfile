# =============================================================================
# Visual RAG Application Dockerfile
# Byaldi (ColQwen2) + Qwen3-VL via vLLM
# NVIDIA runtime needed for ColQwen2 GPU inference
# =============================================================================

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

LABEL maintainer="Visual RAG Document Assistant"
LABEL version="3.0"
LABEL description="Visual RAG with Byaldi (ColQwen2) retrieval"

ENV DEBIAN_FRONTEND=noninteractive

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
    poppler-utils \
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

# Layer 1: PyTorch with CUDA 12.1 (needed for ColQwen2 via Byaldi)
RUN pip install --no-cache-dir \
    torch==2.3.1 \
    torchvision==0.18.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Layer 2: Byaldi + ColPali engine (visual retrieval)
RUN pip install --no-cache-dir \
    byaldi>=0.0.5 \
    "colpali-engine>=0.3.4,<0.4.0" \
    pdf2image \
    ninja

# Layer 3: ML/Tokenization
RUN pip install --no-cache-dir \
    transformers>=4.42.0 \
    "peft>=0.14.0" \
    tiktoken>=0.7.0

# Layer 4: Cache
RUN pip install --no-cache-dir \
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
    /workspace/data/indexes \
    /workspace/huggingface \
    /workspace/logs

# Environment variables
ENV PYTHONPATH=/workspace/src
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/huggingface
ENV TRANSFORMERS_CACHE=/workspace/huggingface
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV CUDA_VISIBLE_DEVICES=0

# Pre-download ColQwen2 model weights (download only, no GPU needed)
# The model will be loaded onto GPU at runtime
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('vidore/colqwen2-v0.1'); \
print('ColQwen2 model files downloaded successfully')"

# Copy application code
COPY src/ /workspace/src/

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

EXPOSE 8000

CMD ["python", "-m", "chainlit", "run", "/workspace/src/app.py", "--host", "0.0.0.0", "--port", "8000"]
