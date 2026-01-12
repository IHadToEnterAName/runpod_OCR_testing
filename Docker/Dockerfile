# Production RAG Application Dockerfile
# Optimized for NVIDIA RTX 5090

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    curl \
    wget \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create Python 3.10 as default
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 support
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install core ML dependencies
RUN pip install --no-cache-dir \
    transformers==4.36.2 \
    sentence-transformers==2.3.1 \
    langchain==0.1.0 \
    langchain-community==0.0.10 \
    langchain-core==0.1.10 \
    langgraph==0.0.20 \
    langchain-openai==0.0.2

# Install RAG stack
RUN pip install --no-cache-dir \
    chromadb==0.4.22 \
    redis==5.0.1 \
    pymupdf==1.23.8 \
    python-docx==1.1.0 \
    pypdf==3.17.4 \
    pillow==10.1.0 \
    tiktoken==0.5.2

# Install web framework
RUN pip install --no-cache-dir \
    chainlit==1.0.0 \
    openai==1.6.1 \
    httpx==0.25.2

# Install Airflow (for DAG container)
RUN pip install --no-cache-dir \
    apache-airflow==2.8.0 \
    apache-airflow-providers-redis==3.4.0

# Install utilities
RUN pip install --no-cache-dir \
    numpy==1.24.3 \
    pandas==2.0.3 \
    pydantic==2.5.3 \
    python-dotenv==1.0.0

# Create necessary directories
RUN mkdir -p /workspace/src \
    /workspace/data/uploads \
    /workspace/data/processed \
    /workspace/data/failed \
    /workspace/huggingface \
    /workspace/chroma_db \
    /workspace/airflow/dags \
    /workspace/airflow/logs

# Set environment variables
ENV PYTHONPATH=/workspace/src
ENV HF_HOME=/workspace/huggingface
ENV TRANSFORMERS_CACHE=/workspace/huggingface
ENV CUDA_VISIBLE_DEVICES=0,1

# Copy application code
COPY src/ /workspace/src/

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /workspace

USER appuser

# Expose ports
EXPOSE 8000 8080

# Default command (overridden by docker-compose)
CMD ["python", "-m", "chainlit", "run", "/workspace/src/app.py", "--host", "0.0.0.0", "--port", "8000"]
