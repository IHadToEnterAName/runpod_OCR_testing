#!/bin/bash
# One-command deployment for RAG system
# Usage: curl -fsSL https://your-repo/deploy.sh | bash

set -e

echo "🚀 RAG System - Automated Deployment"
echo "===================================="
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Installing..."
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker $USER
    echo "✅ Docker installed. Please log out and log back in, then run this script again."
    exit 0
fi

# Check NVIDIA runtime
if ! docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi &> /dev/null 2>&1; then
    echo "❌ NVIDIA Docker runtime not found. Installing..."
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
        sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
    sudo systemctl restart docker
    echo "✅ NVIDIA runtime installed"
fi

# Create directories
mkdir -p rag-system/{data/{uploads,processed,failed},volumes/{huggingface,vllm_cache,models}}
cd rag-system

# Download files (if not present)
if [ ! -f "docker-compose.complete.yml" ]; then
    echo "📥 Downloading configuration..."
    # In production, download from your repo:
    # curl -O https://your-repo/docker-compose.complete.yml
    # curl -O https://your-repo/Dockerfile.complete
    echo "⚠️  Please copy docker-compose.complete.yml and Dockerfile.complete here"
    exit 1
fi

# Build and start
echo "🔨 Building images (this will take 10-20 minutes)..."
docker compose -f docker-compose.complete.yml build

echo "🚀 Starting services..."
docker compose -f docker-compose.complete.yml up -d

echo "⏳ Waiting for services to initialize..."
sleep 30

# Wait for health checks
timeout 300 bash -c 'until curl -s http://localhost:8000 &> /dev/null; do sleep 5; done' || {
    echo "❌ Services failed to start. Check logs with:"
    echo "   docker compose -f docker-compose.complete.yml logs"
    exit 1
}

echo ""
echo "✅ Deployment complete!"
echo ""
echo "🌐 Access your RAG system:"
echo "   http://localhost:8000"
echo ""
echo "📊 Useful commands:"
echo "   Status:  docker compose -f docker-compose.complete.yml ps"
echo "   Logs:    docker compose -f docker-compose.complete.yml logs -f"
echo "   Stop:    docker compose -f docker-compose.complete.yml down"
echo "   Restart: docker compose -f docker-compose.complete.yml restart"
echo ""
