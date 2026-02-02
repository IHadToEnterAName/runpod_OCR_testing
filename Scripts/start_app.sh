#!/bin/bash
# =============================================================================
# RAG App Startup Script (Manual Mode - without Docker)
# =============================================================================
# Use this to run the Chainlit app directly on the host machine
# instead of inside a Docker container. Useful for development/debugging.
#
# Prerequisites:
#   - Virtual environment with dependencies installed
#   - vLLM servers running on ports 8005 and 8006
#   - Redis running (Docker or local)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(dirname "$SCRIPT_DIR")"
VENV_PATH="$APP_DIR/venv"

echo "=================================================="
echo "RAG Document Assistant - Manual Startup"
echo "=================================================="

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "Virtual environment not found at $VENV_PATH"
    echo "   Please create it first: python3.11 -m venv $VENV_PATH"
    exit 1
fi

# Activate virtual environment
source "$VENV_PATH/bin/activate"
echo "Virtual environment activated"

# Navigate to app directory
cd "$APP_DIR"

# Step 1: Clean Python cache
echo ""
echo "Step 1: Cleaning Python cache..."
find src -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find src -name "*.pyc" -delete
echo "Python cache cleared"

# Step 2: Clear environment variables
echo ""
echo "Step 2: Clearing Docker environment variables..."
unset REDIS_HOST VISION_URL REASONING_URL
export PYTHONDONTWRITEBYTECODE=1
echo "Environment variables cleared"

# Step 3: Check if Redis is running (optional)
echo ""
echo "Step 3: Checking Redis..."
if docker ps | grep -q redis; then
    echo "Redis is running"
elif docker ps -a | grep -q redis; then
    echo "Starting existing Redis container..."
    docker start redis
    echo "Redis started"
else
    echo "Redis not found. Starting new Redis container..."
    docker run -d --name redis -p 6379:6379 redis:7
    echo "Redis started (port 6379)"
fi

# Step 4: Check if vLLM servers are running
echo ""
echo "Step 4: Checking vLLM servers..."
if ! curl -s http://localhost:8006/v1/models > /dev/null 2>&1; then
    echo "Vision model not running on port 8006"
    echo ""
    echo "Please start it in a separate terminal:"
    echo "  vllm serve Qwen/Qwen2.5-VL-3B-Instruct \\"
    echo "    --port 8006 \\"
    echo "    --gpu-memory-utilization 0.3 \\"
    echo "    --max-model-len 4096 \\"
    echo "    --limit-mm-per-prompt '{\"image\":12}' \\"
    echo "    --enforce-eager \\"
    echo "    --trust-remote-code"
    echo ""
else
    echo "Vision model running on port 8006"
fi

if ! curl -s http://localhost:8005/v1/models > /dev/null 2>&1; then
    echo "Reasoning model not running on port 8005"
    echo ""
    echo "Please start it in a separate terminal:"
    echo "  vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \\"
    echo "    --port 8005 \\"
    echo "    --gpu-memory-utilization 0.45 \\"
    echo "    --max-model-len 16384 \\"
    echo "    --enforce-eager \\"
    echo "    --enable-prefix-caching"
    echo ""
else
    echo "Reasoning model running on port 8005"
fi

# Step 5: Start Chainlit app
echo ""
echo "Step 5: Starting RAG application on port 8080..."
echo "=================================================="
echo ""

export PYTHONPATH="$APP_DIR/src"
chainlit run src/app.py --host 0.0.0.0 --port 8080
