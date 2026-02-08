#!/bin/bash
# =============================================================================
# Visual RAG Document Assistant - Production Startup
# =============================================================================
#
# This script starts the entire stack:
# 1. vLLM server (Qwen3-VL-32B-AWQ on host, port 8005)
# 2. Docker stack (Redis + RAG App with Byaldi/ColQwen2)
#
# Usage: ./start.sh [options]
#   --no-vllm       Skip vLLM server startup (if already running)
#   --build         Force rebuild Docker images
#   --logs          Follow logs after startup
#   --stop          Stop all services
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DOCKER_DIR="$SCRIPT_DIR"
LOG_DIR="$PROJECT_DIR/logs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default options
START_VLLM=true
FORCE_BUILD=false
FOLLOW_LOGS=false
STOP_SERVICES=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-vllm)
            START_VLLM=false
            shift
            ;;
        --build)
            FORCE_BUILD=true
            shift
            ;;
        --logs)
            FOLLOW_LOGS=true
            shift
            ;;
        --stop)
            STOP_SERVICES=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Load environment variables
if [ -f "$DOCKER_DIR/.env" ]; then
    source "$DOCKER_DIR/.env"
fi

# Default values
VLLM_PORT=${VLLM_PORT:-8005}
RAG_APP_PORT=${RAG_APP_PORT:-8080}
VLLM_GPU_MEMORY=${VLLM_GPU_MEMORY:-0.80}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-32768}
VLLM_MODEL=${VLLM_MODEL:-QuantTrio/Qwen3-VL-32B-Instruct-AWQ}

# Determine docker compose command
if docker compose version &> /dev/null 2>&1; then
    compose_up() { docker compose "$@"; }
elif command -v docker-compose &> /dev/null; then
    compose_up() { docker-compose "$@"; }
else
    compose_up() { echo "Docker Compose not found"; exit 1; }
fi

# Create log directory
mkdir -p "$LOG_DIR"

echo -e "${BLUE}"
echo "============================================================"
echo "    Visual RAG Document Assistant - Production Startup"
echo "============================================================"
echo -e "${NC}"

# =============================================================================
# STOP SERVICES
# =============================================================================
if [ "$STOP_SERVICES" = true ]; then
    echo -e "${YELLOW}Stopping all services...${NC}"

    # Stop Docker containers
    cd "$DOCKER_DIR"
    compose_up down 2>/dev/null || true

    # Stop vLLM process
    pkill -f "vllm serve" 2>/dev/null || true

    echo -e "${GREEN}All services stopped${NC}"
    exit 0
fi

# =============================================================================
# CHECK PREREQUISITES
# =============================================================================
echo -e "${BLUE}Checking prerequisites...${NC}"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker not found. Please install Docker first.${NC}"
    exit 1
fi
echo -e "${GREEN}Docker available${NC}"

# Check Docker Compose
if ! docker compose version &> /dev/null 2>&1 && ! command -v docker-compose &> /dev/null 2>&1; then
    echo -e "${RED}Docker Compose not found.${NC}"
    exit 1
fi
echo -e "${GREEN}Docker Compose available${NC}"

# Check NVIDIA GPU
if ! nvidia-smi &> /dev/null; then
    echo -e "${RED}NVIDIA GPU not detected. Please check drivers.${NC}"
    exit 1
fi
echo -e "${GREEN}NVIDIA GPU detected${NC}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

# =============================================================================
# CONFIGURE NVIDIA DOCKER RUNTIME (must happen before any Docker commands)
# =============================================================================
echo ""
echo -e "${BLUE}Configuring NVIDIA Docker runtime...${NC}"

if ! docker info 2>/dev/null | grep -q "nvidia"; then
    echo -e "${YELLOW}NVIDIA runtime not configured for Docker. Configuring now...${NC}"
    if command -v nvidia-ctk &> /dev/null; then
        sudo nvidia-ctk runtime configure --runtime=docker
        sudo systemctl restart docker
        sleep 3  # Wait for Docker daemon to stabilize
        echo -e "${GREEN}NVIDIA runtime configured and Docker restarted${NC}"
    else
        echo -e "${RED}nvidia-ctk not found. Install NVIDIA Container Toolkit first:${NC}"
        echo "   sudo apt-get install -y nvidia-container-toolkit"
        echo "   sudo nvidia-ctk runtime configure --runtime=docker"
        echo "   sudo systemctl restart docker"
        exit 1
    fi
else
    echo -e "${GREEN}NVIDIA Docker runtime already configured${NC}"
fi

# =============================================================================
# CREATE PERSISTENT STORAGE DIRECTORIES
# =============================================================================
echo ""
echo -e "${BLUE}Ensuring persistent storage directories exist...${NC}"

PERSISTENT_DIR="$PROJECT_DIR/persistent"
mkdir -p "$PERSISTENT_DIR"/{redis,indexes,huggingface,uploads,data}
echo -e "${GREEN}Persistent storage ready: $PERSISTENT_DIR${NC}"

# =============================================================================
# START vLLM SERVER
# =============================================================================
if [ "$START_VLLM" = true ]; then
    echo ""
    echo -e "${BLUE}Starting vLLM server...${NC}"

    # Set cache paths and HF authentication
    export HF_HOME="$HOME/.cache/huggingface"
    export VLLM_CACHE_ROOT="$HOME/.cache/vllm"
    export VLLM_USE_V1=0
    export HF_TOKEN="${HF_TOKEN:-}"
    unset TRANSFORMERS_CACHE
    mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT"

    # Activate virtual environment if it exists
    if [ -f "$PROJECT_DIR/venv/bin/activate" ]; then
        source "$PROJECT_DIR/venv/bin/activate"
        echo -e "${GREEN}Virtual environment activated${NC}"
    fi

    # Check if vLLM is available
    if ! command -v vllm &> /dev/null; then
        echo -e "${RED}vLLM not found. Please install vLLM first.${NC}"
        echo "   pip install vllm"
        exit 1
    fi

    MAX_WAIT=600  # 10 minutes (includes model download time)
    WAIT_INTERVAL=5

    # Start Qwen3-VL model
    if curl -s "http://localhost:$VLLM_PORT/v1/models" > /dev/null 2>&1; then
        echo -e "${GREEN}vLLM already running on port $VLLM_PORT${NC}"
    else
        echo -e "${YELLOW}Starting $VLLM_MODEL on port $VLLM_PORT (GPU: ${VLLM_GPU_MEMORY})...${NC}"
        nohup vllm serve "$VLLM_MODEL" \
            --port $VLLM_PORT \
            --gpu-memory-utilization $VLLM_GPU_MEMORY \
            --max-model-len $VLLM_MAX_MODEL_LEN \
            --limit-mm-per-prompt '{"image":12}' \
            --trust-remote-code \
            --enable-prefix-caching \
            > "$LOG_DIR/vllm.log" 2>&1 &
        VLLM_PID=$!
        echo "   PID: $VLLM_PID (logs: $LOG_DIR/vllm.log)"

        # Wait for model to be ready
        echo -e "${YELLOW}   Waiting for model to load...${NC}"
        ELAPSED=0
        while [ $ELAPSED -lt $MAX_WAIT ]; do
            if curl -s "http://localhost:$VLLM_PORT/v1/models" > /dev/null 2>&1; then
                echo ""
                echo -e "${GREEN}   vLLM server ready${NC}"
                break
            fi
            # Check if process is still alive
            if ! kill -0 $VLLM_PID 2>/dev/null; then
                echo ""
                echo -e "${RED}   vLLM process died. Check logs: $LOG_DIR/vllm.log${NC}"
                exit 1
            fi
            sleep $WAIT_INTERVAL
            ELAPSED=$((ELAPSED + WAIT_INTERVAL))
            printf "\r   Waiting... %ds" $ELAPSED
        done

        if [ $ELAPSED -ge $MAX_WAIT ]; then
            echo ""
            echo -e "${RED}Timeout waiting for vLLM server${NC}"
            echo "   Check logs: tail -f $LOG_DIR/vllm.log"
            exit 1
        fi
    fi
fi

# =============================================================================
# START DOCKER STACK
# =============================================================================
echo ""
echo -e "${BLUE}Starting Docker stack...${NC}"

cd "$DOCKER_DIR"

# Build if needed
if [ "$FORCE_BUILD" = true ]; then
    echo -e "${YELLOW}Building Docker images...${NC}"
    compose_up build --no-cache
fi

# Start services
echo -e "${YELLOW}Starting services...${NC}"
compose_up up -d

# Wait for services to be healthy
echo -e "${YELLOW}Waiting for services to be healthy...${NC}"

MAX_WAIT=120
WAIT_INTERVAL=5
ELAPSED=0

while [ $ELAPSED -lt $MAX_WAIT ]; do
    REDIS_HEALTHY=$(docker inspect --format='{{.State.Health.Status}}' rag_redis 2>/dev/null || echo "unknown")
    APP_RUNNING=$(docker inspect --format='{{.State.Running}}' rag_app 2>/dev/null || echo "false")

    if [ "$REDIS_HEALTHY" = "healthy" ] && [ "$APP_RUNNING" = "true" ]; then
        echo ""
        echo -e "${GREEN}All Docker services are healthy!${NC}"
        break
    fi

    printf "\r   Waiting... %ds (Redis: %s, App: %s)" \
        $ELAPSED "$REDIS_HEALTHY" $([ "$APP_RUNNING" = "true" ] && echo "running" || echo "starting")

    sleep $WAIT_INTERVAL
    ELAPSED=$((ELAPSED + WAIT_INTERVAL))
done

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo -e "${GREEN}"
echo "============================================================"
echo "          Visual RAG Stack Started Successfully!"
echo "============================================================"
echo -e "${NC}"
echo ""
echo "Services:"
echo "  - RAG Application:   http://localhost:$RAG_APP_PORT"
echo "  - vLLM (Qwen3-VL):  http://localhost:$VLLM_PORT/v1"
echo "  - Redis:             localhost:${REDIS_PORT:-6379}"
echo ""
echo "Logs:"
echo "  - vLLM:    tail -f $LOG_DIR/vllm.log"
echo "  - Docker:  docker compose logs -f"
echo ""
echo "Commands:"
echo "  - Stop all:   ./start.sh --stop"
echo "  - View logs:  docker compose logs -f rag_app"
echo "  - Rebuild:    ./start.sh --build"
echo ""
echo "============================================================"

# Follow logs if requested
if [ "$FOLLOW_LOGS" = true ]; then
    echo ""
    echo -e "${YELLOW}Following logs (Ctrl+C to exit)...${NC}"
    compose_up logs -f
fi
