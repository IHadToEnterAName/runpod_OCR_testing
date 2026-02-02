#!/bin/bash
# =============================================================================
# RAG Document Assistant - One-Click Production Startup
# =============================================================================
#
# This script starts the entire RAG stack:
# 1. vLLM Vision Model (on host, port 8006)
# 2. vLLM Reasoning Model (on host, port 8005)
# 3. Docker stack (Redis, ChromaDB, RAG App)
#
# Usage: ./start.sh [options]
#   --no-vllm       Skip vLLM server startup (if already running)
#   --build         Force rebuild Docker images
#   --detach        Run in background (default)
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
NC='\033[0m' # No Color

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

# Default values (conservative GPU memory allocation)
VISION_PORT=${VISION_PORT:-8006}
REASONING_PORT=${REASONING_PORT:-8005}
RAG_APP_PORT=${RAG_APP_PORT:-8080}
VISION_GPU_MEMORY=${VISION_GPU_MEMORY:-0.30}
REASONING_GPU_MEMORY=${REASONING_GPU_MEMORY:-0.45}
VISION_MAX_MODEL_LEN=${VISION_MAX_MODEL_LEN:-4096}
REASONING_MAX_MODEL_LEN=${REASONING_MAX_MODEL_LEN:-16384}
VISION_MODEL=${VISION_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}
REASONING_MODEL=${REASONING_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}

# Determine docker compose command (v2 plugin or v1 standalone)
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
echo "       RAG Document Assistant - Production Startup"
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

    # Stop vLLM processes
    pkill -f "vllm serve" 2>/dev/null || true

    echo -e "${GREEN}✅ All services stopped${NC}"
    exit 0
fi

# =============================================================================
# CHECK PREREQUISITES
# =============================================================================
echo -e "${BLUE}Checking prerequisites...${NC}"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not found. Please install Docker first.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Docker available${NC}"

# Check Docker Compose
if ! docker compose version &> /dev/null 2>&1 && ! command -v docker-compose &> /dev/null 2>&1; then
    echo -e "${RED}❌ Docker Compose not found.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Docker Compose available${NC}"

# Check NVIDIA GPU
if ! nvidia-smi &> /dev/null; then
    echo -e "${RED}❌ NVIDIA GPU not detected. Please check drivers.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ NVIDIA GPU detected${NC}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

# =============================================================================
# START vLLM SERVERS
# =============================================================================
if [ "$START_VLLM" = true ]; then
    echo ""
    echo -e "${BLUE}Starting vLLM servers...${NC}"

    # Force HuggingFace and vLLM cache paths to home directory
    export HF_HOME="$HOME/.cache/huggingface"
    export VLLM_CACHE_ROOT="$HOME/.cache/vllm"
    export VLLM_USE_V1=0
    unset TRANSFORMERS_CACHE
    mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT"

    # Activate virtual environment if it exists
    if [ -f "$PROJECT_DIR/venv/bin/activate" ]; then
        source "$PROJECT_DIR/venv/bin/activate"
        echo -e "${GREEN}✅ Virtual environment activated${NC}"
    fi

    # Check if vLLM is available
    if ! command -v vllm &> /dev/null; then
        echo -e "${RED}❌ vLLM not found. Please install vLLM first.${NC}"
        echo "   pip install vllm"
        exit 1
    fi

    MAX_WAIT=600  # 10 minutes (includes model download time)
    WAIT_INTERVAL=5

    # Start Vision Model FIRST (must claim GPU memory before reasoning model)
    if curl -s "http://localhost:$VISION_PORT/v1/models" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Vision model already running on port $VISION_PORT${NC}"
    else
        echo -e "${YELLOW}Starting Vision model on port $VISION_PORT (GPU: ${VISION_GPU_MEMORY})...${NC}"
        nohup vllm serve "$VISION_MODEL" \
            --port $VISION_PORT \
            --gpu-memory-utilization $VISION_GPU_MEMORY \
            --max-model-len $VISION_MAX_MODEL_LEN \
            --limit-mm-per-prompt '{"image":12}' \
            --trust-remote-code \
            > "$LOG_DIR/vllm_vision.log" 2>&1 &
        VISION_PID=$!
        echo "   PID: $VISION_PID (logs: $LOG_DIR/vllm_vision.log)"

        # Wait for vision model to be fully loaded before starting reasoning
        echo -e "${YELLOW}   Waiting for Vision model to load before starting Reasoning model...${NC}"
        VISION_WAIT=0
        while [ $VISION_WAIT -lt $MAX_WAIT ]; do
            if curl -s "http://localhost:$VISION_PORT/v1/models" > /dev/null 2>&1; then
                echo -e "${GREEN}   ✅ Vision model ready${NC}"
                break
            fi
            # Check if process is still alive
            if ! kill -0 $VISION_PID 2>/dev/null; then
                echo -e "${RED}   ❌ Vision model process died. Check logs: $LOG_DIR/vllm_vision.log${NC}"
                break
            fi
            sleep $WAIT_INTERVAL
            VISION_WAIT=$((VISION_WAIT + WAIT_INTERVAL))
            printf "\r   Waiting... %ds" $VISION_WAIT
        done
        echo ""
    fi

    # Start Reasoning Model SECOND (after vision has claimed its GPU memory)
    if curl -s "http://localhost:$REASONING_PORT/v1/models" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Reasoning model already running on port $REASONING_PORT${NC}"
    else
        echo -e "${YELLOW}Starting Reasoning model on port $REASONING_PORT (GPU: ${REASONING_GPU_MEMORY})...${NC}"
        nohup vllm serve "$REASONING_MODEL" \
            --port $REASONING_PORT \
            --gpu-memory-utilization $REASONING_GPU_MEMORY \
            --max-model-len $REASONING_MAX_MODEL_LEN \
            --enable-prefix-caching \
            > "$LOG_DIR/vllm_reasoning.log" 2>&1 &
        REASONING_PID=$!
        echo "   PID: $REASONING_PID (logs: $LOG_DIR/vllm_reasoning.log)"
    fi

    # Wait for reasoning model to be ready
    echo ""
    echo -e "${YELLOW}Waiting for Reasoning model to start...${NC}"
    ELAPSED=0

    while [ $ELAPSED -lt $MAX_WAIT ]; do
        VISION_READY=false
        REASONING_READY=false

        if curl -s "http://localhost:$VISION_PORT/v1/models" > /dev/null 2>&1; then
            VISION_READY=true
        fi

        if curl -s "http://localhost:$REASONING_PORT/v1/models" > /dev/null 2>&1; then
            REASONING_READY=true
        fi

        if [ "$VISION_READY" = true ] && [ "$REASONING_READY" = true ]; then
            echo ""
            echo -e "${GREEN}✅ Both vLLM servers are ready!${NC}"
            break
        fi

        printf "\r   Waiting... %ds (Vision: %s, Reasoning: %s)" \
            $ELAPSED \
            $([ "$VISION_READY" = true ] && echo "ready" || echo "starting") \
            $([ "$REASONING_READY" = true ] && echo "ready" || echo "starting")

        sleep $WAIT_INTERVAL
        ELAPSED=$((ELAPSED + WAIT_INTERVAL))
    done

    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo ""
        echo -e "${RED}❌ Timeout waiting for vLLM servers${NC}"
        echo "   Check logs: tail -f $LOG_DIR/vllm_*.log"
        exit 1
    fi
fi

# =============================================================================
# VERIFY NVIDIA DOCKER RUNTIME
# =============================================================================
echo ""
echo -e "${BLUE}Checking NVIDIA Docker runtime...${NC}"

if ! docker info 2>/dev/null | grep -q "nvidia"; then
    echo -e "${YELLOW}NVIDIA runtime not configured for Docker. Configuring now...${NC}"
    if command -v nvidia-ctk &> /dev/null; then
        sudo nvidia-ctk runtime configure --runtime=docker
        sudo systemctl restart docker
        echo -e "${GREEN}✅ NVIDIA runtime configured${NC}"
    else
        echo -e "${RED}❌ nvidia-ctk not found. Install NVIDIA Container Toolkit first:${NC}"
        echo "   sudo apt-get install -y nvidia-container-toolkit"
        echo "   sudo nvidia-ctk runtime configure --runtime=docker"
        echo "   sudo systemctl restart docker"
        exit 1
    fi
else
    echo -e "${GREEN}✅ NVIDIA Docker runtime available${NC}"
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
    CHROMA_HEALTHY=$(docker inspect --format='{{.State.Health.Status}}' rag_chromadb 2>/dev/null || echo "unknown")
    APP_RUNNING=$(docker inspect --format='{{.State.Running}}' rag_app 2>/dev/null || echo "false")

    if [ "$REDIS_HEALTHY" = "healthy" ] && [ "$CHROMA_HEALTHY" = "healthy" ] && [ "$APP_RUNNING" = "true" ]; then
        echo ""
        echo -e "${GREEN}✅ All Docker services are healthy!${NC}"
        break
    fi

    printf "\r   Waiting... %ds (Redis: %s, ChromaDB: %s, App: %s)" \
        $ELAPSED "$REDIS_HEALTHY" "$CHROMA_HEALTHY" $([ "$APP_RUNNING" = "true" ] && echo "running" || echo "starting")

    sleep $WAIT_INTERVAL
    ELAPSED=$((ELAPSED + WAIT_INTERVAL))
done

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo -e "${GREEN}"
echo "============================================================"
echo "             ✅ RAG Stack Started Successfully!"
echo "============================================================"
echo -e "${NC}"
echo ""
echo "Services:"
echo "  • RAG Application:   http://localhost:$RAG_APP_PORT"
echo "  • Vision Model:      http://localhost:$VISION_PORT/v1"
echo "  • Reasoning Model:   http://localhost:$REASONING_PORT/v1"
echo "  • Redis:             localhost:${REDIS_PORT:-6379}"
echo "  • ChromaDB:          http://localhost:${CHROMA_PORT:-8003}"
echo ""
echo "Logs:"
echo "  • vLLM Vision:    tail -f $LOG_DIR/vllm_vision.log"
echo "  • vLLM Reasoning: tail -f $LOG_DIR/vllm_reasoning.log"
echo "  • Docker:         docker compose logs -f"
echo ""
echo "Commands:"
echo "  • Stop all:       ./start.sh --stop"
echo "  • View logs:      docker compose logs -f rag_app"
echo "  • Rebuild:        ./start.sh --build"
echo ""
echo "============================================================"

# Follow logs if requested
if [ "$FOLLOW_LOGS" = true ]; then
    echo ""
    echo -e "${YELLOW}Following logs (Ctrl+C to exit)...${NC}"
    compose_up logs -f
fi
