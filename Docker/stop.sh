#!/bin/bash
# =============================================================================
# RAG Document Assistant - Stop All Services
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Determine docker compose command (v2 plugin or v1 standalone)
if docker compose version &> /dev/null 2>&1; then
    compose_cmd() { docker compose "$@"; }
elif command -v docker-compose &> /dev/null; then
    compose_cmd() { docker-compose "$@"; }
else
    compose_cmd() { echo "Docker Compose not found"; }
fi

echo "============================================================"
echo "       Stopping RAG Document Assistant"
echo "============================================================"

# Stop Docker containers
echo "Stopping Docker containers..."
cd "$SCRIPT_DIR"
compose_cmd down 2>/dev/null || true

# Stop vLLM processes
echo "Stopping vLLM servers..."
pkill -f "vllm serve" 2>/dev/null || true

# Clean up
docker container prune -f 2>/dev/null || true

echo ""
echo "All services stopped"
echo "============================================================"
