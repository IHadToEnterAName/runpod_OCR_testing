#!/bin/bash
# =============================================================================
# Azure VM Startup Script
# =============================================================================
# Run this instead of ./start.sh on Azure VMs.
# Fixes common Azure permission issues, then launches the full stack.
#
# Usage: ./azure-start.sh [any start.sh options like --build, --logs, --stop]
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PERSISTENT_DIR="$PROJECT_DIR/persistent"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "============================================================"
echo "    Azure VM Pre-flight Checks"
echo "============================================================"
echo -e "${NC}"

# ── 1. Fix persistent directory ownership (Docker creates dirs as root) ──
echo -e "${YELLOW}Fixing persistent storage permissions...${NC}"
if [ -d "$PERSISTENT_DIR" ]; then
    # Find any root-owned files/dirs and fix them
    ROOT_OWNED=$(find "$PERSISTENT_DIR" ! -user "$USER" 2>/dev/null | head -5)
    if [ -n "$ROOT_OWNED" ]; then
        echo "  Found root-owned files, fixing with sudo..."
        sudo chown -R "$USER:$USER" "$PERSISTENT_DIR"
        echo -e "  ${GREEN}Permissions fixed${NC}"
    else
        echo -e "  ${GREEN}Permissions OK${NC}"
    fi
else
    mkdir -p "$PERSISTENT_DIR"
    echo -e "  ${GREEN}Created persistent directory${NC}"
fi

# ── 2. Ensure all subdirectories exist ──
echo -e "${YELLOW}Ensuring directory structure...${NC}"
mkdir -p "$PERSISTENT_DIR"/{redis,indexes,huggingface/hub,uploads,data,vllm_cache}
echo -e "${GREEN}Directory structure OK${NC}"

# ── 3. Ensure Docker is running ──
echo -e "${YELLOW}Checking Docker...${NC}"
if ! docker info &>/dev/null; then
    echo "  Docker not running, starting..."
    sudo service docker start 2>/dev/null || sudo systemctl start docker 2>/dev/null
    sleep 3
    if docker info &>/dev/null; then
        echo -e "  ${GREEN}Docker started${NC}"
    else
        echo -e "  ${RED}Failed to start Docker. Start it manually.${NC}"
        exit 1
    fi
else
    echo -e "  ${GREEN}Docker running${NC}"
fi

# ── 4. Clean up stale vLLM processes from previous VM session ──
if pgrep -f "vllm serve" &>/dev/null; then
    echo -e "${YELLOW}Cleaning up stale vLLM process from previous session...${NC}"
    pkill -f "vllm serve" 2>/dev/null || true
    sleep 2
    echo -e "${GREEN}Cleaned up${NC}"
fi

# ── Done, hand off to start.sh ──
echo ""
echo -e "${GREEN}Pre-flight checks passed. Launching stack...${NC}"
echo ""

exec "$SCRIPT_DIR/start.sh" "$@"
