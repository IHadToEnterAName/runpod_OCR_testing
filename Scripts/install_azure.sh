#!/bin/bash
# =============================================================================
# RAG Document Assistant - Azure A100 Fresh Installation Script
# =============================================================================
#
# This script installs EVERYTHING from scratch on a fresh Azure A100 VM
# including Python, pip, Docker, NVIDIA toolkit, vLLM, and all dependencies.
#
# Prerequisites:
#   - Ubuntu 22.04 LTS
#   - A100 GPU with NVIDIA drivers installed (Azure NC A100 v4 series)
#   - Sudo access
#
# Usage:
#   cd ~/Nasih_Test/Scripts
#   chmod +x install_azure.sh
#   ./install_azure.sh
#
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "============================================================"
echo "    RAG Document Assistant - Azure A100 Installation"
echo "    Installing everything from scratch..."
echo "============================================================"
echo -e "${NC}"

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="${PROJECT_ROOT}/venv"

echo "Project root: $PROJECT_ROOT"
echo "Venv target:  $VENV_DIR"
echo ""

# =============================================================================
# Step 1: System Updates & Core Packages
# =============================================================================
echo -e "${BLUE}Step 1/9: Updating system and installing core packages...${NC}"

export DEBIAN_FRONTEND=noninteractive

sudo apt-get update -y
sudo apt-get upgrade -y

# Install absolutely everything needed from scratch
sudo apt-get install -y \
    software-properties-common \
    build-essential \
    gcc \
    g++ \
    make \
    git \
    curl \
    wget \
    unzip \
    htop \
    tmux \
    vim \
    nano \
    pkg-config \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libffi-dev \
    libssl-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    liblzma-dev \
    zlib1g-dev \
    ca-certificates \
    gnupg \
    lsb-release \
    apt-transport-https

echo -e "${GREEN}[1/9] System packages installed${NC}"

# =============================================================================
# Step 2: Install Python 3.11 + pip
# =============================================================================
echo ""
echo -e "${BLUE}Step 2/9: Installing Python 3.11 and pip...${NC}"

# Always add deadsnakes PPA and install Python 3.11
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update -y

# Install Python 3.11 core packages (distutils may not exist on all versions)
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
sudo apt-get install -y python3.11-distutils 2>/dev/null || true
sudo apt-get install -y python3-pip 2>/dev/null || true

# Ensure pip is available for python3.11 (bootstrap it directly)
echo "Bootstrapping pip for python3.11..."
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
python3.11 /tmp/get-pip.py --user 2>/dev/null || python3.11 /tmp/get-pip.py
rm -f /tmp/get-pip.py

# Verify python and pip
echo "Verifying Python installation..."
python3.11 --version || { echo -e "${RED}Python 3.11 installation failed!${NC}"; exit 1; }
python3.11 -m pip --version || { echo -e "${RED}pip installation failed!${NC}"; exit 1; }

echo -e "${GREEN}[2/9] Python 3.11 and pip installed${NC}"

# =============================================================================
# Step 3: Install Docker
# =============================================================================
echo ""
echo -e "${BLUE}Step 3/9: Installing Docker...${NC}"

if command -v docker &> /dev/null; then
    echo -e "${YELLOW}Docker already installed: $(docker --version)${NC}"
else
    # Remove any old Docker installations
    sudo apt-get remove -y docker docker-engine docker.io containerd runc 2>/dev/null || true

    # Add Docker's official GPG key
    sudo install -m 0755 -d /etc/apt/keyrings
    if [ -f /etc/apt/keyrings/docker.gpg ]; then
        sudo rm /etc/apt/keyrings/docker.gpg
    fi
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg

    # Add Docker repository
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    # Install Docker
    sudo apt-get update -y
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    # Add current user to docker group
    sudo usermod -aG docker $USER

    # Start Docker
    sudo systemctl enable docker
    sudo systemctl start docker

    echo -e "${GREEN}Docker installed: $(docker --version)${NC}"
    echo -e "${YELLOW}NOTE: You need to log out and back in for docker group to take effect${NC}"
fi

echo -e "${GREEN}[3/9] Docker installed${NC}"

# =============================================================================
# Step 4: Install NVIDIA Container Toolkit
# =============================================================================
echo ""
echo -e "${BLUE}Step 4/9: Installing NVIDIA Container Toolkit...${NC}"

if dpkg -l 2>/dev/null | grep -q nvidia-container-toolkit; then
    echo -e "${YELLOW}NVIDIA Container Toolkit already installed${NC}"
else
    # Add NVIDIA Container Toolkit repository
    distribution=$(. /etc/os-release; echo $ID$VERSION_ID)

    # Clean up old keys if they exist
    if [ -f /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg ]; then
        sudo rm /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    fi

    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
        sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    sudo apt-get update -y
    sudo apt-get install -y nvidia-container-toolkit

    # Configure Docker to use NVIDIA runtime
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker

    echo -e "${GREEN}NVIDIA Container Toolkit installed${NC}"
fi

echo -e "${GREEN}[4/9] NVIDIA Container Toolkit installed${NC}"

# =============================================================================
# Step 5: Verify GPU Access
# =============================================================================
echo ""
echo -e "${BLUE}Step 5/9: Verifying GPU access...${NC}"

if nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo -e "${GREEN}[5/9] GPU verified${NC}"
else
    echo -e "${RED}WARNING: nvidia-smi failed. NVIDIA drivers may not be installed.${NC}"
    echo -e "${RED}Install drivers first: sudo apt install nvidia-driver-535${NC}"
    echo -e "${YELLOW}Continuing anyway...${NC}"
fi

# =============================================================================
# Step 6: Verify Project Directory
# =============================================================================
echo ""
echo -e "${BLUE}Step 6/9: Verifying project directory...${NC}"

if [ ! -d "$PROJECT_ROOT/src" ]; then
    echo -e "${RED}Error: Project source not found at $PROJECT_ROOT/src${NC}"
    echo -e "${RED}Make sure you run this script from within the project's Scripts/ directory${NC}"
    echo -e "${RED}Expected structure:${NC}"
    echo -e "${RED}  ~/Nasih_Test/${NC}"
    echo -e "${RED}  ├── Scripts/install_azure.sh  <-- run from here${NC}"
    echo -e "${RED}  ├── src/${NC}"
    echo -e "${RED}  ├── Docker/${NC}"
    echo -e "${RED}  └── ...${NC}"
    exit 1
fi

mkdir -p "$PROJECT_ROOT/data" "$PROJECT_ROOT/logs"

echo "Project structure verified:"
ls -d "$PROJECT_ROOT"/*/
echo -e "${GREEN}[6/9] Project directory verified at $PROJECT_ROOT${NC}"

# =============================================================================
# Step 7: Create Virtual Environment
# =============================================================================
echo ""
echo -e "${BLUE}Step 7/9: Creating Python virtual environment...${NC}"

# Remove existing venv if corrupted
if [ -d "$VENV_DIR" ] && [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo -e "${YELLOW}Removing corrupted venv...${NC}"
    rm -rf "$VENV_DIR"
fi

# Create fresh venv
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    python3.11 -m venv "$VENV_DIR"
else
    echo -e "${YELLOW}Virtual environment already exists at $VENV_DIR${NC}"
fi

# Activate it
source "$VENV_DIR/bin/activate"

# Verify we're in the venv
echo "Python location: $(which python)"
echo "Pip location:    $(which pip)"

# Upgrade pip inside venv
pip install --upgrade pip setuptools wheel

echo -e "${GREEN}[7/9] Virtual environment created and activated${NC}"

# =============================================================================
# Step 8: Install PyTorch, vLLM, and Dependencies
# =============================================================================
echo ""
echo -e "${BLUE}Step 8/9: Installing PyTorch, vLLM, and all dependencies...${NC}"

# Make sure we're in the venv
source "$VENV_DIR/bin/activate"

# Install PyTorch with CUDA 12.1 (optimized for A100)
echo ""
echo ">>> Installing PyTorch with CUDA 12.1..."
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Install vLLM
echo ""
echo ">>> Installing vLLM..."
pip install vllm

# Install project dependencies from requirements.txt
echo ""
echo ">>> Installing project dependencies..."
if [ -f "$PROJECT_ROOT/Scripts/requirements.txt" ]; then
    pip install -r "$PROJECT_ROOT/Scripts/requirements.txt"
else
    echo -e "${YELLOW}Warning: requirements.txt not found at $PROJECT_ROOT/Scripts/requirements.txt${NC}"
    echo "Installing core dependencies manually..."
    pip install \
        "transformers>=4.40.0" \
        "sentence-transformers>=3.0.0" \
        "tiktoken>=0.7.0" \
        "einops" \
        "langchain>=0.2.0" \
        "langchain-community>=0.2.0" \
        "langchain-core>=0.2.0" \
        "langchain-text-splitters>=0.2.0" \
        "langgraph>=0.2.0" \
        "chromadb>=0.5.0" \
        "redis>=5.0.0" \
        "hiredis>=2.3.0" \
        "PyMuPDF>=1.24.0" \
        "python-docx>=1.1.0" \
        "Pillow>=10.0.0" \
        "chainlit>=1.1.0" \
        "fastapi>=0.110.0" \
        "uvicorn[standard]>=0.29.0" \
        "openai>=1.30.0" \
        "httpx>=0.27.0" \
        "numpy>=1.26.0,<2.0.0" \
        "pandas>=2.2.0" \
        "pydantic>=2.7.0" \
        "python-dotenv>=1.0.0" \
        "pyyaml>=6.0.0" \
        "hf_transfer" \
        "uvloop" \
        "typing-extensions>=4.10.0"
fi

echo -e "${GREEN}[8/9] All Python packages installed${NC}"

# =============================================================================
# Step 9: Verify Everything
# =============================================================================
echo ""
echo -e "${BLUE}Step 9/9: Verifying complete installation...${NC}"

source "$VENV_DIR/bin/activate"

echo ""
echo "--- System ---"
echo "Python:  $(python --version)"
echo "Pip:     $(pip --version)"
echo "Docker:  $(docker --version 2>/dev/null || echo 'not found')"
echo "Compose: $(docker compose version 2>/dev/null || echo 'not found')"

echo ""
echo "--- GPU ---"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "GPU: not detected"

echo ""
echo "--- Python Packages ---"
python -c "
import torch
print(f'PyTorch:        {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version:   {torch.version.cuda}')
    print(f'GPU:            {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    mem = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
    print(f'GPU Memory:     {mem / 1024**3:.1f} GB')
" 2>/dev/null || echo "PyTorch: import failed"

python -c "import vllm; print(f'vLLM:           {vllm.__version__}')" 2>/dev/null || echo "vLLM: import failed (may work after restart)"
python -c "import chainlit; print(f'Chainlit:       {chainlit.__version__}')" 2>/dev/null || echo "Chainlit: not found"
python -c "import langchain; print(f'LangChain:      {langchain.__version__}')" 2>/dev/null || echo "LangChain: not found"
python -c "import chromadb; print(f'ChromaDB:       {chromadb.__version__}')" 2>/dev/null || echo "ChromaDB: not found"
python -c "import redis; print(f'Redis:          {redis.__version__}')" 2>/dev/null || echo "Redis: not found"

# Test Docker GPU access
echo ""
echo "--- Docker GPU Test ---"
sudo docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi 2>/dev/null && \
    echo "Docker GPU: working" || \
    echo -e "${YELLOW}Docker GPU test failed (may need logout/login for group permissions)${NC}"

echo ""
echo -e "${GREEN}"
echo "============================================================"
echo "            Installation Complete!"
echo "============================================================"
echo -e "${NC}"
echo ""
echo "Project directory:   $PROJECT_ROOT"
echo "Virtual environment: $VENV_DIR"
echo ""
echo "============================================================"
echo "  NEXT STEPS"
echo "============================================================"
echo ""
echo "  1. Log out and back in (required for Docker group):"
echo "     exit"
echo "     ssh -i <private-key-file-path> <USER>@<YOUR_VM_IP>"
echo ""
echo "  2. Activate the virtual environment:"
echo "     cd $PROJECT_ROOT"
echo "     source venv/bin/activate"
echo ""
echo "  3. Start the RAG stack:"
echo "     cd Docker"
echo "     ./start.sh"
echo ""
echo "  4. Access the app:"
echo "     http://<YOUR_VM_IP>:8080"
echo ""
echo "============================================================"
