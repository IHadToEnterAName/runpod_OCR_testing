#!/bin/bash

# =============================================================================
# Docker Deployment Script
# Complete containerized RAG system with vLLM
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Helper functions
print_header() {
    echo -e "${BLUE}"
    echo "============================================================"
    echo "$1"
    echo "============================================================"
    echo -e "${NC}"
}

print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }

# Check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker not found. Please install Docker."
        exit 1
    fi
    print_success "Docker installed: $(docker --version)"
    
    # Check Docker Compose
    if ! docker compose version &> /dev/null; then
        print_error "Docker Compose V2 not found. Please install Docker Compose."
        exit 1
    fi
    print_success "Docker Compose installed: $(docker compose version)"
    
    # Check NVIDIA Docker
    if ! docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi &> /dev/null 2>&1; then
        print_warning "NVIDIA Docker runtime not working properly"
        print_info "Please install nvidia-container-toolkit"
        print_info "See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        exit 1
    fi
    print_success "NVIDIA Docker runtime working"
    
    # Check GPU count
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
    print_info "Detected $GPU_COUNT GPU(s)"
    
    if [ "$GPU_COUNT" -lt 2 ]; then
        print_warning "Only $GPU_COUNT GPU detected. System optimized for 2 GPUs."
        print_info "Will adjust configuration for single GPU..."
    fi
    
    echo ""
}

# Create directory structure
create_directories() {
    print_header "Creating Directory Structure"
    
    mkdir -p data/uploads
    mkdir -p data/processed
    mkdir -p data/failed
    mkdir -p volumes/huggingface
    mkdir -p volumes/vllm_cache
    mkdir -p volumes/models
    
    print_success "Directories created"
    echo ""
}

# Build images
build_images() {
    print_header "Building Docker Images"
    
    print_info "This will take 10-20 minutes (vLLM compilation)..."
    print_info "Building..."
    
    docker compose -f docker-compose.complete.yml build
    
    print_success "Images built successfully"
    echo ""
}

# Start services
start_services() {
    print_header "Starting Services"
    
    print_info "Starting containers..."
    docker compose -f docker-compose.complete.yml up -d
    
    print_success "Containers started"
    echo ""
}

# Wait for services
wait_for_services() {
    print_header "Waiting for Services to Initialize"
    
    # ChromaDB
    print_info "Waiting for ChromaDB..."
    timeout 60 bash -c 'until curl -s http://localhost:8000/api/v1/heartbeat &> /dev/null; do sleep 2; done' || {
        print_error "ChromaDB failed to start"
        print_info "Check logs: docker compose logs chromadb"
        exit 1
    }
    print_success "ChromaDB ready"
    
    # Redis
    print_info "Waiting for Redis..."
    timeout 60 bash -c 'until docker exec rag_redis redis-cli ping &> /dev/null; do sleep 2; done' || {
        print_warning "Redis not responding (non-critical)"
    }
    
    # vLLM Vision (this takes time to load models)
    print_info "Waiting for Vision model... (may take 2-3 minutes)"
    timeout 180 bash -c 'until curl -s http://localhost:8006/v1/models &> /dev/null; do sleep 5; done' || {
        print_error "Vision server failed to start"
        print_info "Check logs: docker compose logs vllm_vision"
        exit 1
    }
    print_success "Vision model ready"
    
    # vLLM Reasoning
    print_info "Waiting for Reasoning model... (may take 2-3 minutes)"
    timeout 180 bash -c 'until curl -s http://localhost:8005/v1/models &> /dev/null; do sleep 5; done' || {
        print_error "Reasoning server failed to start"
        print_info "Check logs: docker compose logs vllm_reasoning"
        exit 1
    }
    print_success "Reasoning model ready"
    
    # RAG App
    print_info "Waiting for RAG application..."
    timeout 90 bash -c 'until curl -s http://localhost:8000 &> /dev/null; do sleep 2; done' || {
        print_error "RAG application failed to start"
        print_info "Check logs: docker compose logs rag_app"
        exit 1
    }
    print_success "RAG application ready"
    
    echo ""
}

# Show service info
show_info() {
    print_header "Service URLs"
    
    echo -e "${GREEN}RAG Application:${NC}      http://localhost:8000"
    echo -e "${GREEN}ChromaDB:${NC}             http://localhost:8001"
    echo -e "${GREEN}RedisInsight:${NC}         http://localhost:8002"
    echo -e "${GREEN}Vision API:${NC}           http://localhost:8006/v1/models"
    echo -e "${GREEN}Reasoning API:${NC}        http://localhost:8005/v1/models"
    echo ""
    
    print_header "Useful Commands"
    echo "View all logs:          docker compose -f docker-compose.complete.yml logs -f"
    echo "View app logs:          docker compose -f docker-compose.complete.yml logs -f rag_app"
    echo "View vision logs:       docker compose -f docker-compose.complete.yml logs -f vllm_vision"
    echo "View reasoning logs:    docker compose -f docker-compose.complete.yml logs -f vllm_reasoning"
    echo "Check status:           docker compose -f docker-compose.complete.yml ps"
    echo "Stop services:          docker compose -f docker-compose.complete.yml down"
    echo "GPU monitoring:         watch -n 1 nvidia-smi"
    echo ""
}

# Main menu
main_menu() {
    echo ""
    print_header "Docker RAG System - Control Panel"
    echo "1) First-time setup (build + start)"
    echo "2) Start services"
    echo "3) Stop services"
    echo "4) Restart services"
    echo "5) View logs"
    echo "6) Check status"
    echo "7) Clean up (remove containers)"
    echo "8) Full reset (remove everything including volumes)"
    echo "9) Exit"
    echo ""
    read -p "Select option: " choice
    
    case $choice in
        1)
            check_prerequisites
            create_directories
            build_images
            start_services
            wait_for_services
            show_info
            ;;
        2)
            print_info "Starting services..."
            docker compose -f docker-compose.complete.yml up -d
            wait_for_services
            show_info
            ;;
        3)
            print_info "Stopping services..."
            docker compose -f docker-compose.complete.yml down
            print_success "Services stopped"
            ;;
        4)
            print_info "Restarting services..."
            docker compose -f docker-compose.complete.yml restart
            print_success "Services restarted"
            ;;
        5)
            echo "Which service?"
            echo "1) All"
            echo "2) RAG App"
            echo "3) Vision Server"
            echo "4) Reasoning Server"
            echo "5) ChromaDB"
            echo "6) Redis"
            read -p "Select: " log_choice
            
            case $log_choice in
                1) docker compose -f docker-compose.complete.yml logs -f ;;
                2) docker compose -f docker-compose.complete.yml logs -f rag_app ;;
                3) docker compose -f docker-compose.complete.yml logs -f vllm_vision ;;
                4) docker compose -f docker-compose.complete.yml logs -f vllm_reasoning ;;
                5) docker compose -f docker-compose.complete.yml logs -f chromadb ;;
                6) docker compose -f docker-compose.complete.yml logs -f redis ;;
                *) print_error "Invalid choice" ;;
            esac
            ;;
        6)
            print_header "Service Status"
            docker compose -f docker-compose.complete.yml ps
            echo ""
            print_info "GPU Status:"
            nvidia-smi
            ;;
        7)
            read -p "Remove all containers? (y/N): " confirm
            if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
                print_info "Removing containers..."
                docker compose -f docker-compose.complete.yml down
                print_success "Containers removed"
            fi
            ;;
        8)
            read -p "Remove EVERYTHING including downloaded models? (y/N): " confirm
            if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
                print_warning "This will delete:"
                print_warning "- All containers"
                print_warning "- All volumes"
                print_warning "- Downloaded models (~20GB)"
                print_warning "- Processed documents"
                read -p "Are you absolutely sure? (yes/no): " confirm2
                if [ "$confirm2" = "yes" ]; then
                    print_info "Removing everything..."
                    docker compose -f docker-compose.complete.yml down -v
                    rm -rf volumes/
                    print_success "Full cleanup complete"
                else
                    print_info "Cancelled"
                fi
            else
                print_info "Cancelled"
            fi
            ;;
        9)
            print_info "Exiting..."
            exit 0
            ;;
        *)
            print_error "Invalid option"
            ;;
    esac
    
    echo ""
    read -p "Press Enter to continue..."
    main_menu
}

# Run main menu
main_menu
