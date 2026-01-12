#!/bin/bash

# =============================================================================
# Production RAG System - Startup Script
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "${BLUE}"
    echo "============================================================"
    echo "$1"
    echo "============================================================"
    echo -e "${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker not found. Please install Docker."
        exit 1
    fi
    print_success "Docker installed"
    
    # Check Docker Compose
    if ! command -v docker compose &> /dev/null; then
        print_error "Docker Compose not found. Please install Docker Compose."
        exit 1
    fi
    print_success "Docker Compose installed"
    
    # Check NVIDIA Docker
    if ! docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        print_warning "NVIDIA Docker runtime not working. GPU support may be limited."
    else
        print_success "NVIDIA Docker runtime working"
    fi
    
    # Check .env file
    if [ ! -f ".env" ]; then
        print_warning ".env file not found. Copying from .env.example"
        cp .env.example .env
        print_info "Please edit .env with your settings before continuing"
        exit 0
    fi
    print_success ".env file exists"
    
    echo ""
}

# Create directories
create_directories() {
    print_header "Creating Directories"
    
    mkdir -p data/uploads
    mkdir -p data/processed
    mkdir -p data/failed
    mkdir -p src/agent
    mkdir -p src/cache
    mkdir -p src/airflow/dags
    
    print_success "Directories created"
    echo ""
}

# Check vLLM servers
check_vllm() {
    print_header "Checking vLLM Servers"
    
    # Check Vision server
    if curl -s http://localhost:8006/v1/models &> /dev/null; then
        print_success "Vision server (port 8006) is running"
    else
        print_warning "Vision server (port 8006) not responding"
        print_info "Start with: python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-VL-3B-Instruct --port 8006 --device cuda:0"
    fi
    
    # Check Reasoning server
    if curl -s http://localhost:8005/v1/models &> /dev/null; then
        print_success "Reasoning server (port 8005) is running"
    else
        print_warning "Reasoning server (port 8005) not responding"
        print_info "Start with: python -m vllm.entrypoints.openai.api_server --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --port 8005 --device cuda:1"
    fi
    
    echo ""
}

# Start services
start_services() {
    print_header "Starting Services"
    
    print_info "Building Docker images..."
    docker compose build
    
    print_info "Starting containers..."
    docker compose up -d
    
    print_success "Services started"
    echo ""
}

# Wait for services
wait_for_services() {
    print_header "Waiting for Services"
    
    # Wait for Redis
    print_info "Waiting for Redis..."
    timeout 60 bash -c 'until docker exec rag_redis redis-cli ping &> /dev/null; do sleep 1; done' || {
        print_error "Redis failed to start"
        exit 1
    }
    print_success "Redis ready"
    
    # Wait for ChromaDB
    print_info "Waiting for ChromaDB..."
    timeout 60 bash -c 'until curl -s http://localhost:8000/api/v1/heartbeat &> /dev/null; do sleep 1; done' || {
        print_error "ChromaDB failed to start"
        exit 1
    }
    print_success "ChromaDB ready"
    
    # Wait for Airflow (takes longer)
    print_info "Waiting for Airflow... (this may take a minute)"
    timeout 120 bash -c 'until curl -s http://localhost:8080/health &> /dev/null; do sleep 2; done' || {
        print_warning "Airflow may still be initializing"
    }
    print_success "Airflow ready"
    
    # Wait for RAG app
    print_info "Waiting for RAG application..."
    timeout 60 bash -c 'until curl -s http://localhost:7860 &> /dev/null; do sleep 1; done' || {
        print_error "RAG application failed to start"
        exit 1
    }
    print_success "RAG application ready"
    
    echo ""
}

# Show service URLs
show_urls() {
    print_header "Service URLs"
    
    echo -e "${GREEN}RAG Application:${NC}      http://localhost:7860"
    echo -e "${GREEN}Airflow UI:${NC}           http://localhost:8080 (admin/admin)"
    echo -e "${GREEN}RedisInsight:${NC}         http://localhost:8001"
    echo -e "${GREEN}ChromaDB:${NC}             http://localhost:8000"
    echo ""
}

# Show logs
show_logs() {
    print_header "Service Logs"
    print_info "To view logs, use:"
    echo "  docker compose logs -f rag_app      # Application logs"
    echo "  docker compose logs -f airflow_scheduler  # Airflow logs"
    echo "  docker compose logs -f redis        # Redis logs"
    echo ""
}

# Show status
show_status() {
    print_header "Service Status"
    docker compose ps
    echo ""
}

# Main menu
main_menu() {
    echo ""
    print_header "Production RAG System - Control Panel"
    echo "1) Start all services"
    echo "2) Stop all services"
    echo "3) Restart all services"
    echo "4) View service status"
    echo "5) View logs"
    echo "6) Clean up (remove containers and volumes)"
    echo "7) Check prerequisites"
    echo "8) Exit"
    echo ""
    read -p "Select option: " choice
    
    case $choice in
        1)
            check_prerequisites
            create_directories
            check_vllm
            start_services
            wait_for_services
            show_urls
            show_logs
            ;;
        2)
            print_info "Stopping services..."
            docker compose down
            print_success "Services stopped"
            ;;
        3)
            print_info "Restarting services..."
            docker compose restart
            print_success "Services restarted"
            show_urls
            ;;
        4)
            show_status
            ;;
        5)
            echo "Which service?"
            echo "1) RAG App"
            echo "2) Airflow Scheduler"
            echo "3) Airflow Webserver"
            echo "4) Redis"
            echo "5) ChromaDB"
            echo "6) All"
            read -p "Select: " log_choice
            
            case $log_choice in
                1) docker compose logs -f rag_app ;;
                2) docker compose logs -f airflow_scheduler ;;
                3) docker compose logs -f airflow_webserver ;;
                4) docker compose logs -f redis ;;
                5) docker compose logs -f chromadb ;;
                6) docker compose logs -f ;;
                *) print_error "Invalid choice" ;;
            esac
            ;;
        6)
            read -p "This will remove all containers and volumes. Continue? (y/N): " confirm
            if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
                print_info "Cleaning up..."
                docker compose down -v
                print_success "Cleanup complete"
            else
                print_info "Cancelled"
            fi
            ;;
        7)
            check_prerequisites
            check_vllm
            ;;
        8)
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
