# Configuration Guide

This directory contains centralized configuration for the entire RAG system. All settings can be changed in one place - the `.env` file.

## Quick Start

1. Copy the example environment file:
   ```bash
   cp Configuration/env.example .env
   ```

2. Edit `.env` to customize your settings

3. All Docker services will automatically use these values

## Key Configuration Variables

### Port Configuration

Change these to avoid port conflicts:

```bash
# Application Ports
RAG_APP_PORT=8000              # Chainlit web interface
CHROMA_PORT=8003               # ChromaDB API (external)
REDIS_PORT=6379                # Redis
REDIS_INSIGHT_PORT=8001        # Redis web UI
AIRFLOW_PORT=8080              # Airflow web UI

# vLLM Server Ports
VISION_PORT=8006               # Vision model API
REASONING_PORT=8005            # Reasoning model API
```

### GPU Configuration

Adjust GPU memory and model settings:

```bash
# GPU Selection
CUDA_VISIBLE_DEVICES=0         # Which GPU(s) to use (comma-separated)

# GPU Memory Utilization (0.0-1.0)
VISION_GPU_MEMORY_UTILIZATION=0.3      # Vision model
REASONING_GPU_MEMORY_UTILIZATION=0.54  # Reasoning model

# Model Context Lengths
VISION_MAX_MODEL_LEN=4096       # Vision model max tokens
REASONING_MAX_MODEL_LEN=16384   # Reasoning model max tokens
VISION_MAX_MM_PER_PROMPT=12     # Max images per prompt
```

### Model Configuration

Change models without editing code:

```bash
# Models
VISION_MODEL=Qwen/Qwen2.5-VL-3B-Instruct
REASONING_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5
```

## Common Configuration Changes

### Example 1: Change Port to Avoid Conflict

If port 8000 is already in use:

```bash
# In .env file
RAG_APP_PORT=8080
```

Access the app at `http://localhost:8080` instead.

### Example 2: Increase Model Context Length

To handle longer documents:

```bash
# In .env file
VISION_MAX_MODEL_LEN=8192
REASONING_MAX_MODEL_LEN=32768
```

### Example 3: Adjust GPU Memory

If you're running out of VRAM:

```bash
# In .env file
VISION_GPU_MEMORY_UTILIZATION=0.25
REASONING_GPU_MEMORY_UTILIZATION=0.45
```

### Example 4: Use Different Model

To try a different reasoning model:

```bash
# In .env file
REASONING_MODEL=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
```

## How It Works

1. **Environment Variables**: All settings are defined in `.env`
2. **Docker Compose**: Uses `${VAR_NAME:-default}` syntax to read from `.env`
3. **Application Code**: Uses `os.getenv()` to read environment variables
4. **Single Source of Truth**: Change once, applies everywhere

## File Structure

```
Configuration/
├── env.example       # Template with all settings (copy to .env)
└── README.md         # This file

.env                  # Your actual configuration (gitignored)
```

## Environment Variable Priority

1. **Runtime environment**: Variables set in shell
2. **`.env` file**: Your custom settings
3. **Default values**: Fallback values in docker-compose files

## Best Practices

1. **Never commit `.env`**: It's in `.gitignore` for security
2. **Keep `env.example` updated**: Document new variables there
3. **Use meaningful defaults**: Set sensible defaults in docker-compose
4. **Document changes**: Add comments when changing values

## Troubleshooting

### Ports Already in Use

```bash
# Check what's using a port
sudo lsof -i :8000

# Change port in .env
RAG_APP_PORT=8001
```

### Changes Not Taking Effect

```bash
# Restart services
docker compose down
docker compose up -d
```

### Reset to Defaults

```bash
# Copy fresh env.example
cp Configuration/env.example .env
```

## Advanced Configuration

For detailed configuration options, see:
- `Documentation/DEPLOYMENT_QUICK.md` - Quick deployment guide
- `Documentation/DOCKER_DEPLOYMENT.md` - Comprehensive Docker guide
- `src/config/settings.py` - Application configuration logic
