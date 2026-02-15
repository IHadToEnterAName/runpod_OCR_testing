#!/bin/bash
# =============================================================================
# Start both services: Chainlit UI + Document Chunk API
# =============================================================================

CHAINLIT_PORT=${RAG_APP_PORT:-8000}

echo "============================================================"
echo "Starting services..."
echo "  - Chainlit UI        → port ${CHAINLIT_PORT}"
echo "  - Document Chunk API → port ${API_PORT:-8010}"
echo "============================================================"

# Start the API server in the background
cd /workspace/src
python -m uvicorn api.server:app \
    --host ${API_HOST:-0.0.0.0} \
    --port ${API_PORT:-8010} &

# Start Chainlit in the foreground (keeps container alive)
python -m chainlit run /workspace/src/app.py \
    --host 0.0.0.0 \
    --port ${CHAINLIT_PORT}
