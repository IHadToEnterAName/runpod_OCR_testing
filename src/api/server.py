"""
API Server
===========
Standalone FastAPI application for document management.

Run with:
    cd src && uvicorn api.server:app --host 0.0.0.0 --port 8001 --reload

Or programmatically:
    python -m api.server
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router

# =============================================================================
# APPLICATION
# =============================================================================

app = FastAPI(
    title="Document Chunk API",
    description=(
        "REST API for uploading, searching, and managing document chunks. "
        "Powered by Byaldi (ColQwen2) visual document indexing."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# =============================================================================
# CORS — allow frontends and other backends to connect
# =============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# ROUTES
# =============================================================================

app.include_router(router)


@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {"status": "ok"}


# =============================================================================
# STARTUP
# =============================================================================

@app.on_event("startup")
async def startup():
    """Pre-load the Byaldi model so the first request doesn't wait."""
    from storage.visual_store import get_visual_store
    from api.routes import API_INDEX

    print("=" * 60)
    print("DOCUMENT CHUNK API")
    print("=" * 60)

    store = get_visual_store()
    store.initialize()
    print("Byaldi model loaded.")

    # Load existing API index if present
    if store.index_exists_on_disk(API_INDEX):
        store.load_existing_index(API_INDEX)
        stats = store.get_stats(API_INDEX)
        print(f"Loaded existing index: {stats['total_pages']} pages, {stats['document_count']} documents")

    print("API ready — docs at /docs")
    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
    )
