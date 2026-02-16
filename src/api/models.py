"""
API Models
===========
Pydantic request/response schemas for the document API.
"""

from typing import List, Optional
from pydantic import BaseModel


# =============================================================================
# UPLOAD
# =============================================================================

class UploadResponse(BaseModel):
    """Response after uploading a document."""
    document_id: str
    document_name: str
    chunk_count: int
    message: str


# =============================================================================
# CHUNKS
# =============================================================================

class ChunkMetadata(BaseModel):
    """Metadata for a single document chunk (page)."""
    document_id: str
    document_name: str
    chunk_index: int
    score: float
    image_base64: str  # Base64-encoded page image


class ChunksResponse(BaseModel):
    """Response for chunk retrieval."""
    query: str
    chunks: List[ChunkMetadata]
    total_results: int


# =============================================================================
# DELETE
# =============================================================================

class DeleteResponse(BaseModel):
    """Response after deleting a document."""
    document_id: str
    message: str


# =============================================================================
# QUERY (vLLM)
# =============================================================================

class QueryRequest(BaseModel):
    """Request to ask a question and get a vLLM-generated answer."""
    query: str
    top_k: int = 5
    document_id: Optional[str] = None
    session_id: Optional[str] = None
    stream: bool = False


class SourcePage(BaseModel):
    """A source page used to generate the answer."""
    page_number: int
    document_name: str
    score: float


class QueryResponse(BaseModel):
    """Non-streaming response with the full vLLM answer."""
    query: str
    answer: str
    sources: List[SourcePage]
    model: str
    intent: str
