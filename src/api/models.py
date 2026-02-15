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
