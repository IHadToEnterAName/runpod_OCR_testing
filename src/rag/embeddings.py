"""
Embedding Module
================
Text embedding using SentenceTransformer.
Original logic from your code preserved exactly.
"""

from sentence_transformers import SentenceTransformer
from typing import List
import os
from config.settings import get_config

# =============================================================================
# CONFIGURATION
# =============================================================================

config = get_config()

# =============================================================================
# EMBEDDING MODEL INITIALIZATION (Original Logic)
# =============================================================================

print("🔄 Loading embedding model...")
embedding_model = SentenceTransformer(
    config.models.embedding_model,
    trust_remote_code=True
)
embedding_model.to('cuda')
embedding_model.max_seq_length = 8192
print(f"✅ Embedding model: {config.models.embedding_model}")

# =============================================================================
# EMBEDDING FUNCTIONS (Original Logic)
# =============================================================================

def embed_documents(texts: List[str]) -> List[List[float]]:
    """
    Embed documents for storage.
    Original logic from your code preserved exactly.
    """
    # Add prefix for documents
    prefixed = [f"search_document: {t}" for t in texts]
    
    # Generate embeddings
    embeddings = embedding_model.encode(
        prefixed,
        batch_size=config.performance.embedding_batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    return embeddings.tolist()

def embed_query(query: str) -> List[float]:
    """
    Embed a single query for retrieval.
    Original logic from your code preserved exactly.
    """
    # Add prefix for query
    prefixed = f"search_query: {query}"
    
    # Generate embedding
    embedding = embedding_model.encode(
        [prefixed],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    return embedding[0].tolist()
