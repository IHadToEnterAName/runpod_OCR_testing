"""
Embedding Module
================
Text embedding using SentenceTransformer with Redis caching.
Caches embeddings to avoid redundant computations.
"""

from sentence_transformers import SentenceTransformer
from typing import List, Optional
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
# REDIS CACHE (Lazy initialization)
# =============================================================================

_cache = None

def _get_cache():
    """Get or initialize the Redis cache (lazy loading)."""
    global _cache
    if _cache is None:
        try:
            from rag.cache import get_cache
            _cache = get_cache()
        except Exception as e:
            print(f"⚠️ Redis cache unavailable: {e}")
            _cache = False  # Mark as unavailable
    return _cache if _cache else None

# =============================================================================
# EMBEDDING FUNCTIONS WITH CACHING
# =============================================================================

def embed_documents(texts: List[str], use_cache: bool = True) -> List[List[float]]:
    """
    Embed documents for storage with Redis caching.

    Args:
        texts: List of texts to embed
        use_cache: If True, check/store embeddings in Redis cache

    Returns:
        List of embedding vectors
    """
    if not texts:
        return []

    cache = _get_cache() if use_cache else None

    # Try to get cached embeddings
    if cache:
        cached_embeddings, miss_indices = cache.get_embeddings_batch(texts)

        # If all cached, return immediately
        if not miss_indices:
            print(f"📦 All {len(texts)} embeddings from cache")
            return cached_embeddings

        # Only compute embeddings for misses
        texts_to_embed = [texts[i] for i in miss_indices]
        print(f"🔄 Computing {len(texts_to_embed)}/{len(texts)} embeddings (rest from cache)")
    else:
        texts_to_embed = texts
        miss_indices = list(range(len(texts)))
        cached_embeddings = [None] * len(texts)

    # Add prefix for documents
    prefixed = [f"search_document: {t}" for t in texts_to_embed]

    # Generate embeddings for misses
    new_embeddings = embedding_model.encode(
        prefixed,
        batch_size=config.performance.embedding_batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).tolist()

    # Merge cached and new embeddings
    result = cached_embeddings.copy()
    for idx, emb in zip(miss_indices, new_embeddings):
        result[idx] = emb

    # Cache newly computed embeddings
    if cache and texts_to_embed:
        cache.set_embeddings_batch(texts_to_embed, new_embeddings)

    return result


def embed_query(query: str, use_cache: bool = True) -> List[float]:
    """
    Embed a single query for retrieval with Redis caching.

    Args:
        query: Query text to embed
        use_cache: If True, check/store embedding in Redis cache

    Returns:
        Embedding vector
    """
    cache = _get_cache() if use_cache else None

    # Check cache first
    if cache:
        cached = cache.get_embedding(f"query:{query}")
        if cached:
            return cached

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

    result = embedding[0].tolist()

    # Cache the embedding
    if cache:
        cache.set_embedding(f"query:{query}", result)

    return result


def embed_documents_no_cache(texts: List[str]) -> List[List[float]]:
    """
    Embed documents without caching (for comparison or when cache is problematic).
    Original logic preserved exactly.
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


def embed_query_no_cache(query: str) -> List[float]:
    """
    Embed a single query without caching (original logic preserved).
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
