"""
Redis Caching Module
====================
Semantic caching for queries and embeddings using Redis.
Reduces redundant LLM calls and embedding computations.
"""

import json
import hashlib
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import time

# =============================================================================
# CACHE CONFIGURATION
# =============================================================================

@dataclass
class CacheConfig:
    """Configuration for Redis caching."""
    enabled: bool = True
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0

    # TTL settings (in seconds)
    query_ttl: int = 3600  # 1 hour for query results
    embedding_ttl: int = 86400  # 24 hours for embeddings
    response_ttl: int = 1800  # 30 minutes for LLM responses

    # Semantic similarity threshold for cache hits
    similarity_threshold: float = 0.95

    # Key prefixes
    query_prefix: str = "rag:query:"
    embedding_prefix: str = "rag:embed:"
    response_prefix: str = "rag:response:"


# =============================================================================
# REDIS CACHE CLASS
# =============================================================================

class RedisCache:
    """
    Redis-based caching for RAG system.

    Features:
    - Query result caching (exact match)
    - Embedding caching (avoid recomputation)
    - Semantic cache (similar query matching)
    - Response caching (LLM outputs)
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        """
        Initialize Redis cache.

        Args:
            config: Cache configuration. Uses defaults if None.
        """
        self.config = config or CacheConfig()
        self.client = None
        self._connected = False

        if self.config.enabled:
            self._connect()

    def _connect(self):
        """Establish Redis connection."""
        try:
            import redis

            self.client = redis.Redis(
                host=self.config.host,
                port=self.config.port,
                password=self.config.password,
                db=self.config.db,
                decode_responses=False,  # We'll handle encoding ourselves
                socket_timeout=5,
                socket_connect_timeout=5
            )

            # Test connection
            self.client.ping()
            self._connected = True
            print(f"✅ Redis cache connected: {self.config.host}:{self.config.port}")

        except ImportError:
            print("⚠️ redis package not installed. Caching disabled.")
            self.config.enabled = False
        except Exception as e:
            print(f"⚠️ Redis connection failed: {e}. Caching disabled.")
            self.config.enabled = False

    def _hash_key(self, text: str) -> str:
        """Generate a hash key for text."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]

    def _serialize(self, data: Any) -> bytes:
        """Serialize data for Redis storage."""
        if isinstance(data, np.ndarray):
            return json.dumps(data.tolist()).encode('utf-8')
        elif isinstance(data, list) and data and isinstance(data[0], (int, float)):
            return json.dumps(data).encode('utf-8')
        else:
            return json.dumps(data).encode('utf-8')

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize data from Redis."""
        if data is None:
            return None
        return json.loads(data.decode('utf-8'))

    # =========================================================================
    # EMBEDDING CACHE
    # =========================================================================

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get cached embedding for text.

        Args:
            text: Text to get embedding for

        Returns:
            Cached embedding or None if not found
        """
        if not self._connected:
            return None

        try:
            key = f"{self.config.embedding_prefix}{self._hash_key(text)}"
            data = self.client.get(key)

            if data:
                print(f"📦 Embedding cache hit")
                return self._deserialize(data)

            return None

        except Exception as e:
            print(f"⚠️ Cache get error: {e}")
            return None

    def set_embedding(self, text: str, embedding: List[float]) -> bool:
        """
        Cache an embedding.

        Args:
            text: Original text
            embedding: Embedding vector

        Returns:
            True if cached successfully
        """
        if not self._connected:
            return False

        try:
            key = f"{self.config.embedding_prefix}{self._hash_key(text)}"
            self.client.setex(
                key,
                self.config.embedding_ttl,
                self._serialize(embedding)
            )
            return True

        except Exception as e:
            print(f"⚠️ Cache set error: {e}")
            return False

    def get_embeddings_batch(self, texts: List[str]) -> Tuple[List[Optional[List[float]]], List[int]]:
        """
        Get cached embeddings for multiple texts.

        Args:
            texts: List of texts

        Returns:
            Tuple of (embeddings list with None for misses, indices of misses)
        """
        if not self._connected:
            return [None] * len(texts), list(range(len(texts)))

        try:
            keys = [f"{self.config.embedding_prefix}{self._hash_key(t)}" for t in texts]
            results = self.client.mget(keys)

            embeddings = []
            misses = []

            for i, data in enumerate(results):
                if data:
                    embeddings.append(self._deserialize(data))
                else:
                    embeddings.append(None)
                    misses.append(i)

            hits = len(texts) - len(misses)
            if hits > 0:
                print(f"📦 Embedding cache: {hits}/{len(texts)} hits")

            return embeddings, misses

        except Exception as e:
            print(f"⚠️ Cache batch get error: {e}")
            return [None] * len(texts), list(range(len(texts)))

    def set_embeddings_batch(self, texts: List[str], embeddings: List[List[float]]) -> int:
        """
        Cache multiple embeddings.

        Args:
            texts: List of texts
            embeddings: List of embeddings

        Returns:
            Number of successfully cached embeddings
        """
        if not self._connected:
            return 0

        try:
            pipe = self.client.pipeline()

            for text, embedding in zip(texts, embeddings):
                key = f"{self.config.embedding_prefix}{self._hash_key(text)}"
                pipe.setex(key, self.config.embedding_ttl, self._serialize(embedding))

            pipe.execute()
            return len(texts)

        except Exception as e:
            print(f"⚠️ Cache batch set error: {e}")
            return 0

    # =========================================================================
    # QUERY RESULT CACHE
    # =========================================================================

    def get_query_result(self, query: str, collection_name: str) -> Optional[List[Dict]]:
        """
        Get cached retrieval results for a query.

        Args:
            query: User query
            collection_name: ChromaDB collection name

        Returns:
            Cached chunks or None
        """
        if not self._connected:
            return None

        try:
            key = f"{self.config.query_prefix}{collection_name}:{self._hash_key(query)}"
            data = self.client.get(key)

            if data:
                print(f"📦 Query cache hit")
                return self._deserialize(data)

            return None

        except Exception as e:
            print(f"⚠️ Cache get error: {e}")
            return None

    def set_query_result(self, query: str, collection_name: str, chunks: List[Dict]) -> bool:
        """
        Cache retrieval results.

        Args:
            query: User query
            collection_name: ChromaDB collection name
            chunks: Retrieved chunks

        Returns:
            True if cached successfully
        """
        if not self._connected:
            return False

        try:
            key = f"{self.config.query_prefix}{collection_name}:{self._hash_key(query)}"
            self.client.setex(
                key,
                self.config.query_ttl,
                self._serialize(chunks)
            )
            return True

        except Exception as e:
            print(f"⚠️ Cache set error: {e}")
            return False

    # =========================================================================
    # RESPONSE CACHE (LLM outputs)
    # =========================================================================

    def get_response(self, query: str, context_hash: str) -> Optional[str]:
        """
        Get cached LLM response.

        Args:
            query: User query
            context_hash: Hash of the context used

        Returns:
            Cached response or None
        """
        if not self._connected:
            return None

        try:
            key = f"{self.config.response_prefix}{self._hash_key(query + context_hash)}"
            data = self.client.get(key)

            if data:
                print(f"📦 Response cache hit")
                return data.decode('utf-8')

            return None

        except Exception as e:
            print(f"⚠️ Cache get error: {e}")
            return None

    def set_response(self, query: str, context_hash: str, response: str) -> bool:
        """
        Cache LLM response.

        Args:
            query: User query
            context_hash: Hash of the context used
            response: LLM response

        Returns:
            True if cached successfully
        """
        if not self._connected:
            return False

        try:
            key = f"{self.config.response_prefix}{self._hash_key(query + context_hash)}"
            self.client.setex(
                key,
                self.config.response_ttl,
                response.encode('utf-8')
            )
            return True

        except Exception as e:
            print(f"⚠️ Cache set error: {e}")
            return False

    # =========================================================================
    # SEMANTIC CACHE (Similar query matching)
    # =========================================================================

    def find_similar_query(
        self,
        query_embedding: List[float],
        collection_name: str,
        threshold: float = None
    ) -> Optional[Tuple[str, List[Dict]]]:
        """
        Find a similar cached query using embedding similarity.

        Args:
            query_embedding: Embedding of the current query
            collection_name: Collection to search in
            threshold: Similarity threshold (default from config)

        Returns:
            Tuple of (cached_query, cached_chunks) or None
        """
        if not self._connected:
            return None

        if threshold is None:
            threshold = self.config.similarity_threshold

        try:
            # Get all cached queries for this collection
            pattern = f"{self.config.query_prefix}{collection_name}:*"
            keys = list(self.client.scan_iter(match=pattern, count=100))

            if not keys:
                return None

            # Get embeddings for cached queries
            query_emb = np.array(query_embedding)

            for key in keys[:50]:  # Limit search for performance
                # Get the cached query embedding
                emb_key = key.decode('utf-8').replace(self.config.query_prefix, self.config.embedding_prefix)
                emb_key = emb_key.split(':')[-1]  # Get hash part
                emb_key = f"{self.config.embedding_prefix}{emb_key}"

                cached_emb_data = self.client.get(emb_key)
                if not cached_emb_data:
                    continue

                cached_emb = np.array(self._deserialize(cached_emb_data))

                # Calculate cosine similarity
                similarity = np.dot(query_emb, cached_emb) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(cached_emb)
                )

                if similarity >= threshold:
                    # Found similar query, return cached result
                    cached_chunks = self._deserialize(self.client.get(key))
                    print(f"📦 Semantic cache hit (similarity: {similarity:.3f})")
                    return ("similar_query", cached_chunks)

            return None

        except Exception as e:
            print(f"⚠️ Semantic cache search error: {e}")
            return None

    # =========================================================================
    # CACHE MANAGEMENT
    # =========================================================================

    def clear_collection_cache(self, collection_name: str) -> int:
        """
        Clear all cached data for a collection.

        Args:
            collection_name: Collection to clear cache for

        Returns:
            Number of keys deleted
        """
        if not self._connected:
            return 0

        try:
            pattern = f"{self.config.query_prefix}{collection_name}:*"
            keys = list(self.client.scan_iter(match=pattern))

            if keys:
                deleted = self.client.delete(*keys)
                print(f"🗑️ Cleared {deleted} cached queries for {collection_name}")
                return deleted

            return 0

        except Exception as e:
            print(f"⚠️ Cache clear error: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self._connected:
            return {"connected": False}

        try:
            info = self.client.info()
            return {
                "connected": True,
                "used_memory": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "total_keys": self.client.dbsize(),
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0)
            }

        except Exception as e:
            return {"connected": False, "error": str(e)}


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_cache_instance: Optional[RedisCache] = None


def get_cache(config: Optional[CacheConfig] = None) -> RedisCache:
    """
    Get the global cache instance.

    Args:
        config: Optional config (only used on first call)

    Returns:
        RedisCache instance
    """
    global _cache_instance

    if _cache_instance is None:
        from config.settings import get_config
        app_config = get_config()

        cache_config = CacheConfig(
            enabled=app_config.cache.enabled,
            host=app_config.database.redis_host,
            port=app_config.database.redis_port,
            query_ttl=app_config.cache.query_ttl,
            embedding_ttl=app_config.cache.embedding_ttl,
            response_ttl=app_config.cache.response_ttl,
            similarity_threshold=app_config.cache.similarity_threshold
        )
        _cache_instance = RedisCache(cache_config)

    return _cache_instance
