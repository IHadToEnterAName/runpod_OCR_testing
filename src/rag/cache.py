"""
Redis Caching Module
====================
Simplified caching for Visual RAG: query results and LLM responses.
Embedding cache removed (ColQwen2 handles embeddings via Byaldi).
"""

import json
import hashlib
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

# =============================================================================
# CACHE CONFIGURATION
# =============================================================================

@dataclass
class CacheConfig:
    """Configuration for Redis caching."""
    enabled: bool = True
    host: str = "localhost"
    port: int = 6379
    db: int = 0

    # TTL settings (in seconds)
    query_ttl: int = 3600       # 1 hour for search results
    response_ttl: int = 1800    # 30 minutes for LLM responses

    # Key prefixes
    query_prefix: str = "rag:query:"
    response_prefix: str = "rag:response:"


# =============================================================================
# REDIS CACHE CLASS
# =============================================================================

class RedisCache:
    """
    Redis-based caching for Visual RAG system.

    Caches:
    - Search results (Byaldi page results by query hash)
    - LLM responses (Qwen3-VL outputs)
    """

    def __init__(self, config: Optional[CacheConfig] = None):
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
                db=self.config.db,
                decode_responses=False,
                socket_timeout=5,
                socket_connect_timeout=5
            )

            self.client.ping()
            self._connected = True
            print(f"Redis cache connected: {self.config.host}:{self.config.port}")

        except ImportError:
            print("redis package not installed. Caching disabled.")
            self.config.enabled = False
        except Exception as e:
            print(f"Redis connection failed: {e}. Caching disabled.")
            self.config.enabled = False

    def _hash_key(self, text: str) -> str:
        """Generate a hash key for text."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]

    def _serialize(self, data: Any) -> bytes:
        """Serialize data for Redis storage."""
        return json.dumps(data).encode('utf-8')

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize data from Redis."""
        if data is None:
            return None
        return json.loads(data.decode('utf-8'))

    # =========================================================================
    # QUERY RESULT CACHE
    # =========================================================================

    def get_query_result(self, query: str, index_name: str) -> Optional[List[Dict]]:
        """Get cached search results for a query."""
        if not self._connected:
            return None

        try:
            key = f"{self.config.query_prefix}{index_name}:{self._hash_key(query)}"
            data = self.client.get(key)

            if data:
                print("Query cache hit")
                return self._deserialize(data)
            return None

        except Exception as e:
            print(f"Cache get error: {e}")
            return None

    def set_query_result(self, query: str, index_name: str, results: List[Dict]) -> bool:
        """Cache search results."""
        if not self._connected:
            return False

        try:
            key = f"{self.config.query_prefix}{index_name}:{self._hash_key(query)}"
            self.client.setex(
                key,
                self.config.query_ttl,
                self._serialize(results)
            )
            return True

        except Exception as e:
            print(f"Cache set error: {e}")
            return False

    # =========================================================================
    # RESPONSE CACHE
    # =========================================================================

    def get_response(self, query: str, context_hash: str) -> Optional[str]:
        """Get cached LLM response."""
        if not self._connected:
            return None

        try:
            key = f"{self.config.response_prefix}{self._hash_key(query + context_hash)}"
            data = self.client.get(key)

            if data:
                print("Response cache hit")
                return data.decode('utf-8')
            return None

        except Exception as e:
            print(f"Cache get error: {e}")
            return None

    def set_response(self, query: str, context_hash: str, response: str) -> bool:
        """Cache LLM response."""
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
            print(f"Cache set error: {e}")
            return False

    # =========================================================================
    # CACHE MANAGEMENT
    # =========================================================================

    def clear_index_cache(self, index_name: str) -> int:
        """Clear all cached data for an index."""
        if not self._connected:
            return 0

        try:
            pattern = f"{self.config.query_prefix}{index_name}:*"
            keys = list(self.client.scan_iter(match=pattern))

            if keys:
                deleted = self.client.delete(*keys)
                print(f"Cleared {deleted} cached queries for {index_name}")
                return deleted
            return 0

        except Exception as e:
            print(f"Cache clear error: {e}")
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
                "total_keys": self.client.dbsize(),
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0)
            }
        except Exception as e:
            return {"connected": False, "error": str(e)}


# =============================================================================
# SINGLETON
# =============================================================================

_cache_instance: Optional[RedisCache] = None

def get_cache() -> RedisCache:
    """Get the global cache instance."""
    global _cache_instance

    if _cache_instance is None:
        from config.settings import get_config
        app_config = get_config()

        cache_config = CacheConfig(
            enabled=app_config.cache.enabled,
            host=app_config.database.redis_host,
            port=app_config.database.redis_port,
            query_ttl=app_config.cache.query_ttl,
            response_ttl=app_config.cache.response_ttl,
        )
        _cache_instance = RedisCache(cache_config)

    return _cache_instance
