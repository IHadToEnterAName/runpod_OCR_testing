"""
Vector Store Module
===================
ChromaDB operations with reranking and Redis caching support.
"""

import chromadb
from chromadb.config import Settings
import uuid
import re
from typing import List, Dict, Optional
from config.settings import get_config
from rag.reranker import rerank_chunks, rerank_with_hybrid_scoring

# =============================================================================
# CONFIGURATION
# =============================================================================

config = get_config()

# Initialize ChromaDB client (original logic)
chroma_client = chromadb.PersistentClient(
    path=config.database.chroma_persist_dir,
    settings=Settings(anonymized_telemetry=False, allow_reset=True)
)

print(f"✅ ChromaDB at {config.database.chroma_persist_dir}")

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
# RETRIEVAL FUNCTION (Original Logic)
# =============================================================================

def retrieve_chunks(
    collection: chromadb.Collection,
    query: str,
    query_embedding: List[float],
    top_k: int = None,
    page_filter: Optional[int] = None,
    enable_reranking: bool = True,
    use_cache: bool = True
) -> List[Dict]:
    """
    Retrieve chunks from ChromaDB with optional reranking and Redis caching.

    Args:
        collection: ChromaDB collection
        query: User query string
        query_embedding: Pre-computed query embedding
        top_k: Number of results to return
        page_filter: Optional page number to filter by
        enable_reranking: If True, rerank results using cross-encoder
        use_cache: If True, check/store results in Redis cache

    Returns:
        List of chunk dictionaries with text, page, type, and scores
    """
    if top_k is None:
        top_k = config.chunking.max_context_chunks

    collection_name = collection.name

    # Check cache first (only for queries without page filter)
    cache = _get_cache() if use_cache else None
    if cache and not page_filter:
        try:
            # Try exact match cache
            cached_result = cache.get_query_result(query, collection_name)
            if cached_result and len(cached_result) > 0:
                print(f"📦 Cache hit: {len(cached_result)} chunks")
                return cached_result[:top_k]

            # Try semantic cache (similar query matching)
            similar = cache.find_similar_query(query_embedding, collection_name)
            if similar:
                _, cached_chunks = similar
                if cached_chunks and len(cached_chunks) > 0:
                    print(f"📦 Semantic cache hit: {len(cached_chunks)} chunks")
                    return cached_chunks[:top_k]
        except Exception as e:
            print(f"⚠️ Cache lookup failed, falling through to ChromaDB: {e}")

    # For reranking, fetch more candidates initially
    fetch_k = top_k * 2 if enable_reranking else top_k
    print(f"🔍 Retrieval: query='{query[:50]}...', top_k={top_k}, fetch_k={fetch_k}, reranking={enable_reranking}")

    try:
        # Check collection size
        count = collection.count()
        print(f"📊 Collection '{collection_name}' has {count} chunks")

        if count == 0:
            print("⚠️ Collection is empty!")
            return []

        # Check for page filter in query
        page_match = re.search(r'page\s*(\d+)', query.lower())
        if page_match:
            page_filter = int(page_match.group(1))
            print(f"📄 Page filter detected: {page_filter}")

        chunks = []

        # If page filter, try page-specific retrieval first
        if page_filter:
            try:
                page_results = collection.get(
                    where={"page_number": page_filter},
                    include=["documents", "metadatas"]
                )

                if page_results and page_results.get("documents"):
                    print(f"📄 Found {len(page_results['documents'])} chunks from page {page_filter}")

                    for doc, meta in zip(
                        page_results["documents"],
                        page_results["metadatas"]
                    ):
                        chunks.append({
                            "text": doc,
                            "page": meta.get("page_number"),
                            "type": meta.get("chunk_type", "text"),
                            "distance": 0.0  # Page-specific results prioritized
                        })

                    # If we have enough page-specific results, rerank and return
                    if len(chunks) >= top_k // 2:
                        if enable_reranking:
                            chunks = rerank_chunks(query, chunks, top_k=top_k)
                        return chunks[:top_k]

            except Exception as e:
                print(f"⚠️ Page filter failed: {e}")

        # Semantic search
        try:
            # Cap fetch_k to collection size to avoid ChromaDB errors
            safe_fetch_k = min(fetch_k, count)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=max(safe_fetch_k, 1),  # At least 1
                include=["documents", "metadatas", "distances"]
            )

            if results and results.get("documents") and results["documents"][0]:
                for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                ):
                    # Avoid duplicates
                    if not any(c["text"][:50] == doc[:50] for c in chunks):
                        chunks.append({
                            "text": doc,
                            "page": meta.get("page_number"),
                            "type": meta.get("chunk_type", "text"),
                            "distance": dist,
                            "layout_type": meta.get("layout_type"),
                            "themes": meta.get("themes"),
                            "key_entities": meta.get("key_entities")
                        })

                print(f"✅ Retrieved {len(chunks)} candidates for reranking")
            else:
                print("⚠️ No results from semantic search")

        except Exception as e:
            print(f"❌ Semantic search error: {e}")
            import traceback
            traceback.print_exc()

        # Apply reranking if enabled and we have results
        if enable_reranking and chunks:
            print(f"🔄 Reranking {len(chunks)} chunks...")
            chunks = rerank_with_hybrid_scoring(
                query=query,
                chunks=chunks,
                top_k=top_k,
                embedding_weight=0.3,
                rerank_weight=0.7
            )
            print(f"✅ Reranked to top {len(chunks)} chunks")

        final_chunks = chunks[:top_k]

        if not final_chunks:
            print(f"⚠️ Retrieval returned 0 chunks for query: '{query[:80]}'")
        else:
            print(f"✅ Returning {len(final_chunks)} final chunks")

        # Cache the result (only for queries without page filter)
        if cache and not page_filter and final_chunks:
            try:
                cache.set_query_result(query, collection_name, final_chunks)
            except Exception as e:
                print(f"⚠️ Cache write failed: {e}")

        return final_chunks

    except Exception as e:
        print(f"❌ Retrieval error: {e}")
        import traceback
        traceback.print_exc()
        return []


def retrieve_chunks_simple(
    collection: chromadb.Collection,
    query: str,
    query_embedding: List[float],
    top_k: int = None,
    page_filter: Optional[int] = None
) -> List[Dict]:
    """
    Retrieve chunks without reranking (original logic preserved).
    Use this for comparison or when speed is critical.
    """
    return retrieve_chunks(
        collection=collection,
        query=query,
        query_embedding=query_embedding,
        top_k=top_k,
        page_filter=page_filter,
        enable_reranking=False
    )

# =============================================================================
# COLLECTION MANAGEMENT
# =============================================================================

def create_collection(name: str) -> chromadb.Collection:
    """Create or get collection."""
    return chroma_client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}
    )

def delete_collection(name: str):
    """Delete collection and clear associated Redis cache."""
    try:
        chroma_client.delete_collection(name)
        print(f"🗑️ Deleted collection: {name}")

        # Clear Redis cache for this collection
        cache = _get_cache()
        if cache:
            cache.clear_collection_cache(name)
    except:
        pass

def add_chunks_to_collection(
    collection: chromadb.Collection,
    documents: List[str],
    embeddings: List[List[float]],
    metadatas: List[Dict],
    ids: List[str]
):
    """
    Add chunks to collection.
    Original batch logic preserved with stale collection handling.
    """
    batch_size = 50

    for i in range(0, len(documents), batch_size):
        batch_end = min(i + batch_size, len(documents))

        batch_docs = documents[i:batch_end]
        batch_embs = embeddings[i:batch_end]
        batch_metas = metadatas[i:batch_end]
        batch_ids = ids[i:batch_end]

        try:
            # Verify collection still exists before adding
            _ = collection.count()

            collection.add(
                documents=batch_docs,
                embeddings=batch_embs,
                metadatas=batch_metas,
                ids=batch_ids
            )

            print(f"   Added {batch_end}/{len(documents)} chunks...")

        except Exception as e:
            print(f"❌ Error adding batch {i}-{batch_end}: {e}")
            # Try to re-get the collection by name
            try:
                collection_name = collection.name
                print(f"   Attempting to re-get collection '{collection_name}'...")
                collection = chroma_client.get_collection(collection_name)

                # Retry the add operation
                collection.add(
                    documents=batch_docs,
                    embeddings=batch_embs,
                    metadatas=batch_metas,
                    ids=batch_ids
                )
                print(f"   ✅ Retry successful: Added {batch_end}/{len(documents)} chunks")
            except Exception as retry_error:
                print(f"   ❌ Retry failed: {retry_error}")
                raise
