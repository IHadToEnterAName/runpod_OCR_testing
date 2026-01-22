"""
Retrieval Strategies Module
===========================
Implements different retrieval strategies for various query types.
Provides specialized retrieval methods for optimal performance.
"""

import re
from typing import List, Dict, Optional, Set
from dataclasses import dataclass

from config.settings import get_config
from rag.router import RetrievalStrategy, RoutingDecision

config = get_config()


# =============================================================================
# RETRIEVAL RESULT
# =============================================================================

@dataclass
class RetrievalResult:
    """Result from a retrieval operation."""
    chunks: List[Dict]
    strategy_used: RetrievalStrategy
    total_retrieved: int
    pages_covered: Set[int]
    metadata: Dict


# =============================================================================
# STRATEGY IMPLEMENTATIONS
# =============================================================================

class RetrievalStrategyExecutor:
    """
    Executes different retrieval strategies based on routing decisions.
    """

    def __init__(self):
        """Initialize the strategy executor."""
        pass

    def execute(
        self,
        collection,
        query: str,
        query_embedding: List[float],
        routing_decision: RoutingDecision
    ) -> RetrievalResult:
        """
        Execute the appropriate retrieval strategy.

        Args:
            collection: ChromaDB collection
            query: User query
            query_embedding: Pre-computed query embedding
            routing_decision: Routing decision with strategy

        Returns:
            RetrievalResult with chunks and metadata
        """
        strategy = routing_decision.retrieval_strategy

        if strategy == RetrievalStrategy.PAGE_LOOKUP:
            return self._page_lookup(collection, query, query_embedding, routing_decision)

        elif strategy == RetrievalStrategy.BROAD_RETRIEVAL:
            return self._broad_retrieval(collection, query, query_embedding, routing_decision)

        elif strategy == RetrievalStrategy.FOCUSED_RETRIEVAL:
            return self._focused_retrieval(collection, query, query_embedding, routing_decision)

        elif strategy == RetrievalStrategy.ITERATIVE:
            return self._iterative_retrieval(collection, query, query_embedding, routing_decision)

        elif strategy == RetrievalStrategy.HYBRID_SEARCH:
            return self._hybrid_search(collection, query, query_embedding, routing_decision)

        else:  # SEMANTIC_SEARCH (default)
            return self._semantic_search(collection, query, query_embedding, routing_decision)

    def _semantic_search(
        self,
        collection,
        query: str,
        query_embedding: List[float],
        routing_decision: RoutingDecision
    ) -> RetrievalResult:
        """Standard semantic search with reranking."""
        from storage.vector_store import retrieve_chunks

        chunks = retrieve_chunks(
            collection=collection,
            query=query,
            query_embedding=query_embedding,
            top_k=routing_decision.top_k,
            page_filter=routing_decision.page_filter,
            enable_reranking=routing_decision.use_reranking,
        )

        pages = {c.get("page") for c in chunks if c.get("page")}

        return RetrievalResult(
            chunks=chunks,
            strategy_used=RetrievalStrategy.SEMANTIC_SEARCH,
            total_retrieved=len(chunks),
            pages_covered=pages,
            metadata={"reranked": routing_decision.use_reranking}
        )

    def _page_lookup(
        self,
        collection,
        query: str,
        query_embedding: List[float],
        routing_decision: RoutingDecision
    ) -> RetrievalResult:
        """
        Page-specific retrieval.
        First gets all chunks from the specified page, then supplements with semantic search.
        """
        from storage.vector_store import retrieve_chunks

        page_num = routing_decision.page_filter
        chunks = []

        # First: Get all chunks from the specified page
        if page_num:
            try:
                page_results = collection.get(
                    where={"page_number": page_num},
                    include=["documents", "metadatas"]
                )

                if page_results and page_results.get("documents"):
                    for doc, meta in zip(
                        page_results["documents"],
                        page_results["metadatas"]
                    ):
                        chunks.append({
                            "text": doc,
                            "page": meta.get("page_number"),
                            "type": meta.get("chunk_type", "text"),
                            "distance": 0.0,  # Page-specific prioritized
                            "source": "page_lookup"
                        })

                    print(f"📄 Page lookup: {len(chunks)} chunks from page {page_num}")

            except Exception as e:
                print(f"⚠️ Page lookup failed: {e}")

        # If not enough chunks, supplement with semantic search
        if len(chunks) < routing_decision.top_k // 2:
            semantic_chunks = retrieve_chunks(
                collection=collection,
                query=query,
                query_embedding=query_embedding,
                top_k=routing_decision.top_k - len(chunks),
                enable_reranking=routing_decision.use_reranking,
            )

            # Add non-duplicate semantic results
            seen_texts = {c["text"][:100] for c in chunks}
            for chunk in semantic_chunks:
                if chunk["text"][:100] not in seen_texts:
                    chunk["source"] = "semantic_supplement"
                    chunks.append(chunk)
                    seen_texts.add(chunk["text"][:100])

        # Rerank combined results if we have mixed sources
        if routing_decision.use_reranking and any(c.get("source") == "semantic_supplement" for c in chunks):
            from rag.reranker import rerank_chunks
            chunks = rerank_chunks(query, chunks, top_k=routing_decision.top_k)

        pages = {c.get("page") for c in chunks if c.get("page")}

        return RetrievalResult(
            chunks=chunks[:routing_decision.top_k],
            strategy_used=RetrievalStrategy.PAGE_LOOKUP,
            total_retrieved=len(chunks),
            pages_covered=pages,
            metadata={"target_page": page_num, "found_on_page": page_num in pages if page_num else False}
        )

    def _broad_retrieval(
        self,
        collection,
        query: str,
        query_embedding: List[float],
        routing_decision: RoutingDecision
    ) -> RetrievalResult:
        """
        Broad retrieval for summarization/comparison.
        Retrieves more chunks and ensures coverage across multiple pages.
        """
        from storage.vector_store import retrieve_chunks

        # Fetch more candidates than needed
        fetch_k = int(routing_decision.top_k * 1.5)

        chunks = retrieve_chunks(
            collection=collection,
            query=query,
            query_embedding=query_embedding,
            top_k=fetch_k,
            enable_reranking=routing_decision.use_reranking,
        )

        # Ensure page diversity - don't let one page dominate
        page_counts = {}
        for chunk in chunks:
            page = chunk.get("page")
            if page:
                page_counts[page] = page_counts.get(page, 0) + 1

        # If one page has more than 40% of chunks, diversify
        if page_counts:
            max_from_one_page = routing_decision.top_k * 0.4
            diverse_chunks = []
            page_taken = {p: 0 for p in page_counts}

            for chunk in chunks:
                page = chunk.get("page")
                if page and page_taken[page] < max_from_one_page:
                    diverse_chunks.append(chunk)
                    page_taken[page] += 1
                elif not page:
                    diverse_chunks.append(chunk)

                if len(diverse_chunks) >= routing_decision.top_k:
                    break

            chunks = diverse_chunks

        pages = {c.get("page") for c in chunks if c.get("page")}

        return RetrievalResult(
            chunks=chunks[:routing_decision.top_k],
            strategy_used=RetrievalStrategy.BROAD_RETRIEVAL,
            total_retrieved=len(chunks),
            pages_covered=pages,
            metadata={"page_distribution": dict(page_counts), "diversified": True}
        )

    def _focused_retrieval(
        self,
        collection,
        query: str,
        query_embedding: List[float],
        routing_decision: RoutingDecision
    ) -> RetrievalResult:
        """
        Focused retrieval for factual lookups.
        Retrieves fewer, highly relevant chunks with aggressive reranking.
        """
        from storage.vector_store import retrieve_chunks

        # Fetch more and aggressively filter
        fetch_k = routing_decision.top_k * 3

        chunks = retrieve_chunks(
            collection=collection,
            query=query,
            query_embedding=query_embedding,
            top_k=fetch_k,
            enable_reranking=True,  # Always rerank for focused
        )

        # Filter out low-confidence chunks if rerank scores available
        if chunks and chunks[0].get("rerank_score") is not None:
            # Keep only chunks with rerank score above threshold
            min_score = 0.0  # Rerank scores can be negative
            if len(chunks) > 3:
                scores = [c.get("rerank_score", 0) for c in chunks[:5]]
                min_score = sum(scores) / len(scores) * 0.5  # 50% of average

            chunks = [c for c in chunks if c.get("rerank_score", 0) >= min_score]

        pages = {c.get("page") for c in chunks if c.get("page")}

        return RetrievalResult(
            chunks=chunks[:routing_decision.top_k],
            strategy_used=RetrievalStrategy.FOCUSED_RETRIEVAL,
            total_retrieved=len(chunks),
            pages_covered=pages,
            metadata={"aggressively_filtered": True}
        )

    def _iterative_retrieval(
        self,
        collection,
        query: str,
        query_embedding: List[float],
        routing_decision: RoutingDecision
    ) -> RetrievalResult:
        """
        Iterative retrieval for multi-hop questions.
        Performs multiple retrieval rounds, refining based on found content.
        """
        from storage.vector_store import retrieve_chunks
        from rag.embeddings import embed_query

        all_chunks = []
        seen_texts = set()
        iterations = 2  # Number of retrieval rounds

        # First round: Standard retrieval
        first_chunks = retrieve_chunks(
            collection=collection,
            query=query,
            query_embedding=query_embedding,
            top_k=routing_decision.top_k // 2,
            enable_reranking=routing_decision.use_reranking,
        )

        for chunk in first_chunks:
            text_key = chunk["text"][:100]
            if text_key not in seen_texts:
                chunk["iteration"] = 1
                all_chunks.append(chunk)
                seen_texts.add(text_key)

        print(f"🔄 Iteration 1: {len(all_chunks)} chunks")

        # Second round: Use entities from first round to expand search
        if all_chunks:
            # Extract key terms from first round results
            first_round_text = " ".join([c["text"][:200] for c in all_chunks[:3]])

            # Create expanded query
            expanded_query = f"{query} {first_round_text[:300]}"
            expanded_embedding = embed_query(expanded_query)

            second_chunks = retrieve_chunks(
                collection=collection,
                query=expanded_query,
                query_embedding=expanded_embedding,
                top_k=routing_decision.top_k // 2,
                enable_reranking=routing_decision.use_reranking,
            )

            for chunk in second_chunks:
                text_key = chunk["text"][:100]
                if text_key not in seen_texts:
                    chunk["iteration"] = 2
                    all_chunks.append(chunk)
                    seen_texts.add(text_key)

            print(f"🔄 Iteration 2: {len(all_chunks)} total chunks")

        # Final rerank of combined results
        if routing_decision.use_reranking and len(all_chunks) > routing_decision.top_k:
            from rag.reranker import rerank_chunks
            all_chunks = rerank_chunks(query, all_chunks, top_k=routing_decision.top_k)

        pages = {c.get("page") for c in all_chunks if c.get("page")}

        return RetrievalResult(
            chunks=all_chunks[:routing_decision.top_k],
            strategy_used=RetrievalStrategy.ITERATIVE,
            total_retrieved=len(all_chunks),
            pages_covered=pages,
            metadata={"iterations": iterations, "expanded_search": True}
        )

    def _hybrid_search(
        self,
        collection,
        query: str,
        query_embedding: List[float],
        routing_decision: RoutingDecision
    ) -> RetrievalResult:
        """
        Hybrid search combining keyword and semantic matching.
        Useful for queries with specific terms that should be matched exactly.
        """
        from storage.vector_store import retrieve_chunks

        # Standard semantic search
        semantic_chunks = retrieve_chunks(
            collection=collection,
            query=query,
            query_embedding=query_embedding,
            top_k=routing_decision.top_k,
            enable_reranking=routing_decision.use_reranking,
        )

        # Extract important keywords from query
        keywords = self._extract_keywords(query)

        if keywords:
            # Boost chunks containing exact keyword matches
            for chunk in semantic_chunks:
                text_lower = chunk["text"].lower()
                keyword_matches = sum(1 for kw in keywords if kw.lower() in text_lower)
                chunk["keyword_boost"] = keyword_matches * 0.1

                # Adjust score
                if "rerank_score" in chunk:
                    chunk["hybrid_score"] = chunk["rerank_score"] + chunk["keyword_boost"]
                elif "distance" in chunk:
                    chunk["hybrid_score"] = (1 - chunk["distance"]) + chunk["keyword_boost"]
                else:
                    chunk["hybrid_score"] = chunk["keyword_boost"]

            # Re-sort by hybrid score
            semantic_chunks.sort(key=lambda x: x.get("hybrid_score", 0), reverse=True)

        pages = {c.get("page") for c in semantic_chunks if c.get("page")}

        return RetrievalResult(
            chunks=semantic_chunks[:routing_decision.top_k],
            strategy_used=RetrievalStrategy.HYBRID_SEARCH,
            total_retrieved=len(semantic_chunks),
            pages_covered=pages,
            metadata={"keywords_used": keywords, "keyword_boosted": bool(keywords)}
        )

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query for hybrid search."""
        # Remove common words
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "need", "dare",
            "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
            "this", "that", "these", "those", "i", "you", "he", "she", "it", "we",
            "they", "me", "him", "her", "us", "them", "my", "your", "his", "her",
            "its", "our", "their", "and", "or", "but", "if", "then", "else", "of",
            "to", "from", "in", "on", "at", "by", "for", "with", "about", "as",
            "into", "through", "during", "before", "after", "above", "below",
            "between", "under", "again", "further", "once", "here", "there",
            "all", "each", "few", "more", "most", "other", "some", "such", "no",
            "nor", "not", "only", "own", "same", "so", "than", "too", "very",
            "just", "also", "now", "tell", "me", "please", "find", "show", "give",
            "document", "page", "file", "text", "content"
        }

        # Tokenize and filter
        words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
        keywords = [w for w in words if w not in stop_words]

        # Also extract quoted phrases
        quoted = re.findall(r'"([^"]+)"', query)
        keywords.extend(quoted)

        # Extract numbers and alphanumeric codes
        codes = re.findall(r'\b[A-Z0-9]{2,}\b', query)
        keywords.extend(codes)

        return list(set(keywords))[:10]  # Limit to 10 keywords


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_executor_instance: Optional[RetrievalStrategyExecutor] = None


def get_strategy_executor() -> RetrievalStrategyExecutor:
    """Get the global strategy executor instance."""
    global _executor_instance
    if _executor_instance is None:
        _executor_instance = RetrievalStrategyExecutor()
    return _executor_instance


def execute_retrieval(
    collection,
    query: str,
    query_embedding: List[float],
    routing_decision: RoutingDecision
) -> RetrievalResult:
    """Convenience function to execute retrieval with routing."""
    executor = get_strategy_executor()
    return executor.execute(collection, query, query_embedding, routing_decision)
