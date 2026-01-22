"""
Reranking Module
================
Cross-encoder reranking for improved retrieval precision.
Reranks initial retrieval results using a more accurate (but slower) model.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import torch

# =============================================================================
# RERANKER CONFIGURATION
# =============================================================================

@dataclass
class RerankerConfig:
    """Configuration for the reranker."""
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Fast & accurate
    max_length: int = 512
    batch_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    enabled: bool = True
    top_k_rerank: int = 20  # Rerank top 20, return fewer


# =============================================================================
# RERANKER CLASS
# =============================================================================

class CrossEncoderReranker:
    """
    Cross-encoder reranker using sentence-transformers.

    Cross-encoders process query-document pairs together, providing
    more accurate relevance scores than bi-encoder (embedding) similarity.

    Trade-off: More accurate but slower than embedding similarity.
    """

    def __init__(self, config: Optional[RerankerConfig] = None):
        """
        Initialize the cross-encoder reranker.

        Args:
            config: Reranker configuration. Uses defaults if None.
        """
        self.config = config or RerankerConfig()
        self.model = None
        self._loaded = False

    def _load_model(self):
        """Lazy load the model on first use."""
        if self._loaded:
            return

        try:
            from sentence_transformers import CrossEncoder

            print(f"🔄 Loading reranker model: {self.config.model_name}")
            self.model = CrossEncoder(
                self.config.model_name,
                max_length=self.config.max_length,
                device=self.config.device
            )
            self._loaded = True
            print(f"✅ Reranker loaded on {self.config.device}")

        except ImportError:
            print("⚠️ sentence-transformers not installed. Reranking disabled.")
            self.config.enabled = False
        except Exception as e:
            print(f"⚠️ Failed to load reranker: {e}. Reranking disabled.")
            self.config.enabled = False

    def rerank(
        self,
        query: str,
        chunks: List[Dict],
        top_k: int = None
    ) -> List[Dict]:
        """
        Rerank chunks using cross-encoder scores.

        Args:
            query: The user's query
            chunks: List of chunk dictionaries with 'text' key
            top_k: Number of top results to return (default: half of input)

        Returns:
            Reranked list of chunks with 'rerank_score' added
        """
        if not self.config.enabled:
            return chunks

        if not chunks:
            return chunks

        # Lazy load model
        self._load_model()

        if not self._loaded:
            return chunks

        if top_k is None:
            top_k = max(len(chunks) // 2, 5)

        try:
            # Prepare query-document pairs
            pairs = [(query, chunk.get('text', '')) for chunk in chunks]

            # Get cross-encoder scores
            scores = self.model.predict(
                pairs,
                batch_size=self.config.batch_size,
                show_progress_bar=False
            )

            # Add scores to chunks
            for chunk, score in zip(chunks, scores):
                chunk['rerank_score'] = float(score)

            # Sort by rerank score (descending)
            reranked = sorted(chunks, key=lambda x: x.get('rerank_score', 0), reverse=True)

            print(f"📊 Reranked {len(chunks)} chunks, returning top {top_k}")

            # Log score distribution
            if reranked:
                top_score = reranked[0].get('rerank_score', 0)
                bottom_score = reranked[-1].get('rerank_score', 0)
                print(f"   Score range: {bottom_score:.3f} to {top_score:.3f}")

            return reranked[:top_k]

        except Exception as e:
            print(f"⚠️ Reranking failed: {e}. Returning original order.")
            return chunks[:top_k] if top_k else chunks


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

# Global reranker instance (lazy loaded)
_reranker_instance: Optional[CrossEncoderReranker] = None


def get_reranker(config: Optional[RerankerConfig] = None) -> CrossEncoderReranker:
    """
    Get the global reranker instance.

    Args:
        config: Optional config to use. Only applied on first call.

    Returns:
        CrossEncoderReranker instance
    """
    global _reranker_instance

    if _reranker_instance is None:
        _reranker_instance = CrossEncoderReranker(config)

    return _reranker_instance


def rerank_chunks(
    query: str,
    chunks: List[Dict],
    top_k: int = None,
    enabled: bool = True
) -> List[Dict]:
    """
    Convenience function to rerank chunks.

    Args:
        query: The user's query
        chunks: List of chunk dictionaries
        top_k: Number of results to return
        enabled: Whether to actually rerank (for easy toggling)

    Returns:
        Reranked chunks
    """
    if not enabled:
        return chunks[:top_k] if top_k else chunks

    reranker = get_reranker()
    return reranker.rerank(query, chunks, top_k)


# =============================================================================
# HYBRID SCORING
# =============================================================================

def hybrid_score(
    embedding_distance: float,
    rerank_score: float,
    embedding_weight: float = 0.3,
    rerank_weight: float = 0.7
) -> float:
    """
    Combine embedding distance and rerank score into a hybrid score.

    Args:
        embedding_distance: Cosine distance from embedding search (lower = better)
        rerank_score: Cross-encoder score (higher = better)
        embedding_weight: Weight for embedding similarity (default 0.3)
        rerank_weight: Weight for rerank score (default 0.7)

    Returns:
        Combined score (higher = better)
    """
    # Convert distance to similarity (1 - distance for cosine)
    embedding_similarity = 1.0 - embedding_distance

    # Normalize rerank score to 0-1 range (sigmoid-like)
    normalized_rerank = 1.0 / (1.0 + pow(2.718, -rerank_score))

    return (embedding_weight * embedding_similarity) + (rerank_weight * normalized_rerank)


def rerank_with_hybrid_scoring(
    query: str,
    chunks: List[Dict],
    top_k: int = None,
    embedding_weight: float = 0.3,
    rerank_weight: float = 0.7
) -> List[Dict]:
    """
    Rerank using hybrid scoring that combines embedding and cross-encoder scores.

    Args:
        query: The user's query
        chunks: List of chunks with 'distance' from embedding search
        top_k: Number of results to return
        embedding_weight: Weight for embedding score
        rerank_weight: Weight for rerank score

    Returns:
        Chunks sorted by hybrid score
    """
    reranker = get_reranker()

    if not reranker.config.enabled:
        return chunks[:top_k] if top_k else chunks

    # First, get rerank scores
    reranked = reranker.rerank(query, chunks, top_k=len(chunks))

    # Calculate hybrid scores
    for chunk in reranked:
        distance = chunk.get('distance', 0.5)
        rerank_score = chunk.get('rerank_score', 0.0)
        chunk['hybrid_score'] = hybrid_score(
            distance, rerank_score,
            embedding_weight, rerank_weight
        )

    # Sort by hybrid score
    reranked = sorted(reranked, key=lambda x: x.get('hybrid_score', 0), reverse=True)

    return reranked[:top_k] if top_k else reranked
