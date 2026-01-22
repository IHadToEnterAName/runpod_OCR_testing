"""
Semantic Chunking Module
========================
Intelligent text chunking based on semantic similarity.
Creates chunks at natural semantic boundaries rather than fixed character counts.
"""

import re
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer

from config.settings import get_config

# =============================================================================
# CONFIGURATION
# =============================================================================

config = get_config()

@dataclass
class SemanticChunkerConfig:
    """Configuration for semantic chunking."""
    # Similarity threshold for combining sentences (higher = more splits)
    breakpoint_percentile_threshold: int = 95

    # Chunk size constraints
    min_chunk_size: int = 100  # Minimum characters per chunk
    max_chunk_size: int = 1500  # Maximum characters per chunk

    # Buffer for context preservation
    buffer_size: int = 1  # Sentences to include from adjacent chunks

    # Sentence splitting regex
    sentence_split_regex: str = r'(?<=[.!?])\s+'


# =============================================================================
# SEMANTIC CHUNKER CLASS
# =============================================================================

class SemanticChunker:
    """
    Semantic-aware text chunker using embedding similarity.

    This chunker:
    1. Splits text into sentences
    2. Computes embeddings for each sentence
    3. Calculates similarity between consecutive sentences
    4. Identifies semantic breakpoints where topics shift
    5. Groups sentences into semantically coherent chunks
    """

    def __init__(
        self,
        embedding_model: Optional[SentenceTransformer] = None,
        config: Optional[SemanticChunkerConfig] = None
    ):
        """
        Initialize the semantic chunker.

        Args:
            embedding_model: SentenceTransformer model for embeddings.
                            If None, uses the global embedding model.
            config: Chunker configuration. Uses defaults if None.
        """
        self.config = config or SemanticChunkerConfig()
        self._embedding_model = embedding_model
        self._model_loaded = embedding_model is not None

    def _get_model(self) -> SentenceTransformer:
        """Get or lazily load the embedding model."""
        if not self._model_loaded:
            # Import the global embedding model to avoid loading twice
            try:
                from rag.embeddings import embedding_model
                self._embedding_model = embedding_model
            except ImportError:
                # Fallback: load our own model
                self._embedding_model = SentenceTransformer(
                    config.models.embedding_model,
                    trust_remote_code=True
                )
                self._embedding_model.to('cuda')
            self._model_loaded = True
        return self._embedding_model

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Uses regex to split on sentence boundaries while preserving
        structure for better semantic analysis.
        """
        # Clean up text
        text = text.strip()
        if not text:
            return []

        # Split on sentence boundaries
        sentences = re.split(self.config.sentence_split_regex, text)

        # Filter empty sentences and strip whitespace
        sentences = [s.strip() for s in sentences if s.strip()]

        # Handle very short sentences by combining them
        combined = []
        buffer = ""

        for sentence in sentences:
            buffer += (" " if buffer else "") + sentence
            if len(buffer) >= self.config.min_chunk_size // 2:
                combined.append(buffer)
                buffer = ""

        if buffer:
            if combined:
                combined[-1] += " " + buffer
            else:
                combined.append(buffer)

        return combined

    def _compute_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Compute embeddings for sentences."""
        model = self._get_model()
        embeddings = model.encode(
            sentences,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings

    def _calculate_distances(self, embeddings: np.ndarray) -> List[float]:
        """
        Calculate cosine distances between consecutive sentence embeddings.

        Returns distances (1 - similarity) so higher values indicate
        potential semantic breakpoints.
        """
        distances = []
        for i in range(len(embeddings) - 1):
            # Cosine similarity (embeddings are already normalized)
            similarity = np.dot(embeddings[i], embeddings[i + 1])
            distance = 1 - similarity
            distances.append(distance)
        return distances

    def _identify_breakpoints(self, distances: List[float]) -> List[int]:
        """
        Identify indices where semantic breakpoints occur.

        Uses percentile-based threshold to find significant topic shifts.
        """
        if not distances:
            return []

        # Calculate threshold based on percentile
        threshold = np.percentile(distances, self.config.breakpoint_percentile_threshold)

        # Find indices where distance exceeds threshold
        breakpoints = []
        for i, dist in enumerate(distances):
            if dist > threshold:
                breakpoints.append(i + 1)  # Break AFTER this sentence

        return breakpoints

    def _create_chunks(
        self,
        sentences: List[str],
        breakpoints: List[int]
    ) -> List[str]:
        """
        Create chunks from sentences based on breakpoints.

        Respects min/max chunk size constraints.
        """
        if not sentences:
            return []

        # Add start and end as implicit breakpoints
        all_breaks = [0] + breakpoints + [len(sentences)]

        chunks = []
        current_chunk_sentences = []
        current_length = 0

        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence)

            # Check if we're at a breakpoint
            at_breakpoint = i in breakpoints

            # Check if adding this sentence exceeds max size
            would_exceed_max = (current_length + sentence_length) > self.config.max_chunk_size

            # Decide whether to start a new chunk
            should_split = (at_breakpoint and current_length >= self.config.min_chunk_size) or \
                          (would_exceed_max and current_length >= self.config.min_chunk_size)

            if should_split and current_chunk_sentences:
                # Finalize current chunk
                chunk_text = ' '.join(current_chunk_sentences)
                chunks.append(chunk_text)
                current_chunk_sentences = []
                current_length = 0

            # Add sentence to current chunk
            current_chunk_sentences.append(sentence)
            current_length += sentence_length + 1  # +1 for space

        # Don't forget the last chunk
        if current_chunk_sentences:
            chunk_text = ' '.join(current_chunk_sentences)
            chunks.append(chunk_text)

        return chunks

    def split_text(self, text: str) -> List[str]:
        """
        Split text into semantically coherent chunks.

        Args:
            text: Input text to chunk

        Returns:
            List of text chunks
        """
        # Step 1: Split into sentences
        sentences = self._split_sentences(text)

        if len(sentences) <= 1:
            return [text] if text.strip() else []

        # Step 2: Compute embeddings
        embeddings = self._compute_embeddings(sentences)

        # Step 3: Calculate distances between consecutive sentences
        distances = self._calculate_distances(embeddings)

        # Step 4: Identify semantic breakpoints
        breakpoints = self._identify_breakpoints(distances)

        # Step 5: Create chunks
        chunks = self._create_chunks(sentences, breakpoints)

        return chunks

    def split_text_with_metadata(
        self,
        text: str
    ) -> List[Tuple[str, dict]]:
        """
        Split text and return metadata about each chunk.

        Returns:
            List of (chunk_text, metadata) tuples
        """
        sentences = self._split_sentences(text)

        if len(sentences) <= 1:
            return [(text, {"sentence_count": 1, "is_semantic_split": False})] if text.strip() else []

        embeddings = self._compute_embeddings(sentences)
        distances = self._calculate_distances(embeddings)
        breakpoints = self._identify_breakpoints(distances)

        # Track which sentences go into which chunk
        all_breaks = [0] + breakpoints + [len(sentences)]

        results = []
        for i in range(len(all_breaks) - 1):
            start_idx = all_breaks[i]
            end_idx = all_breaks[i + 1]

            chunk_sentences = sentences[start_idx:end_idx]
            chunk_text = ' '.join(chunk_sentences)

            # Calculate average coherence for this chunk
            chunk_distances = distances[start_idx:end_idx-1] if end_idx > start_idx + 1 else []
            avg_coherence = 1 - np.mean(chunk_distances) if chunk_distances else 1.0

            metadata = {
                "sentence_count": len(chunk_sentences),
                "is_semantic_split": True,
                "coherence_score": float(avg_coherence),
                "start_sentence_idx": start_idx,
                "end_sentence_idx": end_idx
            }

            results.append((chunk_text, metadata))

        return results


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_chunker_instance: Optional[SemanticChunker] = None


def get_semantic_chunker(config: Optional[SemanticChunkerConfig] = None) -> SemanticChunker:
    """
    Get the global semantic chunker instance.

    Args:
        config: Optional config (only used on first call)

    Returns:
        SemanticChunker instance
    """
    global _chunker_instance

    if _chunker_instance is None:
        _chunker_instance = SemanticChunker(config=config)

    return _chunker_instance


def semantic_split_text(text: str) -> List[str]:
    """
    Convenience function to split text semantically.

    Args:
        text: Input text

    Returns:
        List of semantically coherent chunks
    """
    chunker = get_semantic_chunker()
    return chunker.split_text(text)


# =============================================================================
# HYBRID CHUNKER (Semantic + Character fallback)
# =============================================================================

class HybridChunker:
    """
    Hybrid chunker that combines semantic and character-based chunking.

    Uses semantic chunking for long texts but falls back to character-based
    chunking for short texts or when semantic chunking produces poor results.
    """

    def __init__(
        self,
        semantic_config: Optional[SemanticChunkerConfig] = None,
        fallback_chunk_size: int = 600,
        fallback_overlap: int = 100,
        min_text_length_for_semantic: int = 500
    ):
        """
        Initialize hybrid chunker.

        Args:
            semantic_config: Config for semantic chunker
            fallback_chunk_size: Character chunk size for fallback
            fallback_overlap: Overlap for character-based fallback
            min_text_length_for_semantic: Minimum text length to use semantic chunking
        """
        self.semantic_chunker = SemanticChunker(config=semantic_config)
        self.fallback_chunk_size = fallback_chunk_size
        self.fallback_overlap = fallback_overlap
        self.min_text_length = min_text_length_for_semantic

        # Fallback character splitter
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        self.char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=fallback_chunk_size,
            chunk_overlap=fallback_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def split_text(self, text: str, prefer_semantic: bool = True) -> List[str]:
        """
        Split text using the best method.

        Args:
            text: Input text
            prefer_semantic: If True, prefer semantic chunking when possible

        Returns:
            List of chunks
        """
        if not text or not text.strip():
            return []

        # Use character-based for short texts
        if len(text) < self.min_text_length:
            return self.char_splitter.split_text(text)

        if prefer_semantic:
            try:
                chunks = self.semantic_chunker.split_text(text)

                # Validate chunks (fallback if poor quality)
                if chunks and all(len(c) >= 50 for c in chunks):
                    return chunks
            except Exception as e:
                print(f"⚠️ Semantic chunking failed, using fallback: {e}")

        # Fallback to character-based
        return self.char_splitter.split_text(text)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

_hybrid_chunker_instance: Optional[HybridChunker] = None


def get_hybrid_chunker() -> HybridChunker:
    """Get the global hybrid chunker instance."""
    global _hybrid_chunker_instance

    if _hybrid_chunker_instance is None:
        # Use settings from centralized config
        semantic_config = SemanticChunkerConfig(
            breakpoint_percentile_threshold=config.chunking.breakpoint_percentile_threshold,
            min_chunk_size=config.chunking.min_semantic_chunk_size,
            max_chunk_size=config.chunking.max_semantic_chunk_size
        )

        _hybrid_chunker_instance = HybridChunker(
            semantic_config=semantic_config,
            fallback_chunk_size=config.chunking.chunk_size,
            fallback_overlap=config.chunking.chunk_overlap,
            min_text_length_for_semantic=config.chunking.min_text_for_semantic
        )

    return _hybrid_chunker_instance


def smart_split_text(text: str, prefer_semantic: bool = True) -> List[str]:
    """
    Smart text splitting that automatically chooses the best method.

    Args:
        text: Input text
        prefer_semantic: If True, prefer semantic chunking when appropriate

    Returns:
        List of chunks
    """
    chunker = get_hybrid_chunker()
    return chunker.split_text(text, prefer_semantic=prefer_semantic)
