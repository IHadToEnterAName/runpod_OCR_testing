"""
Configuration Module
====================
Centralized configuration for all RAG components.
Preserves all original logic and parameters.
"""

import os
from dataclasses import dataclass
from typing import Optional

# =============================================================================
# MODEL CONFIGURATION (Your Exact Setup)
# =============================================================================

@dataclass
class ModelConfig:
    """Model endpoint configuration matching your vLLM servers."""
    
    # Vision Model (Port 8006)
    vision_base_url: str = os.getenv("VISION_URL", "http://localhost:8006/v1")
    vision_model: str = os.getenv("VISION_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
    
    # Reasoning Model (Port 8005)
    reasoning_base_url: str = os.getenv("REASONING_URL", "http://localhost:8005/v1")
    reasoning_model: str = os.getenv("REASONING_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    
    # Embedding Model
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")

# =============================================================================
# TOKEN LIMITS (Original Logic Preserved)
# =============================================================================

@dataclass
class TokenConfig:
    """Token limits exactly as in your original code."""
    
    # Model limits
    model_max_tokens: int = 16384  # DeepSeek R1 max context
    max_output_tokens: int = 2048  # For generation
    max_input_tokens: int = 16384 - 2048 - 1000  # Reserve 1000 for safety = 13,384
    
    # Vision model limits
    vision_max_tokens: int = 512  # For image descriptions

# =============================================================================
# CHUNKING PARAMETERS
# =============================================================================

@dataclass
class ChunkingConfig:
    """Document chunking parameters with semantic chunking support."""

    # Character-based chunking (fallback)
    chunk_size: int = 600  # Smaller chunks = better retrieval
    chunk_overlap: int = 100
    max_context_chunks: int = 12  # How many chunks to retrieve

    # Semantic chunking settings
    enable_semantic_chunking: bool = True  # Use semantic-aware chunking
    breakpoint_percentile_threshold: int = 95  # Higher = more splits
    min_semantic_chunk_size: int = 100  # Minimum chars per semantic chunk
    max_semantic_chunk_size: int = 1500  # Maximum chars per semantic chunk
    min_text_for_semantic: int = 500  # Min text length to use semantic chunking

# =============================================================================
# PERFORMANCE SETTINGS (Original Logic)
# =============================================================================

@dataclass
class PerformanceConfig:
    """Concurrency and batch settings from original."""
    
    # Concurrency limits
    vision_concurrent_limit: int = 1
    llm_concurrent_limit: int = 2
    
    # Batch processing
    embedding_batch_size: int = 64
    file_processing_workers: int = 8

# =============================================================================
# GENERATION PARAMETERS (Original Logic)
# =============================================================================

@dataclass
class GenerationConfig:
    """LLM generation settings from original code."""

    temperature: float = 0.3
    top_p: float = 0.85

    # Thinking tag filtering
    filter_thinking_tags: bool = True
    thinking_timeout_seconds: int = 30

    # Auto-continuation when hitting token limits
    enable_auto_continuation: bool = True
    max_continuations: int = 5

# =============================================================================
# ENHANCED OCR CONFIGURATION
# =============================================================================

@dataclass
class OCRConfig:
    """Configuration for enhanced OCR features."""

    # Multi-Stage Verification
    enable_verification: bool = True
    confidence_threshold: float = 0.8  # Below this triggers self-correction
    max_correction_attempts: int = 2

    # Layout Awareness
    enable_layout_awareness: bool = True
    detect_tables: bool = True
    detect_columns: bool = True
    preserve_reading_order: bool = True

    # Context Stitching
    enable_context_stitching: bool = True
    max_entities_per_page: int = 20
    max_themes_per_page: int = 5
    context_summary_max_length: int = 500

# =============================================================================
# RERANKER CONFIGURATION
# =============================================================================

@dataclass
class RerankerConfig:
    """Configuration for cross-encoder reranking."""

    # Enable/disable reranking
    enabled: bool = True

    # Model selection (smaller = faster, larger = more accurate)
    # Options:
    #   - "cross-encoder/ms-marco-MiniLM-L-6-v2" (22M params, fast, good quality)
    #   - "cross-encoder/ms-marco-MiniLM-L-12-v2" (33M params, balanced)
    #   - "BAAI/bge-reranker-base" (278M params, high quality)
    #   - "BAAI/bge-reranker-large" (560M params, highest quality, slowest)
    model_name: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Processing settings
    max_length: int = 512  # Max tokens per query-doc pair
    batch_size: int = 32

    # Retrieval settings
    candidates_multiplier: int = 2  # Fetch 2x top_k for reranking

    # Scoring weights (should sum to 1.0)
    embedding_weight: float = 0.3  # Weight for embedding similarity
    rerank_weight: float = 0.7  # Weight for cross-encoder score

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

@dataclass
class DatabaseConfig:
    """Vector store and cache configuration."""

    # ChromaDB
    chroma_host: str = os.getenv("CHROMA_HOST", "localhost")
    chroma_port: int = int(os.getenv("CHROMA_PORT", "8003"))
    chroma_persist_dir: str = "/workspace/chroma_db"

    # Redis
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))

# =============================================================================
# CACHE CONFIGURATION
# =============================================================================

@dataclass
class CacheConfig:
    """Redis caching configuration."""

    # Enable/disable caching
    enabled: bool = True

    # TTL settings (in seconds)
    query_ttl: int = 3600  # 1 hour for query results
    embedding_ttl: int = 86400  # 24 hours for embeddings
    response_ttl: int = 1800  # 30 minutes for LLM responses

    # Semantic similarity threshold for cache hits
    similarity_threshold: float = 0.95

# =============================================================================
# ROUTING CONFIGURATION
# =============================================================================

@dataclass
class RoutingConfig:
    """Query routing configuration."""

    # Enable/disable intelligent routing
    enabled: bool = True

    # Retrieval adjustments by query type
    page_specific_top_k: int = 8        # Fewer chunks for page queries
    summarization_top_k: int = 20       # More chunks for summaries
    factual_lookup_top_k: int = 6       # Focused retrieval
    comparison_top_k: int = 16          # Need chunks about both items
    default_top_k: int = 12             # Standard retrieval

    # Temperature adjustments by query type
    factual_temperature: float = 0.1    # Low temp for facts
    analytical_temperature: float = 0.4 # Higher for reasoning
    default_temperature: float = 0.3    # Standard temperature

    # Context size adjustments
    summarization_context_tokens: int = 7000
    factual_context_tokens: int = 3000
    default_context_tokens: int = 5000

# =============================================================================
# TRAFFIC CONTROLLER CONFIGURATION
# =============================================================================

@dataclass
class TrafficConfig:
    """Traffic controller configuration."""

    # Enable/disable traffic control
    enabled: bool = True

    # Rate limiting (requests per minute)
    vision_rpm: int = 30       # Vision model (slower processing)
    reasoning_rpm: int = 60    # Reasoning model

    # Circuit breaker settings
    vision_failure_threshold: int = 3      # Failures before circuit opens
    reasoning_failure_threshold: int = 5
    vision_recovery_timeout: float = 30.0  # Seconds before retry
    reasoning_recovery_timeout: float = 20.0

    # Health check interval (seconds)
    health_check_interval: float = 30.0

    # Request timeouts (seconds)
    vision_timeout: float = 60.0
    reasoning_timeout: float = 120.0

# =============================================================================
# SYSTEM PROMPT (Original Logic)
# =============================================================================

SYSTEM_PROMPT = """You are a Document Assistant. You have access to document content provided below.

CRITICAL RULES:
1. The DOCUMENT CONTEXT below contains REAL content from uploaded files
2. Use ONLY this content to answer questions  
3. DO NOT say "I don't have access" - the content IS provided
4. For page-specific questions, look for [Page X] markers
5. Quote from documents when helpful
6. If info isn't in the context, say "This isn't in the retrieved sections"

FORMAT:
- Be direct, start with the answer
- Reference page numbers
- No <think> tags
- English only"""

# =============================================================================
# CJK FILTER CONFIGURATION (Original Logic)
# =============================================================================

CJK_RANGES = [
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0x3400, 0x4DBF),  # CJK Extension A
    (0x3040, 0x309F),  # Hiragana
    (0x30A0, 0x30FF),  # Katakana
    (0xAC00, 0xD7AF),  # Hangul
]

# =============================================================================
# MASTER CONFIG (Combines All)
# =============================================================================

@dataclass
class Config:
    """Master configuration combining all settings."""

    models: ModelConfig = None
    tokens: TokenConfig = None
    chunking: ChunkingConfig = None
    performance: PerformanceConfig = None
    generation: GenerationConfig = None
    database: DatabaseConfig = None
    ocr: OCRConfig = None
    reranker: RerankerConfig = None
    cache: CacheConfig = None
    routing: RoutingConfig = None
    traffic: TrafficConfig = None

    # System prompt
    system_prompt: str = SYSTEM_PROMPT

    # CJK filter
    cjk_ranges: list = None

    def __post_init__(self):
        if self.models is None:
            self.models = ModelConfig()
        if self.tokens is None:
            self.tokens = TokenConfig()
        if self.chunking is None:
            self.chunking = ChunkingConfig()
        if self.performance is None:
            self.performance = PerformanceConfig()
        if self.generation is None:
            self.generation = GenerationConfig()
        if self.database is None:
            self.database = DatabaseConfig()
        if self.ocr is None:
            self.ocr = OCRConfig()
        if self.reranker is None:
            self.reranker = RerankerConfig()
        if self.cache is None:
            self.cache = CacheConfig()
        if self.routing is None:
            self.routing = RoutingConfig()
        if self.traffic is None:
            self.traffic = TrafficConfig()
        if self.cjk_ranges is None:
            self.cjk_ranges = CJK_RANGES

# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def get_config() -> Config:
    """Get the master configuration instance."""
    return Config()

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

# Set HuggingFace cache location
os.environ["HF_HOME"] = "/workspace/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/huggingface"
