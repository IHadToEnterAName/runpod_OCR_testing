"""
Configuration Module
====================
Centralized configuration for Visual RAG system.
Byaldi (ColQwen2) + Qwen3-VL-32B-AWQ architecture.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

@dataclass
class ModelConfig:
    """Single Qwen3-VL model endpoint configuration."""

    # Qwen3-VL-32B-AWQ via vLLM (port 8005)
    base_url: str = os.getenv("VLLM_URL", "http://localhost:8005/v1")
    model_name: str = os.getenv("VLLM_MODEL", "QuantTrio/Qwen3-VL-32B-Instruct-AWQ")

# =============================================================================
# BYALDI (ColQwen2) CONFIGURATION
# =============================================================================

@dataclass
class ByaldiConfig:
    """Byaldi visual retrieval configuration."""

    # ColQwen2 model for visual embeddings
    model_name: str = os.getenv("BYALDI_MODEL", "vidore/colqwen2-v0.1")

    # Where to store indexes on disk
    index_path: str = os.getenv("BYALDI_INDEX_PATH", "/workspace/data/indexes")

    # Store page images within the index for retrieval
    store_collection_with_index: bool = True

# =============================================================================
# VISUAL RAG CONFIGURATION
# =============================================================================

@dataclass
class VisualRAGConfig:
    """Visual RAG pipeline configuration."""

    # Search: how many pages Byaldi returns (sent directly to Qwen3-VL)
    search_top_k: int = int(os.getenv("SEARCH_TOP_K", "5"))

    # Grounding: require bounding boxes in responses
    enable_grounding: bool = os.getenv("ENABLE_GROUNDING", "true").lower() == "true"

    # Image quality for page screenshots (DPI)
    image_dpi: int = int(os.getenv("IMAGE_DPI", "200"))

# =============================================================================
# GENERATION PARAMETERS
# =============================================================================

@dataclass
class GenerationConfig:
    """LLM generation settings for Qwen3-VL."""

    temperature: float = float(os.getenv("TEMPERATURE", "0.3"))
    top_p: float = float(os.getenv("TOP_P", "0.85"))
    max_output_tokens: int = int(os.getenv("MAX_TOKENS", "4096"))

    # Auto-continuation when hitting token limits
    enable_auto_continuation: bool = True
    max_continuations: int = 5

# =============================================================================
# PERFORMANCE SETTINGS
# =============================================================================

@dataclass
class PerformanceConfig:
    """Concurrency settings."""

    # Concurrency limit for Qwen3-VL requests
    model_concurrent_limit: int = 2

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

@dataclass
class DatabaseConfig:
    """Redis configuration (ChromaDB removed)."""

    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))

# =============================================================================
# CACHE CONFIGURATION
# =============================================================================

@dataclass
class CacheConfig:
    """Redis caching configuration (simplified - no embedding cache)."""

    enabled: bool = os.getenv("CACHE_ENABLED", "true").lower() == "true"

    # TTL settings (in seconds)
    query_ttl: int = 3600       # 1 hour for search results
    response_ttl: int = 1800    # 30 minutes for LLM responses

# =============================================================================
# ROUTING CONFIGURATION
# =============================================================================

@dataclass
class RoutingConfig:
    """Query routing configuration."""

    enabled: bool = os.getenv("ROUTING_ENABLED", "true").lower() == "true"

    # Search top_k adjustments by query type (pages go directly to generation)
    page_specific_search_k: int = 3
    summarization_search_k: int = 8
    factual_search_k: int = 3
    default_search_k: int = 5

    # Temperature adjustments
    factual_temperature: float = 0.1
    analytical_temperature: float = 0.4
    default_temperature: float = 0.3

# =============================================================================
# TRAFFIC CONTROLLER CONFIGURATION
# =============================================================================

@dataclass
class TrafficConfig:
    """Single-model traffic controller configuration."""

    enabled: bool = True

    # Rate limiting (requests per minute)
    model_rpm: int = int(os.getenv("MODEL_RPM", "60"))

    # Circuit breaker
    failure_threshold: int = int(os.getenv("MODEL_FAILURE_THRESHOLD", "5"))
    recovery_timeout: float = float(os.getenv("MODEL_RECOVERY_TIMEOUT", "20.0"))

    # Health check interval (seconds)
    health_check_interval: float = 30.0

    # Request timeout (seconds)
    model_timeout: float = float(os.getenv("MODEL_TIMEOUT", "120.0"))

# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are a Visual Document Assistant. You analyze document page images provided to you to answer user queries.

CORE RESPONSIBILITIES:
1. Analyze the provided page images (text, layout, tables, charts, handwritten notes).
2. Answer user questions based STRICTLY on the visual evidence provided.
3. Manage specific page requests accurately.

CRITICAL RULES:
1. **Source of Truth:** Use ONLY the visual content in the provided images. Do not use outside knowledge or assumptions to fill gaps.
2. **Page Targeting:** - If the user specifies a page (e.g., "Summarize page 3"), check if that page is available.
   - If yes: Derive the answer ONLY from that page.
   - If no: State "Page [X] is not included in the provided images."
3. **Visual Elements:** Treat tables, graphs, and diagrams as core content. When answering from a chart, provide specific data points to support your answer.
4. **Citations:** Every claim must end with a reference to the page number where the evidence is found. Format: [Page X].
5. **Uncertainty:** If the text is cut off, blurry, or ambiguous, explicitly state: "The document is unclear on this point due to image quality/cropping."
6. **Negative Constraints:** If the answer is not in the images, say "This information is not visible in the retrieved pages." Do not make up an answer.

FORMATTING GUIDELINES:
- Be direct. Start immediately with the answer (no filler phrases like "Based on the document...").
- Use bullet points for lists or extracted data.
- When extracting specific terms, clauses, or dollar amounts, quote them exactly as they appear.
- Language: English only.

EXAMPLES:
User: "What is the total revenue?"
Assistant: The total revenue is $5.2 million [Page 4].

User: "What does page 2 say about liability?"
Assistant: Page 2 is not included in the provided images.
"""

# =============================================================================
# MASTER CONFIG
# =============================================================================

@dataclass
class Config:
    """Master configuration combining all settings."""

    models: ModelConfig = None
    byaldi: ByaldiConfig = None
    visual_rag: VisualRAGConfig = None
    generation: GenerationConfig = None
    performance: PerformanceConfig = None
    database: DatabaseConfig = None
    cache: CacheConfig = None
    routing: RoutingConfig = None
    traffic: TrafficConfig = None

    system_prompt: str = SYSTEM_PROMPT

    def __post_init__(self):
        if self.models is None:
            self.models = ModelConfig()
        if self.byaldi is None:
            self.byaldi = ByaldiConfig()
        if self.visual_rag is None:
            self.visual_rag = VisualRAGConfig()
        if self.generation is None:
            self.generation = GenerationConfig()
        if self.performance is None:
            self.performance = PerformanceConfig()
        if self.database is None:
            self.database = DatabaseConfig()
        if self.cache is None:
            self.cache = CacheConfig()
        if self.routing is None:
            self.routing = RoutingConfig()
        if self.traffic is None:
            self.traffic = TrafficConfig()

# =============================================================================
# FACTORY FUNCTION
# =============================================================================

_config_instance = None

def get_config() -> Config:
    """Get the master configuration singleton."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

os.environ["HF_HOME"] = os.getenv("HF_HOME", "/workspace/huggingface")
