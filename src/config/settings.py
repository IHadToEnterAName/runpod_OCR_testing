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
# CHUNKING PARAMETERS (Original Logic)
# =============================================================================

@dataclass
class ChunkingConfig:
    """Document chunking parameters from original code."""
    
    chunk_size: int = 600  # Smaller chunks = better retrieval
    chunk_overlap: int = 100
    max_context_chunks: int = 12  # How many chunks to retrieve

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

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

@dataclass
class DatabaseConfig:
    """Vector store and cache configuration."""
    
    # ChromaDB
    chroma_host: str = os.getenv("CHROMA_HOST", "localhost")
    chroma_port: int = int(os.getenv("CHROMA_PORT", "8001"))
    chroma_persist_dir: str = "/workspace/chroma_db"
    
    # Redis
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))

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
    
    models: ModelConfig = ModelConfig()
    tokens: TokenConfig = TokenConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    performance: PerformanceConfig = PerformanceConfig()
    generation: GenerationConfig = GenerationConfig()
    database: DatabaseConfig = DatabaseConfig()
    
    # System prompt
    system_prompt: str = SYSTEM_PROMPT
    
    # CJK filter
    cjk_ranges: list = None
    
    def __post_init__(self):
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
