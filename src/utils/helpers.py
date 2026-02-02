"""
Utility Functions
=================
Helper functions for Visual RAG system.
Simplified: removed CJK filter, thinking tag filter (not needed for Qwen3-VL).
"""

import re
from typing import List
import tiktoken

# =============================================================================
# TOKENIZER
# =============================================================================

try:
    tokenizer = tiktoken.get_encoding("cl100k_base")
except Exception:
    tokenizer = None

def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    if tokenizer:
        return len(tokenizer.encode(text))
    return len(text) // 4  # Fallback estimate

# =============================================================================
# PAGE NUMBER EXTRACTION
# =============================================================================

def extract_page_numbers(text: str) -> List[int]:
    """Extract page numbers from text like 'page 5' or 'p. 10'."""
    pages = re.findall(r'page\s*(\d+)', text.lower())
    return [int(p) for p in pages]
