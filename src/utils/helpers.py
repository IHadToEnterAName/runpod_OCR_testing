"""
Utility Functions
=================
All helper functions from original code, preserved exactly.
"""

import re
import time
from typing import Tuple, List
import tiktoken
from config.settings import CJK_RANGES

# =============================================================================
# TOKENIZER (Original Logic)
# =============================================================================

try:
    tokenizer = tiktoken.get_encoding("cl100k_base")
except:
    tokenizer = None

def count_tokens(text: str) -> int:
    """Count tokens in text. Original logic preserved."""
    if tokenizer:
        return len(tokenizer.encode(text))
    else:
        return len(text) // 4  # Fallback estimate

# =============================================================================
# CJK FILTER (Original Logic)
# =============================================================================

def filter_cjk(text: str) -> str:
    """
    Filter out CJK characters.
    Original logic from your code preserved exactly.
    """
    return ''.join(
        c for c in text 
        if not any(start <= ord(c) <= end for start, end in CJK_RANGES)
    )

# =============================================================================
# THINKING TAG CLEANER (Original Logic)
# =============================================================================

def clean_thinking(text: str) -> str:
    """
    Remove <think> tags from text.
    Original logic preserved.
    """
    # Remove content between tags
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Remove any standalone tags
    text = re.sub(r'</?think>', '', text)
    return text.strip()

# =============================================================================
# THINKING FILTER CLASS (Original Logic)
# =============================================================================

class ThinkingFilter:
    """
    Streaming filter for thinking tags.
    Original logic from your code preserved exactly.
    """
    
    def __init__(self):
        self.in_thinking = False
        self.buffer = ""
        self.start_time = None
    
    def reset(self):
        """Reset filter state."""
        self.in_thinking = False
        self.buffer = ""
        self.start_time = None
    
    def process(self, token: str) -> Tuple[str, bool]:
        """
        Process a token and return (output, is_thinking).
        Original logic preserved.
        """
        self.buffer += token
        
        # Check for opening tag
        if "<think>" in self.buffer and not self.in_thinking:
            self.in_thinking = True
            self.start_time = time.time()
            before = self.buffer.split("<think>")[0]
            self.buffer = ""
            return before, True
        
        # If in thinking mode
        if self.in_thinking:
            # Check for closing tag
            if "</think>" in self.buffer:
                self.in_thinking = False
                after = self.buffer.split("</think>")[-1]
                self.buffer = after
                self.start_time = None
                return after, False
            
            # Check for timeout
            if self.start_time and (time.time() - self.start_time) > 30:
                self.in_thinking = False
                self.buffer = ""
                return "", False
            
            return "", True
        
        # Not in thinking mode - return buffer
        result = self.buffer
        self.buffer = ""
        return result, False
    
    def flush(self) -> str:
        """Flush remaining buffer."""
        result = self.buffer
        self.buffer = ""
        return result

# =============================================================================
# PAGE NUMBER EXTRACTION (Original Logic)
# =============================================================================

def extract_page_numbers(text: str) -> List[int]:
    """
    Extract page numbers from text like "page 5" or "page 10".
    Original logic preserved.
    """
    pages = re.findall(r'page\s*(\d+)', text.lower())
    return [int(p) for p in pages]

# =============================================================================
# CONTEXT FORMATTING (Original Logic)
# =============================================================================

def format_context(chunks: List[dict], max_tokens: int = 5000) -> str:
    """
    Format retrieved chunks into context string.
    Original logic from your code preserved.
    """
    if not chunks:
        return "[No document content retrieved]"
    
    parts = []
    tokens = 0
    
    for i, chunk in enumerate(chunks):
        page = chunk.get("page", "?")
        chunk_type = chunk.get("type", "text")
        
        # Create header
        header = f"[Page {page}]" if page else f"[Section {i+1}]"
        if chunk_type == "image":
            header += " (Image)"
        
        # Format text
        text = f"{header}\n{chunk['text']}\n"
        text_tokens = count_tokens(text)
        
        # Check if we exceed limit
        if tokens + text_tokens > max_tokens:
            break
        
        parts.append(text)
        tokens += text_tokens
    
    return "\n---\n".join(parts)
