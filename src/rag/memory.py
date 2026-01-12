"""
Conversation Memory Module
===========================
Manages conversation history.
Original logic from your code preserved exactly.
"""

import re
from typing import List, Tuple

# =============================================================================
# CONVERSATION MEMORY (Original Logic)
# =============================================================================

class ConversationMemory:
    """
    Manages conversation history.
    Original logic from your code preserved exactly.
    """
    
    def __init__(self):
        self.turns: List[Tuple[str, str]] = []
        self.mentioned_pages: List[int] = []
    
    def add(self, user: str, assistant: str):
        """
        Add a turn to history.
        Original logic preserved.
        """
        self.turns.append((user, assistant))
        
        # Keep only last 10 turns
        if len(self.turns) > 10:
            self.turns.pop(0)
        
        # Track mentioned pages
        pages = re.findall(r'page\s*(\d+)', user.lower())
        self.mentioned_pages.extend([int(p) for p in pages])
    
    def get_history(self, n: int = 3) -> List[Tuple[str, str]]:
        """
        Get last n turns.
        Original logic preserved.
        """
        return self.turns[-n:]
    
    def clear(self):
        """
        Clear all history.
        Original logic preserved.
        """
        self.turns.clear()
        self.mentioned_pages.clear()
