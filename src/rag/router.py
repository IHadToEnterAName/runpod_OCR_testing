"""
Query Router Module
===================
Intelligent query classification and routing for optimized RAG performance.
Routes queries to appropriate retrieval strategies and generation approaches.
"""

import re
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

from config.settings import get_config

config = get_config()


# =============================================================================
# QUERY INTENT CLASSIFICATION
# =============================================================================

class QueryIntent(Enum):
    """Types of user queries."""
    PAGE_SPECIFIC = "page_specific"      # "What's on page 5?"
    SUMMARIZATION = "summarization"       # "Summarize the document"
    FACTUAL_LOOKUP = "factual_lookup"    # "What is the revenue?"
    COMPARISON = "comparison"             # "Compare X and Y"
    ANALYTICAL = "analytical"             # "Why did X happen?"
    DEFINITION = "definition"             # "What is X?"
    LIST_EXTRACTION = "list_extraction"   # "List all the..."
    VISUAL_CONTENT = "visual_content"     # "Describe the chart/image"
    MULTI_HOP = "multi_hop"              # Requires multiple retrievals
    GENERAL = "general"                   # Default fallback


class RetrievalStrategy(Enum):
    """Retrieval strategies based on query type."""
    PAGE_LOOKUP = "page_lookup"           # Direct page filter
    SEMANTIC_SEARCH = "semantic_search"   # Embedding similarity
    HYBRID_SEARCH = "hybrid_search"       # BM25 + Semantic
    BROAD_RETRIEVAL = "broad_retrieval"   # More chunks for summarization
    FOCUSED_RETRIEVAL = "focused"         # Fewer, more relevant chunks
    ITERATIVE = "iterative"               # Multi-step retrieval


class GenerationStrategy(Enum):
    """Generation approaches based on query type."""
    DIRECT_ANSWER = "direct"              # Short, factual response
    DETAILED_EXPLANATION = "detailed"     # Comprehensive response
    STRUCTURED_LIST = "structured_list"   # Bullet points/numbered
    COMPARISON_TABLE = "comparison"       # Side-by-side comparison
    STEP_BY_STEP = "step_by_step"        # Reasoning walkthrough
    SUMMARY = "summary"                   # Condensed overview


@dataclass
class RoutingDecision:
    """Complete routing decision for a query."""
    intent: QueryIntent
    retrieval_strategy: RetrievalStrategy
    generation_strategy: GenerationStrategy

    # Retrieval parameters
    top_k: int = 12
    use_reranking: bool = True
    page_filter: Optional[int] = None

    # Generation parameters
    temperature: float = 0.3
    max_tokens: int = 2048
    system_prompt_modifier: str = ""

    # Confidence and metadata
    confidence: float = 1.0
    reasoning: str = ""


# =============================================================================
# PATTERN-BASED CLASSIFIERS
# =============================================================================

# Page-specific patterns
PAGE_PATTERNS = [
    r'\bpage\s*(\d+)\b',
    r'\bp\.?\s*(\d+)\b',
    r'\bon\s+page\s*(\d+)\b',
    r'\bfrom\s+page\s*(\d+)\b',
]

# Summarization patterns
SUMMARY_PATTERNS = [
    r'\bsummar(y|ize|ise)\b',
    r'\boverview\b',
    r'\bmain\s+points?\b',
    r'\bkey\s+(points?|takeaways?|findings?)\b',
    r'\btl;?dr\b',
    r'\bin\s+brief\b',
    r'\bbrief(ly)?\s+(describe|explain)\b',
]

# Comparison patterns
COMPARISON_PATTERNS = [
    r'\bcompare\b',
    r'\bcomparison\b',
    r'\bdifference\s+between\b',
    r'\bvs\.?\b',
    r'\bversus\b',
    r'\bhow\s+does?\s+.+\s+differ\b',
    r'\bsimilarit(y|ies)\s+(between|of)\b',
]

# Definition patterns
DEFINITION_PATTERNS = [
    r'\bwhat\s+is\s+(a|an|the)?\s*\w+\b',
    r'\bdefine\b',
    r'\bdefinition\s+of\b',
    r'\bmeaning\s+of\b',
    r'\bexplain\s+what\b',
]

# List extraction patterns
LIST_PATTERNS = [
    r'\blist\s+(all|the|every)\b',
    r'\bwhat\s+are\s+(all|the)\b',
    r'\benumerate\b',
    r'\bgive\s+me\s+(a\s+)?list\b',
    r'\bhow\s+many\b',
]

# Analytical/reasoning patterns
ANALYTICAL_PATTERNS = [
    r'\bwhy\b',
    r'\bhow\s+does?\b',
    r'\bexplain\s+why\b',
    r'\bwhat\s+caused\b',
    r'\bwhat\s+is\s+the\s+reason\b',
    r'\banalyze\b',
    r'\bevaluate\b',
    r'\bassess\b',
]

# Visual content patterns
VISUAL_PATTERNS = [
    r'\bchart\b',
    r'\bgraph\b',
    r'\bfigure\b',
    r'\bimage\b',
    r'\bpicture\b',
    r'\bdiagram\b',
    r'\btable\b',
    r'\billustration\b',
    r'\bvisualization\b',
    r'\bdescribe\s+the\s+(chart|graph|figure|image|table)\b',
]

# Multi-hop indicators
MULTI_HOP_PATTERNS = [
    r'\band\s+then\b',
    r'\bafter\s+that\b',
    r'\bbased\s+on\s+.+,\s+what\b',
    r'\busing\s+.+,\s+(find|calculate|determine)\b',
    r'\bfirst\s+.+,\s+then\b',
]


# =============================================================================
# QUERY ROUTER CLASS
# =============================================================================

class QueryRouter:
    """
    Intelligent query router that classifies queries and determines
    optimal retrieval and generation strategies.
    """

    def __init__(self):
        """Initialize the query router."""
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile regex patterns for efficiency."""
        self.page_re = [re.compile(p, re.IGNORECASE) for p in PAGE_PATTERNS]
        self.summary_re = [re.compile(p, re.IGNORECASE) for p in SUMMARY_PATTERNS]
        self.comparison_re = [re.compile(p, re.IGNORECASE) for p in COMPARISON_PATTERNS]
        self.definition_re = [re.compile(p, re.IGNORECASE) for p in DEFINITION_PATTERNS]
        self.list_re = [re.compile(p, re.IGNORECASE) for p in LIST_PATTERNS]
        self.analytical_re = [re.compile(p, re.IGNORECASE) for p in ANALYTICAL_PATTERNS]
        self.visual_re = [re.compile(p, re.IGNORECASE) for p in VISUAL_PATTERNS]
        self.multi_hop_re = [re.compile(p, re.IGNORECASE) for p in MULTI_HOP_PATTERNS]

    def _check_patterns(self, query: str, patterns: List[re.Pattern]) -> Tuple[bool, Optional[re.Match]]:
        """Check if query matches any pattern."""
        for pattern in patterns:
            match = pattern.search(query)
            if match:
                return True, match
        return False, None

    def _extract_page_number(self, query: str) -> Optional[int]:
        """Extract page number from query if present."""
        for pattern in self.page_re:
            match = pattern.search(query)
            if match:
                try:
                    return int(match.group(1))
                except (IndexError, ValueError):
                    pass
        return None

    def classify_intent(self, query: str) -> Tuple[QueryIntent, float, str]:
        """
        Classify the query intent using pattern matching.

        Returns:
            Tuple of (intent, confidence, reasoning)
        """
        query_lower = query.lower().strip()

        # Check page-specific first (highest priority)
        is_page, match = self._check_patterns(query, self.page_re)
        if is_page:
            return QueryIntent.PAGE_SPECIFIC, 0.95, f"Page reference detected: {match.group()}"

        # Check visual content
        is_visual, match = self._check_patterns(query, self.visual_re)
        if is_visual:
            return QueryIntent.VISUAL_CONTENT, 0.9, f"Visual content reference: {match.group()}"

        # Check summarization
        is_summary, match = self._check_patterns(query, self.summary_re)
        if is_summary:
            return QueryIntent.SUMMARIZATION, 0.9, f"Summarization request: {match.group()}"

        # Check comparison
        is_comparison, match = self._check_patterns(query, self.comparison_re)
        if is_comparison:
            return QueryIntent.COMPARISON, 0.85, f"Comparison request: {match.group()}"

        # Check list extraction
        is_list, match = self._check_patterns(query, self.list_re)
        if is_list:
            return QueryIntent.LIST_EXTRACTION, 0.85, f"List extraction: {match.group()}"

        # Check multi-hop
        is_multi_hop, match = self._check_patterns(query, self.multi_hop_re)
        if is_multi_hop:
            return QueryIntent.MULTI_HOP, 0.8, f"Multi-hop reasoning: {match.group()}"

        # Check analytical
        is_analytical, match = self._check_patterns(query, self.analytical_re)
        if is_analytical:
            return QueryIntent.ANALYTICAL, 0.8, f"Analytical question: {match.group()}"

        # Check definition
        is_definition, match = self._check_patterns(query, self.definition_re)
        if is_definition:
            return QueryIntent.DEFINITION, 0.75, f"Definition request: {match.group()}"

        # Default to factual lookup for short queries, general for longer ones
        if len(query_lower.split()) <= 8:
            return QueryIntent.FACTUAL_LOOKUP, 0.6, "Short factual query (default)"
        else:
            return QueryIntent.GENERAL, 0.5, "General query (default)"

    def determine_retrieval_strategy(
        self,
        intent: QueryIntent,
        query: str
    ) -> Tuple[RetrievalStrategy, Dict[str, Any]]:
        """
        Determine the best retrieval strategy for the given intent.

        Returns:
            Tuple of (strategy, parameters)
        """
        params = {
            "top_k": config.visual_rag.search_top_k,
            "use_reranking": True,
            "page_filter": None,
        }

        if intent == QueryIntent.PAGE_SPECIFIC:
            page_num = self._extract_page_number(query)
            return RetrievalStrategy.PAGE_LOOKUP, {
                **params,
                "page_filter": page_num,
                "top_k": 8,  # Fewer chunks for page-specific
                "use_reranking": True,
            }

        elif intent == QueryIntent.SUMMARIZATION:
            return RetrievalStrategy.BROAD_RETRIEVAL, {
                **params,
                "top_k": 20,  # More chunks for summarization
                "use_reranking": True,
            }

        elif intent == QueryIntent.COMPARISON:
            return RetrievalStrategy.BROAD_RETRIEVAL, {
                **params,
                "top_k": 16,  # Need chunks about both items
                "use_reranking": True,
            }

        elif intent == QueryIntent.FACTUAL_LOOKUP:
            return RetrievalStrategy.FOCUSED_RETRIEVAL, {
                **params,
                "top_k": 6,  # Fewer, highly relevant chunks
                "use_reranking": True,
            }

        elif intent == QueryIntent.DEFINITION:
            return RetrievalStrategy.FOCUSED_RETRIEVAL, {
                **params,
                "top_k": 5,
                "use_reranking": True,
            }

        elif intent == QueryIntent.LIST_EXTRACTION:
            return RetrievalStrategy.BROAD_RETRIEVAL, {
                **params,
                "top_k": 15,
                "use_reranking": True,
            }

        elif intent == QueryIntent.ANALYTICAL:
            return RetrievalStrategy.SEMANTIC_SEARCH, {
                **params,
                "top_k": 12,
                "use_reranking": True,
            }

        elif intent == QueryIntent.VISUAL_CONTENT:
            return RetrievalStrategy.SEMANTIC_SEARCH, {
                **params,
                "top_k": 8,
                "use_reranking": True,
            }

        elif intent == QueryIntent.MULTI_HOP:
            return RetrievalStrategy.ITERATIVE, {
                **params,
                "top_k": 10,
                "use_reranking": True,
            }

        else:  # GENERAL
            return RetrievalStrategy.SEMANTIC_SEARCH, params

    def determine_generation_strategy(
        self,
        intent: QueryIntent
    ) -> Tuple[GenerationStrategy, Dict[str, Any]]:
        """
        Determine the best generation strategy for the given intent.

        Returns:
            Tuple of (strategy, parameters)
        """
        base_params = {
            "temperature": config.generation.temperature,
            "max_tokens": config.generation.max_output_tokens,
            "system_prompt_modifier": "",
        }

        if intent == QueryIntent.PAGE_SPECIFIC:
            return GenerationStrategy.DIRECT_ANSWER, {
                **base_params,
                "temperature": 0.2,
                "system_prompt_modifier": "Focus on content from the specified page. Quote directly when relevant.",
            }

        elif intent == QueryIntent.SUMMARIZATION:
            return GenerationStrategy.SUMMARY, {
                **base_params,
                "temperature": 0.3,
                "max_tokens": 1500,
                "system_prompt_modifier": "Provide a comprehensive but concise summary. Highlight key points and main themes.",
            }

        elif intent == QueryIntent.COMPARISON:
            return GenerationStrategy.COMPARISON_TABLE, {
                **base_params,
                "temperature": 0.2,
                "system_prompt_modifier": "Structure your response as a comparison. Highlight similarities and differences clearly.",
            }

        elif intent == QueryIntent.FACTUAL_LOOKUP:
            return GenerationStrategy.DIRECT_ANSWER, {
                **base_params,
                "temperature": 0.1,
                "max_tokens": 1500,  # Increased from 500
                "system_prompt_modifier": "Provide a direct, factual answer. Be precise and cite the source page.",
            }

        elif intent == QueryIntent.DEFINITION:
            return GenerationStrategy.DIRECT_ANSWER, {
                **base_params,
                "temperature": 0.2,
                "max_tokens": 1000,  # Increased from 400
                "system_prompt_modifier": "Provide a clear, concise definition based on the document content.",
            }

        elif intent == QueryIntent.LIST_EXTRACTION:
            return GenerationStrategy.STRUCTURED_LIST, {
                **base_params,
                "temperature": 0.1,
                "system_prompt_modifier": "Format your response as a numbered or bulleted list. Be comprehensive.",
            }

        elif intent == QueryIntent.ANALYTICAL:
            return GenerationStrategy.STEP_BY_STEP, {
                **base_params,
                "temperature": 0.4,
                "max_tokens": 2048,
                "system_prompt_modifier": "Provide analytical reasoning. Explain the 'why' with supporting evidence from the documents.",
            }

        elif intent == QueryIntent.VISUAL_CONTENT:
            return GenerationStrategy.DETAILED_EXPLANATION, {
                **base_params,
                "temperature": 0.3,
                "system_prompt_modifier": "Describe visual content in detail. Explain what the chart/figure shows and its significance.",
            }

        elif intent == QueryIntent.MULTI_HOP:
            return GenerationStrategy.STEP_BY_STEP, {
                **base_params,
                "temperature": 0.3,
                "max_tokens": 2048,
                "system_prompt_modifier": "Break down your reasoning into clear steps. Show how you connected information from different parts.",
            }

        else:  # GENERAL
            return GenerationStrategy.DETAILED_EXPLANATION, base_params

    def route(self, query: str) -> RoutingDecision:
        """
        Main routing method. Analyzes the query and returns a complete routing decision.

        Args:
            query: The user's query string

        Returns:
            RoutingDecision with all parameters for retrieval and generation
        """
        # Classify intent
        intent, confidence, reasoning = self.classify_intent(query)

        # Determine strategies
        retrieval_strategy, retrieval_params = self.determine_retrieval_strategy(intent, query)
        generation_strategy, generation_params = self.determine_generation_strategy(intent)

        # Build routing decision
        decision = RoutingDecision(
            intent=intent,
            retrieval_strategy=retrieval_strategy,
            generation_strategy=generation_strategy,
            top_k=retrieval_params["top_k"],
            use_reranking=retrieval_params["use_reranking"],
            page_filter=retrieval_params.get("page_filter"),
            temperature=generation_params["temperature"],
            max_tokens=generation_params["max_tokens"],
            system_prompt_modifier=generation_params["system_prompt_modifier"],
            confidence=confidence,
            reasoning=reasoning,
        )

        # Log routing decision
        print(f"🔀 Routing: {intent.value} (confidence: {confidence:.2f})")
        print(f"   Retrieval: {retrieval_strategy.value}, top_k={decision.top_k}")
        print(f"   Generation: {generation_strategy.value}, temp={decision.temperature}")

        return decision


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_router_instance: Optional[QueryRouter] = None


def get_router() -> QueryRouter:
    """Get the global query router instance."""
    global _router_instance
    if _router_instance is None:
        _router_instance = QueryRouter()
    return _router_instance


def route_query(query: str) -> RoutingDecision:
    """Convenience function to route a query."""
    router = get_router()
    return router.route(query)


# =============================================================================
# ENHANCED SYSTEM PROMPT BUILDER
# =============================================================================

def build_enhanced_prompt(
    base_prompt: str,
    routing_decision: RoutingDecision
) -> str:
    """
    Build an enhanced system prompt based on routing decision.

    Args:
        base_prompt: The base system prompt
        routing_decision: The routing decision with modifiers

    Returns:
        Enhanced system prompt
    """
    if not routing_decision.system_prompt_modifier:
        return base_prompt

    # Add routing-specific instructions
    enhanced = f"""{base_prompt}

QUERY-SPECIFIC INSTRUCTIONS:
{routing_decision.system_prompt_modifier}"""

    return enhanced
