"""
Visual Reranker Module
=======================
Uses Qwen3-VL to rerank search results by visual relevance.
Replaces cross-encoder text reranking with vision-based reranking.
"""

import re
from typing import List
from openai import AsyncOpenAI

from config.settings import get_config
from storage.visual_store import PageResult

config = get_config()

# Lazy-initialized client
_client = None

def _get_client() -> AsyncOpenAI:
    """Get the Qwen3-VL client."""
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            base_url=config.models.base_url,
            api_key="EMPTY"
        )
    return _client

# =============================================================================
# RERANKING PROMPT
# =============================================================================

RERANK_PROMPT = """You are a document retrieval expert. Given the user's question and {n} document page images, determine which pages are most relevant to answering the question.

User question: {query}

The pages are labeled [Page 1] through [Page {n}] in the order shown.
Rank the top {top_k} most relevant pages.

Return ONLY a comma-separated list of page labels in order of relevance, like:
RANKING: [Page 3], [Page 1], [Page 5], [Page 7], [Page 2]"""

# =============================================================================
# VISUAL RERANKING
# =============================================================================

async def visual_rerank(
    query: str,
    search_results: List[PageResult],
    top_k: int = None
) -> List[PageResult]:
    """
    Use Qwen3-VL to rerank search results by visual relevance.

    Sends all search result page images to the model and asks it to
    rank them by relevance to the query. Returns the top_k most relevant.

    Args:
        query: User's query
        search_results: Page results from Byaldi search
        top_k: Number of pages to return (defaults to config)

    Returns:
        Reranked list of top_k PageResults
    """
    if top_k is None:
        top_k = config.visual_rag.rerank_top_k

    # If we have fewer results than top_k, skip reranking
    if len(search_results) <= top_k:
        return search_results

    client = _get_client()

    # Build multi-image message for reranking
    content = []
    for i, result in enumerate(search_results):
        content.append({
            "type": "text",
            "text": f"[Page {i + 1}] (Document: {result.document_name}, Original Page: {result.page_number})"
        })
        if result.image_base64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{result.image_base64}"}
            })

    # Add the reranking prompt
    content.append({
        "type": "text",
        "text": RERANK_PROMPT.format(
            query=query,
            n=len(search_results),
            top_k=top_k
        )
    })

    try:
        response = await client.chat.completions.create(
            model=config.models.model_name,
            messages=[{"role": "user", "content": content}],
            max_tokens=256,
            temperature=0.0
        )

        ranking_text = response.choices[0].message.content or ""
        print(f"Reranking response: {ranking_text}")

        # Parse the ranking
        ranked_indices = _parse_ranking(ranking_text, len(search_results))

        # Return top_k reranked results
        reranked = []
        for idx in ranked_indices[:top_k]:
            reranked.append(search_results[idx])

        # If parsing failed or returned too few, fill with remaining by original score
        if len(reranked) < top_k:
            used_indices = set(ranked_indices[:top_k])
            for i, result in enumerate(search_results):
                if i not in used_indices and len(reranked) < top_k:
                    reranked.append(result)

        print(f"Reranked: selected {len(reranked)} pages from {len(search_results)} candidates")
        return reranked

    except Exception as e:
        print(f"Warning: Visual reranking failed ({e}), returning top results by score")
        return search_results[:top_k]


def _parse_ranking(text: str, total: int) -> List[int]:
    """
    Parse ranking output from Qwen3-VL.

    Expected format: RANKING: [Page 3], [Page 1], [Page 5]
    Returns 0-indexed list of page indices.
    """
    # Extract page numbers from the ranking line
    matches = re.findall(r'\[Page\s*(\d+)\]', text, re.IGNORECASE)

    if not matches:
        # Fallback: try to find bare numbers after "RANKING:"
        ranking_line = text.split("RANKING:")[-1] if "RANKING:" in text else text
        matches = re.findall(r'(\d+)', ranking_line)

    indices = []
    for m in matches:
        idx = int(m) - 1  # Convert to 0-indexed
        if 0 <= idx < total and idx not in indices:
            indices.append(idx)

    return indices
