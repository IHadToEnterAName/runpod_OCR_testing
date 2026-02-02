"""
Visual RAG Pipeline Module
============================
Core pipeline: Byaldi search -> Qwen3-VL rerank -> grounded generation.
Replaces text-based RAG with visual document understanding.
"""

import asyncio
from typing import List, Optional
import chainlit as cl
from openai import AsyncOpenAI

from config.settings import get_config
from utils.helpers import count_tokens
from storage.visual_store import get_visual_store, PageResult
from rag.visual_reranker import visual_rerank
from rag.router import route_query, build_enhanced_prompt, QueryIntent

# =============================================================================
# CONFIGURATION
# =============================================================================

config = get_config()

# Lazy-initialized vLLM client
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

# Concurrency semaphore
_semaphore = None

def _get_semaphore():
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(config.performance.model_concurrent_limit)
    return _semaphore

# =============================================================================
# VISUAL MESSAGE BUILDER
# =============================================================================

def build_visual_messages(
    query: str,
    pages: List[PageResult],
    memory,
    system_prompt: str,
    enable_grounding: bool = True
) -> list:
    """
    Build OpenAI-format messages with page images for Qwen3-VL.

    Args:
        query: User's query
        pages: Reranked page results with images
        memory: Conversation memory
        system_prompt: System prompt (may be routing-enhanced)
        enable_grounding: Whether to request bounding boxes

    Returns:
        List of message dicts for the OpenAI chat API
    """
    messages = [{"role": "system", "content": system_prompt}]

    # Build multi-image user message
    content_items = []

    # Add each page image
    for page in pages:
        content_items.append({
            "type": "text",
            "text": f"[Page {page.page_number} from {page.document_name}]"
        })
        if page.image_base64:
            content_items.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{page.image_base64}"}
            })

    # Add conversation history context (text only, no images)
    history = memory.get_history(2)
    if history:
        history_text = "\n\nRecent conversation:\n"
        for user_q, asst_a in history:
            shortened = asst_a[:200] + "..." if len(asst_a) > 200 else asst_a
            history_text += f"User: {user_q}\nAssistant: {shortened}\n"
        content_items.append({"type": "text", "text": history_text})

    # Add the actual query
    content_items.append({"type": "text", "text": f"\nQuestion: {query}"})

    # Add grounding instruction if enabled
    if enable_grounding:
        content_items.append({
            "type": "text",
            "text": "\nWhen referencing specific data points, provide bounding box coordinates as <box>[ymin, xmin, ymax, xmax]</box> (normalized 0-1000)."
        })

    messages.append({"role": "user", "content": content_items})

    return messages

# =============================================================================
# RESPONSE GENERATION
# =============================================================================

async def generate_response(
    query: str,
    index_name: str,
    memory,
    msg: cl.Message
):
    """
    Generate a visual RAG response.

    Pipeline:
    1. Route the query (determine search/generation parameters)
    2. Byaldi search (top 10 pages by visual similarity)
    3. Qwen3-VL rerank (select top 5 most relevant)
    4. Build multi-image message
    5. Stream response from Qwen3-VL

    Args:
        query: User's query
        index_name: Byaldi index name for this session
        memory: Conversation memory
        msg: Chainlit message for streaming
    """
    try:
        print(f"\n{'='*60}")
        print(f"Query: {query}")

        # STEP 1: Route the query
        routing_decision = route_query(query)
        print(f"Intent: {routing_decision.intent.value}")
        print(f"Confidence: {routing_decision.confidence:.2f}")

        # Determine search parameters from routing
        search_top_k = config.visual_rag.search_top_k
        rerank_top_k = config.visual_rag.rerank_top_k
        temperature = routing_decision.temperature
        max_tokens = routing_decision.max_tokens

        # Adjust search_top_k based on intent
        if routing_decision.intent == QueryIntent.SUMMARIZATION:
            search_top_k = config.routing.summarization_search_k
        elif routing_decision.intent == QueryIntent.FACTUAL_LOOKUP:
            search_top_k = config.routing.factual_search_k
        elif routing_decision.intent == QueryIntent.PAGE_SPECIFIC:
            search_top_k = config.routing.page_specific_search_k

        # STEP 2: Byaldi visual search
        store = get_visual_store()
        search_results = store.search(index_name, query, top_k=search_top_k)

        if not search_results.results:
            await msg.stream_token("No relevant pages found for your query.")
            await msg.update()
            return

        print(f"Search returned {len(search_results.results)} pages")
        for r in search_results.results[:3]:
            print(f"  Page {r.page_number} ({r.document_name}): score={r.score:.3f}")

        # STEP 3: Visual reranking with Qwen3-VL
        reranked_pages = await visual_rerank(
            query=query,
            search_results=search_results.results,
            top_k=rerank_top_k
        )

        print(f"Reranked to {len(reranked_pages)} pages")
        for r in reranked_pages:
            print(f"  Page {r.page_number} ({r.document_name})")

        # STEP 4: Build visual messages
        system_prompt = build_enhanced_prompt(config.system_prompt, routing_decision)

        messages = build_visual_messages(
            query=query,
            pages=reranked_pages,
            memory=memory,
            system_prompt=system_prompt,
            enable_grounding=config.visual_rag.enable_grounding
        )

        print(f"Generation: temp={temperature}, max_tokens={max_tokens}")
        print(f"{'='*60}\n")

        # STEP 5: Stream response with auto-continuation
        client = _get_client()
        semaphore = _get_semaphore()

        full_response = ""
        continuation_count = 0
        max_continuations = config.generation.max_continuations
        enable_auto_continuation = config.generation.enable_auto_continuation
        last_finish_reason = None

        while continuation_count <= max_continuations:
            # Check for cancellation
            cancelled = cl.user_session.get("cancelled", False)
            if cancelled:
                print("Cancelled by user")
                cl.user_session.set("cancelled", False)
                break

            # Build continuation messages
            if continuation_count == 0:
                current_messages = messages
            else:
                # For continuations, use text-only messages (no re-sending images)
                current_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"You were answering: {query}"},
                    {"role": "assistant", "content": full_response},
                    {"role": "user", "content": "Please continue from where you left off. Do not repeat what you already said."}
                ]
                print(f"Continuation {continuation_count}: Resuming generation...")

            async with semaphore:
                stream = await client.chat.completions.create(
                    model=config.models.model_name,
                    messages=current_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=config.generation.top_p,
                    stream=True
                )

                async for chunk in stream:
                    # Check for cancellation mid-stream
                    cancelled = cl.user_session.get("cancelled", False)
                    if cancelled:
                        cl.user_session.set("cancelled", False)
                        break

                    if chunk.choices[0].delta.content:
                        token = chunk.choices[0].delta.content
                        full_response += token
                        await msg.stream_token(token)

                    if chunk.choices[0].finish_reason:
                        last_finish_reason = chunk.choices[0].finish_reason
                        print(f"Stream finished: {last_finish_reason}")
                        break

            # Check if we should continue
            if last_finish_reason == "length" and enable_auto_continuation:
                continuation_count += 1
                print(f"Token limit reached, auto-continuing ({continuation_count}/{max_continuations})...")
                continue
            elif last_finish_reason in ("stop", "eos"):
                print(f"Response completed naturally ({last_finish_reason})")
                break
            else:
                if last_finish_reason == "length" and not enable_auto_continuation:
                    print("Token limit reached but auto-continuation is disabled")
                break

        if continuation_count > max_continuations:
            print(f"Reached max continuations ({max_continuations}), stopping")
            await msg.stream_token("\n\n[Response truncated: reached maximum continuation limit]")

        await msg.update()

        # Save to memory
        if full_response:
            memory.add(query, full_response)

        return full_response

    except Exception as e:
        print(f"Generation error: {e}")
        import traceback
        traceback.print_exc()
        await msg.stream_token(f"Error generating response: {e}")
        await msg.update()
        return f"Error: {e}"
