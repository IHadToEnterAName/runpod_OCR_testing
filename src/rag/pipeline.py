"""
RAG Pipeline Module
===================
Core RAG logic with intelligent routing, generation and streaming.
Includes query routing for optimized retrieval and generation.
"""

import asyncio
from openai import AsyncOpenAI
from typing import List, Dict, Optional
import chainlit as cl

from config.settings import get_config
from utils.helpers import (
    ThinkingFilter,
    filter_cjk,
    clean_thinking,
    count_tokens,
    format_context
)
from storage.vector_store import retrieve_chunks
from rag.router import (
    QueryRouter,
    RoutingDecision,
    route_query,
    build_enhanced_prompt,
    QueryIntent,
    GenerationStrategy
)
from rag.retrieval_strategies import execute_retrieval, RetrievalResult

# =============================================================================
# CONFIGURATION
# =============================================================================

config = get_config()

# Initialize reasoning client (original logic)
reasoning_client = AsyncOpenAI(
    base_url=config.models.reasoning_base_url,
    api_key="EMPTY"
)

# Semaphore for concurrency (original logic)
llm_semaphore = asyncio.Semaphore(config.performance.llm_concurrent_limit)

# Query router instance
_router = None

def get_router() -> QueryRouter:
    """Get the global query router."""
    global _router
    if _router is None:
        _router = QueryRouter()
    return _router

# =============================================================================
# RESPONSE GENERATION (Original Logic Preserved)
# =============================================================================

async def generate_response(
    query: str,
    collection,
    query_embedding: List[float],
    memory,
    msg: cl.Message,
    use_routing: bool = True
):
    """
    Generate streaming response with intelligent routing.

    Args:
        query: User's query
        collection: ChromaDB collection
        query_embedding: Pre-computed query embedding
        memory: Conversation memory
        msg: Chainlit message for streaming
        use_routing: If True, use intelligent query routing (default)
    """

    thinking_filter = ThinkingFilter()

    try:
        print(f"\n{'='*60}")
        print(f"🔍 Query: {query}")

        # STEP 1: ROUTE THE QUERY
        if use_routing:
            routing_decision = route_query(query)
            print(f"🔀 Intent: {routing_decision.intent.value}")
            print(f"   Strategy: {routing_decision.retrieval_strategy.value}")
            print(f"   Confidence: {routing_decision.confidence:.2f}")
        else:
            routing_decision = None

        # STEP 2: RETRIEVE with routing-optimized parameters
        if use_routing and routing_decision:
            retrieval_result = execute_retrieval(
                collection=collection,
                query=query,
                query_embedding=query_embedding,
                routing_decision=routing_decision
            )
            chunks = retrieval_result.chunks
            print(f"📦 Retrieved {len(chunks)} chunks ({routing_decision.retrieval_strategy.value})")
            print(f"   Pages covered: {sorted(retrieval_result.pages_covered)}")
        else:
            # Fallback to standard retrieval
            chunks = retrieve_chunks(
                collection,
                query,
                query_embedding,
                top_k=config.chunking.max_context_chunks
            )
            print(f"📦 Retrieved {len(chunks)} chunks (standard)")

        if chunks:
            for i, c in enumerate(chunks[:3]):
                score_info = ""
                if c.get("rerank_score"):
                    score_info = f" [score: {c['rerank_score']:.2f}]"
                print(f"   [{i+1}] Page {c.get('page', '?')}{score_info}: {c['text'][:60]}...")

        # STEP 3: FORMAT CONTEXT
        # Adjust context size based on routing
        max_context_tokens = 5000
        if use_routing and routing_decision:
            if routing_decision.intent == QueryIntent.SUMMARIZATION:
                max_context_tokens = 7000  # More context for summaries
            elif routing_decision.intent == QueryIntent.FACTUAL_LOOKUP:
                max_context_tokens = 3000  # Less context, more focused

        context = format_context(chunks, max_tokens=max_context_tokens)
        context_tokens = count_tokens(context)
        print(f"📝 Context: {context_tokens} tokens")

        # STEP 4: BUILD MESSAGES with routing-enhanced prompt
        if use_routing and routing_decision:
            system_prompt = build_enhanced_prompt(config.system_prompt, routing_decision)
        else:
            system_prompt = config.system_prompt

        messages = [{"role": "system", "content": system_prompt}]

        # Add context as system message
        context_msg = f"""## DOCUMENT CONTEXT

{context}

## END CONTEXT

Answer the user's question using ONLY the above content."""

        messages.append({"role": "system", "content": context_msg})

        # Add history (original logic)
        for user_q, asst_a in memory.get_history(2):
            messages.append({"role": "user", "content": user_q})
            shortened = asst_a[:200] + "..." if len(asst_a) > 200 else asst_a
            messages.append({"role": "assistant", "content": shortened})

        # Add current query
        messages.append({"role": "user", "content": query})

        total_tokens = sum(count_tokens(m["content"]) for m in messages)
        print(f"📊 Total input: {total_tokens} tokens")

        # STEP 5: GENERATE with routing-optimized parameters
        if use_routing and routing_decision:
            temperature = routing_decision.temperature
            max_tokens = routing_decision.max_tokens
            print(f"⚡ Generation: temp={temperature}, max_tokens={max_tokens}")
        else:
            temperature = config.generation.temperature
            max_tokens = config.tokens.max_output_tokens

        print(f"{'='*60}\n")

        async with llm_semaphore:
            stream = await reasoning_client.chat.completions.create(
                model=config.models.reasoning_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=config.generation.top_p,
                stream=True
            )

            full_response = ""
            thinking_shown = False

            async for chunk in stream:
                # Check for cancellation
                cancelled = cl.user_session.get("cancelled", False)
                if cancelled:
                    print("🛑 Cancelled")
                    cl.user_session.set("cancelled", False)
                    break

                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content

                    if config.generation.filter_thinking_tags:
                        filtered, is_thinking = thinking_filter.process(token)

                        if is_thinking and not thinking_shown:
                            await msg.stream_token("🤔 ")
                            thinking_shown = True

                        if filtered:
                            clean = filter_cjk(filtered)
                            if clean:
                                full_response += clean
                                await msg.stream_token(clean)
                    else:
                        clean = filter_cjk(token)
                        full_response += clean
                        await msg.stream_token(clean)

                if chunk.choices[0].finish_reason:
                    break

            # Flush remaining (original logic)
            remaining = thinking_filter.flush()
            if remaining:
                clean = filter_cjk(remaining)
                full_response += clean
                await msg.stream_token(clean)

        full_response = clean_thinking(full_response)
        await msg.update()

        # Save to memory (original logic)
        if full_response:
            memory.add(query, full_response)

        return full_response

    except Exception as e:
        print(f"❌ Generation error: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {e}"


# =============================================================================
# LEGACY FUNCTION (without routing)
# =============================================================================

async def generate_response_simple(
    query: str,
    collection,
    query_embedding: List[float],
    memory,
    msg: cl.Message
):
    """
    Generate response without routing (original behavior).
    Use this for comparison or when routing is not desired.
    """
    return await generate_response(
        query=query,
        collection=collection,
        query_embedding=query_embedding,
        memory=memory,
        msg=msg,
        use_routing=False
    )
