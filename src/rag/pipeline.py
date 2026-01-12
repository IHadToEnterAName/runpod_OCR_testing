"""
RAG Pipeline Module
===================
Core RAG logic with generation and streaming.
All original logic from your code preserved exactly.
"""

import asyncio
from openai import AsyncOpenAI
from typing import List, Dict
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

# =============================================================================
# RESPONSE GENERATION (Original Logic Preserved)
# =============================================================================

async def generate_response(
    query: str,
    collection,
    query_embedding: List[float],
    memory,
    msg: cl.Message
):
    """
    Generate streaming response with retrieval.
    Original logic from your code preserved EXACTLY.
    """
    
    thinking_filter = ThinkingFilter()
    
    try:
        # STEP 1: RETRIEVE (original logic)
        print(f"\n{'='*50}")
        print(f"🔍 Query: {query}")
        
        chunks = retrieve_chunks(
            collection, 
            query, 
            query_embedding,
            top_k=config.chunking.max_context_chunks
        )
        
        print(f"📦 Retrieved {len(chunks)} chunks")
        if chunks:
            for i, c in enumerate(chunks[:3]):
                print(f"   [{i+1}] Page {c.get('page', '?')}: {c['text'][:80]}...")
        
        # STEP 2: FORMAT CONTEXT (original logic)
        context = format_context(chunks, max_tokens=5000)
        context_tokens = count_tokens(context)
        print(f"📝 Context: {context_tokens} tokens")
        
        # STEP 3: BUILD MESSAGES (original logic)
        messages = [{"role": "system", "content": config.system_prompt}]
        
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
        print(f"{'='*50}\n")
        
        # STEP 4: GENERATE (original logic)
        async with llm_semaphore:
            stream = await reasoning_client.chat.completions.create(
                model=config.models.reasoning_model,
                messages=messages,
                max_tokens=config.tokens.max_output_tokens,
                temperature=config.generation.temperature,
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
