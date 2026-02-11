"""
Visual RAG Pipeline Module
============================
Core pipeline: Byaldi search -> grounded generation via Qwen3-VL.
"""

import asyncio
import re
from typing import List, Optional
import chainlit as cl
from openai import AsyncOpenAI

from config.settings import get_config
from utils.helpers import count_tokens
from storage.visual_store import get_visual_store, PageResult
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
# BOX TAG FILTER (strips <box>...</box> from streamed output)
# =============================================================================

class BoxTagFilter:
    """Filters <box>...</box> tags from streamed tokens in real-time."""

    def __init__(self):
        self._buffer = ""
        self._in_tag = False

    def feed(self, token: str) -> str:
        """Feed a token, return text safe to display (boxes stripped)."""
        self._buffer += token
        output = ""

        while self._buffer:
            if self._in_tag:
                close_pos = self._buffer.find("</box>")
                if close_pos >= 0:
                    self._buffer = self._buffer[close_pos + 6:]
                    self._in_tag = False
                    continue
                # Could be partial "</box>" at end — keep buffering
                for i in range(1, 7):
                    if self._buffer.endswith("</box>"[:i]):
                        return output
                break  # no partial match either, keep waiting
            else:
                open_pos = self._buffer.find("<box>")
                if open_pos >= 0:
                    output += self._buffer[:open_pos]
                    self._buffer = self._buffer[open_pos + 5:]
                    self._in_tag = True
                    continue
                # Check for partial "<box>" at end of buffer
                for i in range(1, 6):
                    if self._buffer.endswith("<box>"[:i]):
                        output += self._buffer[:-i]
                        self._buffer = self._buffer[-i:]
                        return output
                output += self._buffer
                self._buffer = ""
                break

        return output

    def flush(self) -> str:
        """Flush remaining buffer at end of stream."""
        if self._in_tag:
            self._buffer = ""
            self._in_tag = False
            return ""
        result = self._buffer
        self._buffer = ""
        return result


def strip_box_tags(text: str) -> str:
    """Remove all <box>...</box> tags from text."""
    return re.sub(r'<box>.*?</box>', '', text)


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
            "text": "\nUse visual grounding internally to locate information accurately on the page images. Do NOT include any bounding box coordinates or <box> tags in your response."
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
    2. Byaldi search (top pages by visual similarity)
    3. Build multi-image message
    4. Stream response from Qwen3-VL

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
        temperature = routing_decision.temperature
        max_tokens = routing_decision.max_tokens

        # Adjust search_top_k based on intent
        if routing_decision.intent == QueryIntent.SUMMARIZATION:
            # Check if document is large enough for text-based map-reduce
            store = get_visual_store()
            stats = store.get_stats(index_name)
            total_pages = stats['total_pages']

            if total_pages > config.long_doc.page_threshold:
                file_list = cl.user_session.get("files", [])
                result = await generate_long_doc_summary(
                    query=query,
                    index_name=index_name,
                    file_list=file_list,
                    memory=memory,
                    msg=msg,
                    total_pages=total_pages,
                )
                return result

            search_top_k = config.routing.summarization_search_k
        elif routing_decision.intent == QueryIntent.FACTUAL_LOOKUP:
            search_top_k = config.routing.factual_search_k
        elif routing_decision.intent == QueryIntent.PAGE_SPECIFIC:
            search_top_k = config.routing.page_specific_search_k
        elif routing_decision.intent == QueryIntent.DOCUMENT_SPECIFIC:
            search_top_k = config.routing.document_specific_search_k

        # STEP 2: Byaldi visual search
        store = get_visual_store()

        # For page-specific queries, directly retrieve the requested page
        if routing_decision.intent == QueryIntent.PAGE_SPECIFIC and routing_decision.page_filter:
            target_page = routing_decision.page_filter
            page_result = store.get_page_by_number(index_name, target_page)
            if page_result:
                pages = [page_result]
                print(f"Direct page lookup: Page {target_page}")
            else:
                stats = store.get_stats(index_name)
                total = stats['total_pages']
                await msg.stream_token(
                    f"Page {target_page} is not included in the provided images. "
                    f"The document has {total} page(s)."
                )
                await msg.update()
                return

        # For document-specific queries, retrieve pages from that document
        elif routing_decision.intent == QueryIntent.DOCUMENT_SPECIFIC and routing_decision.document_filter:
            doc_filter = routing_decision.document_filter
            print(f"Document-specific query: filtering to '{doc_filter}'")

            doc_pages = store.get_document_pages(index_name, doc_filter)

            if doc_pages:
                # Cap at 8 pages to stay within vLLM's --limit-mm-per-prompt
                pages = doc_pages[:8]
                print(f"Found {len(doc_pages)} pages for '{doc_filter}', using {len(pages)}")
            else:
                # Fallback: regular search with document filter
                search_results = store.search(
                    index_name, query, top_k=search_top_k, document_filter=doc_filter
                )
                if not search_results.results:
                    await msg.stream_token(
                        f"No pages found for document '{doc_filter}'. "
                        f"Check the document name with /files."
                    )
                    await msg.update()
                    return
                pages = search_results.results

        else:
            search_results = store.search(index_name, query, top_k=search_top_k)

            if not search_results.results:
                await msg.stream_token("No relevant pages found for your query.")
                await msg.update()
                return

            pages = search_results.results

        print(f"Search returned {len(pages)} pages")
        for r in pages:
            print(f"  Page {r.page_number} ({r.document_name}): score={r.score:.3f}")

        # STEP 3: Build visual messages
        system_prompt = build_enhanced_prompt(config.system_prompt, routing_decision)

        messages = build_visual_messages(
            query=query,
            pages=pages,
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
        box_filter = BoxTagFilter()

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

                chunk_count = 0
                async for chunk in stream:
                    # Check for cancellation mid-stream
                    cancelled = cl.user_session.get("cancelled", False)
                    if cancelled:
                        cl.user_session.set("cancelled", False)
                        break

                    chunk_count += 1
                    delta = chunk.choices[0].delta
                    content = delta.content if delta else None

                    # Debug first few chunks
                    if chunk_count <= 5:
                        print(f"Chunk {chunk_count}: content={repr(content)}, finish_reason={chunk.choices[0].finish_reason}, role={getattr(delta, 'role', None)}")

                    if content:
                        full_response += content
                        # Stream filtered output (strips <box>...</box> tags)
                        filtered = box_filter.feed(content)
                        if filtered:
                            await msg.stream_token(filtered)

                    if chunk.choices[0].finish_reason:
                        last_finish_reason = chunk.choices[0].finish_reason
                        print(f"Stream finished: {last_finish_reason} (total chunks: {chunk_count})")
                        break

                if chunk_count == 0:
                    print("WARNING: Stream returned 0 chunks")

                # Flush any remaining buffered text
                remaining = box_filter.flush()
                if remaining:
                    await msg.stream_token(remaining)

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

        # Save clean response to memory (strip any remaining box tags)
        if full_response:
            memory.add(query, strip_box_tags(full_response))

        return full_response

    except Exception as e:
        print(f"Generation error: {e}")
        import traceback
        traceback.print_exc()
        await msg.stream_token(f"Error generating response: {e}")
        await msg.update()
        return f"Error: {e}"


# =============================================================================
# LONG-DOCUMENT SUMMARIZATION (Map-Reduce)
# =============================================================================

MAP_SUMMARY_PROMPT = (
    "You are a document summarization assistant. You will be given text "
    "extracted from a section of a document. Provide a concise summary of "
    "the key information, main points, and important details in this section. "
    "Be factual and preserve specific numbers, names, dates, and findings. "
    "Do not add information not present in the text."
)

REDUCE_SUMMARY_PROMPT = (
    "You are a document summarization assistant. You will be given multiple "
    "partial summaries from different sections of a large document. Synthesize "
    "them into one coherent, comprehensive summary. Organize by themes or "
    "sections rather than just concatenating. Highlight the most important "
    "findings, conclusions, and key details. Provide a well-structured summary "
    "with clear sections."
)


async def _summarize_chunk(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    chunk: dict,
    query: str,
    chunk_index: int,
    total_chunks: int,
) -> str:
    """Map phase: Summarize a single text chunk via vLLM (text-only, no images)."""
    messages = [
        {"role": "system", "content": MAP_SUMMARY_PROMPT},
        {"role": "user", "content": (
            f"User's request: {query}\n\n"
            f"Document section (pages {chunk['page_start']}-{chunk['page_end']}, "
            f"chunk {chunk_index + 1} of {total_chunks}):\n\n"
            f"{chunk['text']}"
        )}
    ]

    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=config.models.model_name,
                messages=messages,
                max_tokens=config.long_doc.map_max_tokens,
                temperature=config.long_doc.temperature,
                stream=False,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"Map phase error for chunk {chunk_index + 1}: {e}")
            return f"[Error summarizing pages {chunk['page_start']}-{chunk['page_end']}]"


async def generate_long_doc_summary(
    query: str,
    index_name: str,
    file_list: list,
    memory,
    msg: cl.Message,
    total_pages: int,
):
    """
    Generate a summary of a long document using text extraction + map-reduce.

    Pipeline:
    1. Extract text from all PDF pages (PyMuPDF + optional pytesseract OCR)
    2. Chunk text respecting page boundaries
    3. Map: Summarize each chunk concurrently via text-only vLLM calls
    4. Reduce: Combine partial summaries into final summary, streamed to user
    """
    from processing.text_extractor import extract_text_from_pdf, chunk_pages_by_tokens

    try:
        print(f"\n{'='*60}")
        print(f"LONG-DOCUMENT SUMMARIZATION")
        print(f"Query: {query}")
        print(f"Total pages: {total_pages}")
        print(f"Page threshold: {config.long_doc.page_threshold}")

        # --- STEP 1: Collect PDF paths ---
        pdf_paths = []
        for f in file_list:
            if f.get("type") in ("pdf", "docx", "txt") and f.get("path"):
                pdf_paths.append((f["name"], f["path"]))

        if not pdf_paths:
            await msg.stream_token(
                "Cannot perform long-document summarization: PDF file paths "
                "are not available. Please re-upload your documents."
            )
            await msg.update()
            return

        # --- STEP 2: Extract text from all pages ---
        await msg.stream_token("Extracting text from document...\n\n")

        all_pages = []
        for doc_name, pdf_path in pdf_paths:
            try:
                pages = extract_text_from_pdf(pdf_path)
                if len(pdf_paths) > 1:
                    pages = [(pn, f"[{doc_name}]\n{text}") for pn, text in pages]
                all_pages.extend(pages)
            except Exception as e:
                print(f"Text extraction failed for {doc_name}: {e}")
                await msg.stream_token(
                    f"Warning: Could not extract text from {doc_name}.\n"
                )

        if not all_pages or all(
            len(t) < config.long_doc.min_chars_per_page for _, t in all_pages
        ):
            await msg.stream_token(
                "Could not extract sufficient text from the document. "
                "This may be a scanned PDF without OCR support. "
                "Falling back to visual summarization of the most relevant pages.\n\n"
            )
            await msg.update()
            return

        # --- STEP 3: Chunk pages ---
        chunks = chunk_pages_by_tokens(all_pages)

        if not chunks:
            await msg.stream_token("No text content found to summarize.")
            await msg.update()
            return

        print(f"Created {len(chunks)} chunks for map phase")

        # --- STEP 4: Map phase ---
        await msg.stream_token(
            f"Summarizing {total_pages} pages in {len(chunks)} sections...\n\n"
        )

        client = _get_client()
        semaphore = asyncio.Semaphore(config.long_doc.map_concurrency)

        partial_summaries = []
        batch_size = config.long_doc.map_concurrency

        for batch_start in range(0, len(chunks), batch_size):
            cancelled = cl.user_session.get("cancelled", False)
            if cancelled:
                cl.user_session.set("cancelled", False)
                await msg.stream_token("\n\n[Summarization cancelled]")
                await msg.update()
                return

            batch_end = min(batch_start + batch_size, len(chunks))
            batch = chunks[batch_start:batch_end]

            first_chunk = batch[0]
            last_chunk = batch[-1]
            await msg.stream_token(
                f"Processing pages {first_chunk['page_start']}-"
                f"{last_chunk['page_end']}...\n"
            )

            tasks = [
                _summarize_chunk(
                    client=client,
                    semaphore=semaphore,
                    chunk=chunk,
                    query=query,
                    chunk_index=batch_start + j,
                    total_chunks=len(chunks),
                )
                for j, chunk in enumerate(batch)
            ]

            batch_results = await asyncio.gather(*tasks)
            partial_summaries.extend(batch_results)

        print(f"Map phase complete: {len(partial_summaries)} partial summaries")

        # --- STEP 5: Reduce phase ---
        await msg.stream_token("\nCompiling final summary...\n\n---\n\n")

        combined_partials = "\n\n---\n\n".join(
            f"Section {i+1} (pages {chunks[i]['page_start']}-"
            f"{chunks[i]['page_end']}):\n{summary}"
            for i, summary in enumerate(partial_summaries)
            if summary and not summary.startswith("[Error")
        )

        reduce_messages = [
            {"role": "system", "content": REDUCE_SUMMARY_PROMPT},
            {"role": "user", "content": (
                f"User's request: {query}\n\n"
                f"The document has {total_pages} pages across "
                f"{len(pdf_paths)} file(s).\n\n"
                f"Here are the section summaries:\n\n{combined_partials}"
            )}
        ]

        box_filter = BoxTagFilter()
        full_response = ""

        async with _get_semaphore():
            stream = await client.chat.completions.create(
                model=config.models.model_name,
                messages=reduce_messages,
                max_tokens=config.long_doc.reduce_max_tokens,
                temperature=config.long_doc.temperature,
                stream=True,
            )

            async for chunk in stream:
                cancelled = cl.user_session.get("cancelled", False)
                if cancelled:
                    cl.user_session.set("cancelled", False)
                    break

                delta = chunk.choices[0].delta
                content = delta.content if delta else None

                if content:
                    full_response += content
                    filtered = box_filter.feed(content)
                    if filtered:
                        await msg.stream_token(filtered)

                if chunk.choices[0].finish_reason:
                    break

            remaining = box_filter.flush()
            if remaining:
                await msg.stream_token(remaining)

        await msg.update()

        if full_response:
            memory.add(query, strip_box_tags(full_response))

        print(f"Long-document summarization complete")
        print(f"{'='*60}\n")

        return full_response

    except Exception as e:
        print(f"Long-document summarization error: {e}")
        import traceback
        traceback.print_exc()
        await msg.stream_token(f"\nError during summarization: {e}")
        await msg.update()
        return f"Error: {e}"
