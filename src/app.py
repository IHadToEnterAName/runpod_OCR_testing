"""
Main Application
================
Chainlit interface for Visual RAG Document Assistant.
Byaldi (ColQwen2) + Qwen3-VL-32B-AWQ architecture.
"""

import chainlit as cl

from config.settings import get_config
from rag.memory import ConversationMemory
from rag.pipeline import generate_response
from storage.visual_store import get_visual_store
from processing.file_processor import process_files
from rag.cache import get_cache

# =============================================================================
# CONFIGURATION
# =============================================================================

config = get_config()

# Persistent index name shared across all sessions
PERSISTENT_INDEX = "documents"

# Pre-load the Byaldi (ColQwen2) model at startup before any user sessions
print("Pre-loading Byaldi model...")
_startup_store = get_visual_store()
_startup_store.initialize()
print("Byaldi model ready.")

# =============================================================================
# CHAINLIT HANDLERS
# =============================================================================

@cl.on_chat_start
async def start():
    """Initialize session with persistent Byaldi index."""

    store = get_visual_store()

    # Check if an existing index is already on disk
    has_existing = store.index_exists_on_disk(PERSISTENT_INDEX)
    if has_existing:
        store.load_existing_index(PERSISTENT_INDEX)
        stats = store.get_stats(PERSISTENT_INDEX)
        page_count = stats["total_pages"]
        doc_count = stats["document_count"]
    else:
        page_count = 0
        doc_count = 0

    # Load persisted file metadata (survives session/container restarts)
    file_metadata = store.load_file_metadata(PERSISTENT_INDEX) if has_existing else []

    # Store in session
    cl.user_session.set("index_name", PERSISTENT_INDEX)
    cl.user_session.set("files", file_metadata)
    cl.user_session.set("memory", ConversationMemory())
    cl.user_session.set("cancelled", False)
    cl.user_session.set("has_documents", has_existing)

    print(f"Session started (index: {PERSISTENT_INDEX}, existing: {has_existing})")

    # Non-blocking welcome message
    existing_info = ""
    if has_existing:
        existing_info = (
            f"\n\n**Existing index loaded:** {page_count} pages from {doc_count} document(s).\n"
            f"You can ask questions or upload more files."
        )

    await cl.Message(
        content=f"**Visual Document Assistant**\n\n"
                f"**Model:** {config.models.model_name}\n"
                f"**Retrieval:** Byaldi (ColQwen2)\n\n"
                f"**Features:**\n"
                f"- PDF, Images (PNG, JPG)\n"
                f"- Visual page search\n"
                f"- Page-aware retrieval\n\n"
                f"Upload files or ask a question to start!"
                f"{existing_info}"
    ).send()


@cl.on_stop
async def on_stop():
    """Handle stop button."""
    cl.user_session.set("cancelled", True)
    print("Stop requested")


@cl.on_message
async def on_message(message: cl.Message):
    """Handle messages and commands."""

    query = message.content.strip()
    index_name = cl.user_session.get("index_name")
    file_list = cl.user_session.get("files")
    memory = cl.user_session.get("memory")
    has_documents = cl.user_session.get("has_documents", False)

    # === COMMANDS ===

    if query.lower() == '/clear':
        # Delete persistent index and reset
        store = get_visual_store()
        store.delete_index(PERSISTENT_INDEX)

        cache = get_cache()
        cache.clear_index_cache(PERSISTENT_INDEX)

        cl.user_session.set("files", [])
        cl.user_session.set("has_documents", False)
        memory.clear()

        await cl.Message(content="Cleared all documents. Ready for new files.").send()
        return

    if query.lower() == '/files':
        if file_list:
            names = "\n".join([f"- {f['name']} ({f.get('pages', '?')} pages)" for f in file_list])
            await cl.Message(content=f"**Files:**\n{names}").send()
        else:
            await cl.Message(content="No files uploaded.").send()
        return

    if query.lower() == '/stats':
        store = get_visual_store()
        stats = store.get_stats(index_name)
        cache = get_cache()
        cache_stats = cache.get_stats()

        await cl.Message(
            content=f"**Stats:**\n"
                    f"- Pages indexed: {stats['total_pages']}\n"
                    f"- Documents: {stats['document_count']}\n"
                    f"- Cache: {cache_stats.get('used_memory', 'N/A')}\n"
                    f"- Files: {len(file_list)}"
        ).send()
        return

    if query.lower() == '/debug':
        store = get_visual_store()
        stats = store.get_stats(index_name)
        docs = stats.get("documents", [])

        result = f"**Debug:**\n- Index: {index_name}\n- Pages: {stats['total_pages']}\n"
        if docs:
            result += "- Documents:\n"
            for doc in docs:
                result += f"  - {doc}\n"
        await cl.Message(content=result).send()
        return

    if query.lower() == '/health':
        from openai import AsyncOpenAI
        client = AsyncOpenAI(base_url=config.models.base_url, api_key="EMPTY")
        try:
            models = await client.models.list()
            model_names = [m.id for m in models.data]
            await cl.Message(
                content=f"**Health Check:**\n- vLLM: Online\n- Models: {', '.join(model_names)}"
            ).send()
        except Exception as e:
            await cl.Message(content=f"**Health Check:**\n- vLLM: Offline ({e})").send()
        return

    # === FILE UPLOAD ===
    if message.elements:
        if not index_name:
            await cl.Message(content="Session not initialized. Refresh page.").send()
            return

        is_first = not has_documents
        pages = await process_files(message.elements, index_name, file_list, is_first_upload=is_first)
        if pages and pages > 0:
            cl.user_session.set("has_documents", True)
        return

    # === CHECK FOR DOCUMENTS ===
    if not has_documents:
        await cl.Message(content="No documents loaded. Please upload files first.").send()
        return

    # === GENERATE RESPONSE ===
    response_msg = cl.Message(content="")
    await response_msg.send()

    await generate_response(query, index_name, memory, response_msg)


@cl.on_chat_end
async def on_end():
    """Session ended - index is preserved for future sessions."""
    print(f"Session ended (index '{PERSISTENT_INDEX}' preserved on disk)")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("VISUAL DOCUMENT ASSISTANT")
    print("=" * 60)
    print(f"Model: {config.models.model_name}")
    print(f"Retrieval: Byaldi ({config.byaldi.model_name})")
    print(f"Search Top-K: {config.visual_rag.search_top_k}")
    print(f"Grounding: {config.visual_rag.enable_grounding}")
    print("=" * 60 + "\n")
