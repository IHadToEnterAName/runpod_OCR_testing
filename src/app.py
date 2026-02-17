"""
Main Application
================
Chainlit interface for Visual RAG Document Assistant.
Byaldi (ColQwen2) + Qwen3-VL-32B-AWQ architecture.
"""

import os
import shutil

import chainlit as cl

from config.settings import get_config, SHARED_INDEX
from rag.memory import ConversationMemory
from rag.pipeline import generate_response
from storage.visual_store import get_visual_store
from processing.file_processor import process_files, CHAINLIT_UPLOAD_STORE
from rag.cache import get_cache

# =============================================================================
# CONFIGURATION
# =============================================================================

config = get_config()

# Index name imported from config (shared with API)

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
    has_existing = store.index_exists_on_disk(SHARED_INDEX)
    if has_existing:
        store.load_existing_index(SHARED_INDEX)
        stats = store.get_stats(SHARED_INDEX)
        page_count = stats["total_pages"]
        doc_count = stats["document_count"]
    else:
        page_count = 0
        doc_count = 0

    # Load persisted file metadata (survives session/container restarts)
    file_metadata = store.load_file_metadata(SHARED_INDEX) if has_existing else []

    # Store in session
    cl.user_session.set("index_name", SHARED_INDEX)
    cl.user_session.set("files", file_metadata)
    cl.user_session.set("memory", ConversationMemory())
    cl.user_session.set("cancelled", False)
    cl.user_session.set("has_documents", has_existing)

    print(f"Session started (index: {SHARED_INDEX}, existing: {has_existing})")

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
        store.delete_index(SHARED_INDEX)

        cache = get_cache()
        cache.clear_index_cache(SHARED_INDEX)

        # Clear the API document registry
        from api.document_registry import get_document_registry
        registry = get_document_registry()
        for rec in list(registry.list_all()):
            # Delete stored upload files
            if rec.file_path and os.path.exists(rec.file_path):
                file_dir = os.path.dirname(rec.file_path)
                if os.path.exists(file_dir):
                    shutil.rmtree(file_dir, ignore_errors=True)
            registry.remove(rec.document_id)

        # Delete all persistent upload copies
        for upload_dir in [CHAINLIT_UPLOAD_STORE,
                           os.path.join(config.byaldi.index_path, "api_uploads")]:
            if os.path.exists(upload_dir):
                shutil.rmtree(upload_dir, ignore_errors=True)
                os.makedirs(upload_dir, exist_ok=True)

        cl.user_session.set("files", [])
        cl.user_session.set("has_documents", False)
        memory.clear()

        await cl.Message(content="Cleared all documents and uploads. Ready for new files.").send()
        return

    if query.lower().startswith('/delete'):
        doc_name = query[7:].strip()
        if not doc_name:
            await cl.Message(
                content="**Usage:** `/delete <document_name>`\n"
                        "**Example:** `/delete report.pdf`\n\n"
                        "Use `/files` to see uploaded document names."
            ).send()
            return

        # Find the document (case-insensitive, supports partial match)
        matching = [f for f in file_list if f["name"].lower() == doc_name.lower()]
        if not matching:
            matching = [f for f in file_list if doc_name.lower() in f["name"].lower()]

        if not matching:
            names = "\n".join([f"- {f['name']}" for f in file_list]) if file_list else "No documents uploaded."
            await cl.Message(
                content=f"Document '{doc_name}' not found.\n\n**Available documents:**\n{names}"
            ).send()
            return

        target = matching[0]
        target_name = target["name"]

        progress_msg = cl.Message(content=f"Deleting '{target_name}'...")
        await progress_msg.send()

        # Remove target from file_list, keep the rest
        remaining_files = [f for f in file_list if f["name"] != target_name]

        # Delete the current index (Byaldi doesn't support removing individual docs)
        store = get_visual_store()
        store.delete_index(SHARED_INDEX)

        cache = get_cache()
        cache.clear_index_cache(SHARED_INDEX)

        # Clear ALL registry entries (byaldi_doc_ids become stale after rebuild)
        from api.document_registry import get_document_registry, DocumentRecord
        registry = get_document_registry()
        for rec in list(registry.list_all()):
            # Delete API-uploaded file copies for the target document
            if rec.document_name == target_name and rec.file_path and os.path.exists(rec.file_path):
                file_dir = os.path.dirname(rec.file_path)
                if os.path.exists(file_dir):
                    shutil.rmtree(file_dir, ignore_errors=True)
            registry.remove(rec.document_id)

        # Delete the persistent Chainlit upload copy of the target
        chainlit_copy = os.path.join(CHAINLIT_UPLOAD_STORE, target_name)
        if os.path.exists(chainlit_copy):
            os.unlink(chainlit_copy)

        # Re-index remaining documents
        if remaining_files:
            rebuild_ok = []
            for idx, f_meta in enumerate(remaining_files):
                file_path = f_meta.get("path", "")

                # Try alternative locations if path is missing or stale
                if not file_path or not os.path.exists(file_path):
                    alt = os.path.join(CHAINLIT_UPLOAD_STORE, f_meta["name"])
                    if os.path.exists(alt):
                        file_path = alt

                if not file_path or not os.path.exists(file_path):
                    print(f"Cannot re-index '{f_meta['name']}': file not found")
                    continue

                try:
                    progress_msg.content = f"Re-indexing ({idx + 1}/{len(remaining_files)}): {f_meta['name']}..."
                    await progress_msg.update()

                    if not rebuild_ok:
                        store.create_index(SHARED_INDEX, file_path, f_meta["name"])
                    else:
                        store.add_to_index(SHARED_INDEX, file_path, f_meta["name"])
                    rebuild_ok.append(f_meta)
                except Exception as e:
                    print(f"Failed to re-index {f_meta['name']}: {e}")

            remaining_files = rebuild_ok
            store.save_file_metadata(SHARED_INDEX, remaining_files)

            # Rebuild registry from scratch with correct byaldi_doc_ids
            for i, f_meta in enumerate(remaining_files):
                record = DocumentRecord(
                    document_id=f"chainlit-{f_meta['name']}",
                    document_name=f_meta["name"],
                    file_path=f_meta.get("path", ""),
                    chunk_count=f_meta.get("pages", 0),
                    byaldi_doc_id=i,
                    file_type=f_meta.get("type", "unknown"),
                )
                registry.add(record)

            cl.user_session.set("files", remaining_files)
            cl.user_session.set("has_documents", True)

            names = ", ".join([f["name"] for f in remaining_files])
            await cl.Message(
                content=f"Deleted '{target_name}'.\n\n**Remaining documents:** {names}"
            ).send()
        else:
            cl.user_session.set("files", [])
            cl.user_session.set("has_documents", False)
            memory.clear()
            await cl.Message(content=f"Deleted '{target_name}'. No documents remaining.").send()

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
            # Sync to DocumentRegistry so API can see Chainlit-uploaded docs
            from api.document_registry import get_document_registry, DocumentRecord
            registry = get_document_registry()
            for f_meta in file_list:
                if not any(r.document_name == f_meta["name"] for r in registry.list_all()):
                    record = DocumentRecord(
                        document_id=f"chainlit-{f_meta['name']}",
                        document_name=f_meta["name"],
                        file_path=f_meta.get("path", ""),
                        chunk_count=f_meta.get("pages", 0),
                        byaldi_doc_id=registry.next_byaldi_doc_id(),
                        file_type=f_meta.get("type", "unknown"),
                    )
                    registry.add(record)
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
    print(f"Session ended (index '{SHARED_INDEX}' preserved on disk)")


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
