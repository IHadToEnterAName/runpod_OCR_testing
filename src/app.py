"""
Main Application
================
Chainlit interface with persistent storage and optional file upload.
"""

import chainlit as cl

from config.settings import get_config
from rag.memory import ConversationMemory
from rag.embeddings import embed_query
from rag.pipeline import generate_response
from storage.vector_store import create_collection, delete_collection, chroma_client, retrieve_chunks
from processing.file_processor import process_files
from rag.traffic_controller import get_traffic_controller, check_servers_health

# =============================================================================
# CONFIGURATION
# =============================================================================

config = get_config()

# Fixed collection name for persistent storage
COLLECTION_NAME = "documents"

# =============================================================================
# CHAINLIT HANDLERS
# =============================================================================

@cl.on_chat_start
async def start():
    """
    Initialize session with persistent ChromaDB collection.
    Does NOT force file upload - user can chat immediately.
    """

    # Get or create the persistent collection
    collection = create_collection(COLLECTION_NAME)
    doc_count = collection.count()

    # Store in session
    cl.user_session.set("collection", collection)
    cl.user_session.set("collection_name", COLLECTION_NAME)
    cl.user_session.set("files", [])
    cl.user_session.set("memory", ConversationMemory())
    cl.user_session.set("cancelled", False)

    print(f"Collection '{COLLECTION_NAME}': {doc_count} chunks")

    # Build welcome message based on whether documents exist
    if doc_count > 0:
        # Get file stats from existing collection
        try:
            all_meta = collection.get(include=["metadatas"])
            sources = set(
                m.get("source", "unknown")
                for m in all_meta["metadatas"]
                if m.get("source")
            )
            pages = [
                m.get("page_number")
                for m in all_meta["metadatas"]
                if m.get("page_number")
            ]
            page_range = f"{min(pages)}-{max(pages)}" if pages else "N/A"

            file_names = "\n".join(f"  - {s}" for s in sorted(sources))

            welcome = (
                f"**Document Assistant**\n\n"
                f"Loaded **{doc_count}** chunks from previous session:\n"
                f"{file_names}\n"
                f"Pages: {page_range}\n\n"
                f"You can ask questions about your documents or upload more files.\n"
                f"Use `/clear` to wipe all documents and start fresh."
            )
        except Exception:
            welcome = (
                f"**Document Assistant**\n\n"
                f"Loaded **{doc_count}** chunks from previous session.\n\n"
                f"You can ask questions or upload more files.\n"
                f"Use `/clear` to wipe all documents and start fresh."
            )
    else:
        welcome = (
            f"**Document Assistant**\n\n"
            f"**Models:**\n"
            f"  Vision: {config.models.vision_model}\n"
            f"  Reasoning: {config.models.reasoning_model}\n\n"
            f"**Supported:** PDF, DOCX, TXT, Images\n\n"
            f"Upload files to get started, or just ask a general question."
        )

    await cl.Message(content=welcome).send()


@cl.on_stop
async def on_stop():
    """Handle stop button."""
    cl.user_session.set("cancelled", True)
    print("Stop requested")


@cl.on_message
async def on_message(message: cl.Message):
    """
    Handle messages.
    Supports chatting with or without documents loaded.
    """

    query = message.content.strip()
    collection = cl.user_session.get("collection")
    file_list = cl.user_session.get("files")
    memory = cl.user_session.get("memory")

    # COMMANDS

    if query.lower() == '/clear':
        coll_name = cl.user_session.get("collection_name")
        delete_collection(coll_name)

        # Recreate empty collection
        new_coll = create_collection(COLLECTION_NAME)

        cl.user_session.set("collection", new_coll)
        cl.user_session.set("collection_name", COLLECTION_NAME)
        cl.user_session.set("files", [])
        memory.clear()

        await cl.Message(content="Cleared all documents. Ready for new files.").send()
        return

    if query.lower() == '/files':
        if collection:
            try:
                all_meta = collection.get(include=["metadatas"])
                sources = set(
                    m.get("source", "unknown")
                    for m in all_meta["metadatas"]
                    if m.get("source")
                )
                if sources:
                    names = "\n".join(f"  - {s}" for s in sorted(sources))
                    await cl.Message(content=f"**Files:**\n{names}").send()
                else:
                    await cl.Message(content="No files.").send()
            except Exception:
                await cl.Message(content="No files.").send()
        else:
            await cl.Message(content="No files.").send()
        return

    if query.lower() == '/stats':
        count = collection.count() if collection else 0
        # Count sources from metadata
        source_count = 0
        if collection and count > 0:
            try:
                all_meta = collection.get(include=["metadatas"])
                source_count = len(set(
                    m.get("source", "")
                    for m in all_meta["metadatas"]
                    if m.get("source")
                ))
            except Exception:
                pass
        await cl.Message(
            content=f"**Stats:**\n  Chunks: {count}\n  Files: {source_count}"
        ).send()
        return

    if query.lower() == '/test':
        if collection and collection.count() > 0:
            query_emb = embed_query("summary overview")
            test_chunks = retrieve_chunks(collection, "summary overview", query_emb, top_k=3)

            if test_chunks:
                result = "**Retrieval Test Passed!**\n\n"
                for i, c in enumerate(test_chunks):
                    result += f"**[{i+1}] Page {c.get('page', '?')}:**\n{c['text'][:150]}...\n\n"
                await cl.Message(content=result).send()
            else:
                await cl.Message(content="Retrieval returned empty").send()
        else:
            await cl.Message(content="No documents to test").send()
        return

    if query.lower() == '/debug':
        if collection:
            count = collection.count()
            if count > 0:
                sample = collection.get(limit=3, include=["documents", "metadatas"])
                result = f"**Debug:**\n  Count: {count}\n\n**Samples:**\n"
                for doc, meta in zip(sample["documents"], sample["metadatas"]):
                    result += f"  Page {meta.get('page_number', '?')}: {doc[:100]}...\n"
                await cl.Message(content=result).send()
            else:
                await cl.Message(content="Collection empty").send()
        else:
            await cl.Message(content="No collection").send()
        return

    if query.lower() == '/traffic':
        controller = get_traffic_controller()
        stats = controller.get_stats()
        health_summary = controller.get_health_summary()

        result = (
            f"**Traffic Controller Stats:**\n\n"
            f"**Requests:**\n"
            f"  Total: {stats['total_requests']}\n"
            f"  Vision: {stats['vision_requests']}\n"
            f"  Reasoning: {stats['reasoning_requests']}\n\n"
            f"**Performance:**\n"
            f"  Avg Response Time: {stats['avg_response_time_ms']:.0f}ms\n"
            f"  Rate Limited: {stats['rate_limited']}\n"
            f"  Errors: {stats['errors']}\n\n"
            f"{health_summary}"
        )
        await cl.Message(content=result).send()
        return

    if query.lower() == '/health':
        await cl.Message(content="Checking server health...").send()
        stats = await check_servers_health()
        controller = get_traffic_controller()

        result = (
            f"**Health Check Complete:**\n\n"
            f"  Vision Model: {stats['health']['vision']}\n"
            f"  Reasoning Model: {stats['health']['reasoning']}\n\n"
            f"{controller.get_health_summary()}"
        )
        await cl.Message(content=result).send()
        return

    # FILE UPLOAD via message attachments
    if message.elements:
        if not collection:
            await cl.Message(
                content="Session not initialized. Refresh page."
            ).send()
            return
        await process_files(message.elements, collection, file_list)
        return

    # GENERATE RESPONSE
    if not collection or collection.count() == 0:
        # No documents loaded - let user know but still respond via LLM
        await cl.Message(
            content="No documents loaded yet. Upload files to ask document-specific questions.\n"
                    "You can drag & drop files into the chat, or attach them with the paperclip icon."
        ).send()
        return

    response_msg = cl.Message(content="")
    await response_msg.send()

    # Generate query embedding
    query_embedding = embed_query(query)

    # Generate response
    await generate_response(query, collection, query_embedding, memory, response_msg)


@cl.on_chat_end
async def on_end():
    """
    Session ended - do NOT delete the collection.
    Documents persist for future sessions.
    """
    print(f"Session ended. Collection '{COLLECTION_NAME}' preserved.")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("DOCUMENT ASSISTANT - RAG SYSTEM")
    print("="*60)
    print("Persistent ChromaDB storage")
    print("Optional file upload (not required to chat)")
    print("Page-aware retrieval")
    print("="*60 + "\n")
