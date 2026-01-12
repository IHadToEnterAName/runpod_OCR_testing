"""
Main Application
================
Chainlit interface with all original logic preserved EXACTLY.
"""

import chainlit as cl
import uuid

from config.settings import get_config
from rag.memory import ConversationMemory
from rag.embeddings import embed_query
from rag.pipeline import generate_response
from storage.vector_store import create_collection, delete_collection, chroma_client, retrieve_chunks
from processing.file_processor import process_files

# =============================================================================
# CONFIGURATION
# =============================================================================

config = get_config()

# =============================================================================
# CHAINLIT HANDLERS (Original Logic Preserved)
# =============================================================================

@cl.on_chat_start
async def start():
    """
    Initialize session with ChromaDB collection.
    Original logic from your code preserved EXACTLY.
    """
    
    # Create unique collection for this session (original logic)
    session_id = str(uuid.uuid4())[:8]
    collection_name = f"docs_{session_id}"
    
    # Create collection
    collection = create_collection(collection_name)
    
    # Store in session (original logic)
    cl.user_session.set("collection", collection)
    cl.user_session.set("collection_name", collection_name)
    cl.user_session.set("files", [])
    cl.user_session.set("memory", ConversationMemory())
    cl.user_session.set("cancelled", False)
    
    print(f"📚 Created collection: {collection_name}")
    
    # Ask for files (original logic)
    files = await cl.AskFileMessage(
        content=f"🚀 **Document Assistant**\n\n"
                f"**Models:**\n"
                f"• Vision: {config.models.vision_model}\n"
                f"• Reasoning: {config.models.reasoning_model}\n\n"
                f"**Features:**\n"
                f"• ✅ PDF, DOCX, TXT, Images\n"
                f"• ✅ Page-aware retrieval\n"
                f"• ✅ Vision analysis\n"
                f"• ✅ Context-aware responses\n\n"
                f"Upload files to start!",
        accept=[
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain",
            "image/png",
            "image/jpeg"
        ],
        max_size_mb=50,
        timeout=300
    ).send()
    
    if files:
        file_list = cl.user_session.get("files")
        await process_files(files, collection, file_list)


@cl.on_stop
async def on_stop():
    """
    Handle stop button.
    Original logic preserved.
    """
    cl.user_session.set("cancelled", True)
    print("🛑 Stop requested")


@cl.on_message
async def on_message(message: cl.Message):
    """
    Handle messages.
    Original logic from your code preserved EXACTLY.
    """
    
    query = message.content.strip()
    collection = cl.user_session.get("collection")
    file_list = cl.user_session.get("files")
    memory = cl.user_session.get("memory")
    
    # COMMANDS (original logic)
    
    if query.lower() == '/clear':
        coll_name = cl.user_session.get("collection_name")
        delete_collection(coll_name)
        
        # Create new collection
        new_name = f"docs_{str(uuid.uuid4())[:8]}"
        new_coll = create_collection(new_name)
        
        cl.user_session.set("collection", new_coll)
        cl.user_session.set("collection_name", new_name)
        cl.user_session.set("files", [])
        memory.clear()
        
        await cl.Message(content="🗑️ Cleared. Ready for new files.").send()
        return
    
    if query.lower() == '/files':
        if file_list:
            names = "\n".join([f"• {f['name']}" for f in file_list])
            await cl.Message(content=f"📁 **Files:**\n{names}").send()
        else:
            await cl.Message(content="📭 No files.").send()
        return
    
    if query.lower() == '/stats':
        count = collection.count() if collection else 0
        await cl.Message(
            content=f"📊 **Stats:**\n• Chunks: {count}\n• Files: {len(file_list)}"
        ).send()
        return
    
    if query.lower() == '/test':
        # Test retrieval (original logic)
        if collection and collection.count() > 0:
            query_emb = embed_query("summary overview")
            test_chunks = retrieve_chunks(collection, "summary overview", query_emb, top_k=3)
            
            if test_chunks:
                result = "✅ **Retrieval Test Passed!**\n\n"
                for i, c in enumerate(test_chunks):
                    result += f"**[{i+1}] Page {c.get('page', '?')}:**\n{c['text'][:150]}...\n\n"
                await cl.Message(content=result).send()
            else:
                await cl.Message(content="❌ Retrieval returned empty").send()
        else:
            await cl.Message(content="📭 No documents to test").send()
        return
    
    if query.lower() == '/debug':
        if collection:
            count = collection.count()
            if count > 0:
                sample = collection.get(limit=3, include=["documents", "metadatas"])
                result = f"📊 **Debug:**\n• Count: {count}\n\n**Samples:**\n"
                for doc, meta in zip(sample["documents"], sample["metadatas"]):
                    result += f"• Page {meta.get('page_number', '?')}: {doc[:100]}...\n"
                await cl.Message(content=result).send()
            else:
                await cl.Message(content="📭 Collection empty").send()
        else:
            await cl.Message(content="❌ No collection").send()
        return
    
    # FILE UPLOAD (original logic)
    if message.elements:
        if not collection:
            await cl.Message(
                content="❌ Session not initialized. Refresh page."
            ).send()
            return
        await process_files(message.elements, collection, file_list)
        return
    
    # CHECK COLLECTION (original logic)
    if not collection or collection.count() == 0:
        await cl.Message(
            content="📭 No documents loaded. Please upload files first."
        ).send()
        return
    
    # GENERATE RESPONSE (original logic)
    response_msg = cl.Message(content="")
    await response_msg.send()
    
    # Generate query embedding
    query_embedding = embed_query(query)
    
    # Generate response
    await generate_response(query, collection, query_embedding, memory, response_msg)


@cl.on_chat_end
async def on_end():
    """
    Cleanup on session end.
    Original logic preserved.
    """
    coll_name = cl.user_session.get("collection_name")
    if coll_name:
        delete_collection(coll_name)


# =============================================================================
# MAIN (Original Logic)
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 DOCUMENT ASSISTANT - RAG SYSTEM")
    print("="*60)
    print("✅ Session-based ChromaDB collections")
    print("✅ Proper query vs document embeddings")
    print("✅ Verified context injection")
    print("✅ Page-aware retrieval")
    print("="*60 + "\n")
