"""
File Processing Orchestrator
=============================
Coordinates document processing workflow.
Original logic from your code preserved exactly.
"""

import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import List
import chainlit as cl
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import get_config
from processing.document_extractor import (
    extract_pdf_pages,
    extract_pdf_images,
    extract_docx,
    extract_txt
)
from processing.vision import analyze_image
from rag.embeddings import embed_documents
from storage.vector_store import add_chunks_to_collection

# =============================================================================
# CONFIGURATION
# =============================================================================

config = get_config()

# Text splitter (original logic)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.chunking.chunk_size,
    chunk_overlap=config.chunking.chunk_overlap,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# Thread pool (original logic)
executor = ThreadPoolExecutor(
    max_workers=config.performance.file_processing_workers
)

# =============================================================================
# FILE PROCESSING (Original Logic Preserved)
# =============================================================================

async def process_files(files, collection, file_list: List[dict]):
    """
    Process uploaded files into ChromaDB.
    Original logic from your code preserved EXACTLY.
    """
    
    progress = cl.Message(content="📂 Starting processing...")
    await progress.send()
    
    total_chunks = 0
    
    for i, file in enumerate(files):
        fname = file.name
        fpath = file.path
        
        progress.content = f"📂 Processing {i+1}/{len(files)}: {fname}"
        await progress.update()
        
        file_list.append({"name": fname})
        
        # PDF PROCESSING (original logic)
        if fname.lower().endswith('.pdf'):
            # Extract text
            loop = asyncio.get_event_loop()
            pages = await loop.run_in_executor(
                executor, 
                extract_pdf_pages, 
                fpath
            )
            
            progress.content = f"📄 Chunking {len(pages)} pages..."
            await progress.update()
            
            # Chunk each page
            all_docs = []
            all_metas = []
            all_ids = []
            
            for page_num, page_text in pages:
                chunks = text_splitter.split_text(page_text)
                for idx, chunk in enumerate(chunks):
                    all_docs.append(chunk)
                    all_metas.append({
                        "page_number": page_num,
                        "source": fname,
                        "chunk_type": "text"
                    })
                    all_ids.append(
                        f"{fname}_p{page_num}_c{idx}_{uuid.uuid4().hex[:6]}"
                    )
            
            print(f"📊 Created {len(all_docs)} chunks from {fname}")
            
            # Embed and add in batches (original logic)
            if all_docs:
                progress.content = f"⚡ Embedding {len(all_docs)} chunks..."
                await progress.update()
                
                # Generate embeddings
                all_embeddings = embed_documents(all_docs)
                
                # Add to collection
                add_chunks_to_collection(
                    collection,
                    all_docs,
                    all_embeddings,
                    all_metas,
                    all_ids
                )
                
                total_chunks += len(all_docs)
            
            # Process images (original logic)
            progress.content = f"🖼️ Extracting images..."
            await progress.update()
            
            images = await loop.run_in_executor(
                executor,
                extract_pdf_images,
                fpath,
                25
            )
            
            if images:
                for j, (page_num, img) in enumerate(images):
                    progress.content = f"🖼️ Analyzing image {j+1}/{len(images)}..."
                    await progress.update()
                    
                    desc = await analyze_image(img, page_num, j+1)
                    
                    # Embed and add
                    img_embedding = embed_documents([desc])[0]
                    
                    collection.add(
                        documents=[desc],
                        embeddings=[img_embedding],
                        metadatas=[{
                            "page_number": page_num,
                            "source": fname,
                            "chunk_type": "image"
                        }],
                        ids=[f"{fname}_img_p{page_num}_{j}_{uuid.uuid4().hex[:6]}"]
                    )
                    
                    total_chunks += 1
                    await asyncio.sleep(0.2)
        
        # DOCX PROCESSING (original logic)
        elif fname.lower().endswith('.docx'):
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(executor, extract_docx, fpath)
            
            if text:
                chunks = text_splitter.split_text(text)
                embeddings = embed_documents(chunks)
                
                docs = []
                metas = []
                ids = []
                
                for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                    docs.append(chunk)
                    metas.append({
                        "source": fname,
                        "chunk_type": "text"
                    })
                    ids.append(f"{fname}_c{idx}_{uuid.uuid4().hex[:6]}")
                
                add_chunks_to_collection(
                    collection,
                    docs,
                    embeddings,
                    metas,
                    ids
                )
                
                total_chunks += len(chunks)
        
        # TXT PROCESSING (original logic)
        elif fname.lower().endswith('.txt'):
            text = extract_txt(fpath)
            
            if text:
                chunks = text_splitter.split_text(text)
                embeddings = embed_documents(chunks)
                
                docs = []
                metas = []
                ids = []
                
                for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                    docs.append(chunk)
                    metas.append({
                        "source": fname,
                        "chunk_type": "text"
                    })
                    ids.append(f"{fname}_c{idx}_{uuid.uuid4().hex[:6]}")
                
                add_chunks_to_collection(
                    collection,
                    docs,
                    embeddings,
                    metas,
                    ids
                )
                
                total_chunks += len(chunks)
        
        # IMAGE PROCESSING (original logic)
        elif fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            from PIL import Image
            img = Image.open(fpath)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            desc = await analyze_image(img, 0, 1)
            emb = embed_documents([desc])[0]
            
            collection.add(
                documents=[desc],
                embeddings=[emb],
                metadatas=[{
                    "source": fname,
                    "chunk_type": "image"
                }],
                ids=[f"{fname}_img_{uuid.uuid4().hex[:6]}"]
            )
            
            total_chunks += 1
    
    # Final summary (original logic)
    final_count = collection.count()
    
    # Get statistics
    try:
        all_meta = collection.get(include=["metadatas"])
        pages = [
            m.get("page_number") 
            for m in all_meta["metadatas"] 
            if m.get("page_number")
        ]
        page_range = f"{min(pages)}-{max(pages)}" if pages else "N/A"
        
        text_count = sum(
            1 for m in all_meta["metadatas"] 
            if m.get("chunk_type") == "text"
        )
        img_count = sum(
            1 for m in all_meta["metadatas"] 
            if m.get("chunk_type") == "image"
        )
    except:
        page_range = "N/A"
        text_count = final_count
        img_count = 0
    
    await cl.Message(
        content=f"✅ **Processing Complete!**\n\n"
                f"📊 **ChromaDB:**\n"
                f"• Total chunks: **{final_count}**\n"
                f"• Text: **{text_count}** | Images: **{img_count}**\n"
                f"• Pages: **{page_range}**\n"
                f"• Files: **{len(file_list)}**\n\n"
                f"💡 Try: 'What is on page 50?' or 'Summarize the document'\n\n"
                f"Commands: `/clear`, `/files`, `/stats`, `/test`"
    ).send()
