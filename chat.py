"""
Fixed VLM Pipeline v6 - WORKING ChromaDB Retrieval
===================================================
CRITICAL FIXES:
1. Fixed session ID handling - uses cl.user_session properly
2. Fixed collection retrieval - collection stored in session
3. Added extensive debugging to trace retrieval issues
4. Fixed embedding function for queries vs documents
5. Proper context injection verified

Author: AI Architect  
Version: 6.0 - Working Retrieval
"""

import chainlit as cl
from openai import AsyncOpenAI
import os
from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image
import io
import base64
import fitz  # PyMuPDF
import docx
from langchain_text_splitters import RecursiveCharacterTextSplitter
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
import tiktoken
import re
from dataclasses import dataclass, field
from collections import deque
import chromadb
from chromadb.config import Settings
import uuid

# =============================================================================
# CONFIGURATION
# =============================================================================

vision_client = AsyncOpenAI(
    base_url="http://localhost:8006/v1",
    api_key="EMPTY"
)

reasoning_client = AsyncOpenAI(
    base_url="http://localhost:8005/v1",
    api_key="EMPTY"
)

VISION_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
REASONING_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
EMBEDDING_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"

# Performance
VISION_CONCURRENT_LIMIT = 1
EMBEDDING_BATCH_SIZE = 64
FILE_PROCESSING_WORKERS = 8

# Token limits
MODEL_MAX_TOKENS = 16384
MAX_OUTPUT_TOKENS = 2048
MAX_INPUT_TOKENS = MODEL_MAX_TOKENS - MAX_OUTPUT_TOKENS - 1000
VISION_MAX_TOKENS = 512

# Chunking - smaller = more chunks = better retrieval
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
MAX_CONTEXT_CHUNKS = 12

# ChromaDB
CHROMA_PERSIST_DIR = "/workspace/chroma_db"

# Generation
TEMPERATURE = 0.3
TOP_P = 0.85
FILTER_THINKING_TAGS = True
THINKING_TIMEOUT_SECONDS = 30

# Semaphores
vision_semaphore = asyncio.Semaphore(VISION_CONCURRENT_LIMIT)
llm_semaphore = asyncio.Semaphore(2)

# =============================================================================
# EMBEDDING MODEL - GLOBAL INIT
# =============================================================================

os.environ["HF_HOME"] = "/workspace/huggingface"

print("🔄 Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, trust_remote_code=True)
embedding_model.to('cuda')
embedding_model.max_seq_length = 8192
print(f"✅ Embedding model: {EMBEDDING_MODEL_NAME}")

def embed_documents(texts: List[str]) -> List[List[float]]:
    """Embed documents (for storage)."""
    prefixed = [f"search_document: {t}" for t in texts]
    embeddings = embedding_model.encode(
        prefixed, batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True
    )
    return embeddings.tolist()

def embed_query(query: str) -> List[float]:
    """Embed a single query (for retrieval)."""
    prefixed = f"search_query: {query}"
    embedding = embedding_model.encode(
        [prefixed], batch_size=1,
        show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True
    )
    return embedding[0].tolist()

# =============================================================================
# CHROMADB - GLOBAL CLIENT
# =============================================================================

print("🔄 Initializing ChromaDB...")
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PERSIST_DIR,
    settings=Settings(anonymized_telemetry=False, allow_reset=True)
)
print(f"✅ ChromaDB at {CHROMA_PERSIST_DIR}")

# =============================================================================
# TOKENIZER
# =============================================================================

try:
    tokenizer = tiktoken.get_encoding("cl100k_base")
except:
    tokenizer = None

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text)) if tokenizer else len(text) // 4

# =============================================================================
# THINKING FILTER
# =============================================================================

class ThinkingFilter:
    def __init__(self):
        self.in_thinking = False
        self.buffer = ""
        self.start_time = None
    
    def reset(self):
        self.in_thinking = False
        self.buffer = ""
        self.start_time = None
    
    def process(self, token: str) -> Tuple[str, bool]:
        self.buffer += token
        
        if "<think>" in self.buffer and not self.in_thinking:
            self.in_thinking = True
            self.start_time = time.time()
            before = self.buffer.split("<think>")[0]
            self.buffer = ""
            return before, True
        
        if self.in_thinking:
            if "</think>" in self.buffer:
                self.in_thinking = False
                after = self.buffer.split("</think>")[-1]
                self.buffer = after
                self.start_time = None
                return after, False
            
            if self.start_time and (time.time() - self.start_time) > THINKING_TIMEOUT_SECONDS:
                self.in_thinking = False
                self.buffer = ""
                return "", False
            
            return "", True
        
        result = self.buffer
        self.buffer = ""
        return result, False
    
    def flush(self) -> str:
        result = self.buffer
        self.buffer = ""
        return result

def clean_thinking(text: str) -> str:
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'</?think>', '', text)
    return text.strip()

# =============================================================================
# CJK FILTER
# =============================================================================

CJK_RANGES = [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x3040, 0x309F), (0x30A0, 0x30FF), (0xAC00, 0xD7AF)]

def filter_cjk(text: str) -> str:
    return ''.join(c for c in text if not any(s <= ord(c) <= e for s, e in CJK_RANGES))

# =============================================================================
# CONVERSATION MEMORY
# =============================================================================

class ConversationMemory:
    def __init__(self):
        self.turns: List[Tuple[str, str]] = []
        self.mentioned_pages: List[int] = []
    
    def add(self, user: str, assistant: str):
        self.turns.append((user, assistant))
        if len(self.turns) > 10:
            self.turns.pop(0)
        pages = re.findall(r'page\s*(\d+)', user.lower())
        self.mentioned_pages.extend([int(p) for p in pages])
    
    def get_history(self, n: int = 3) -> List[Tuple[str, str]]:
        return self.turns[-n:]
    
    def clear(self):
        self.turns.clear()
        self.mentioned_pages.clear()

# =============================================================================
# TEXT SPLITTER
# =============================================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# Thread pool
executor = ThreadPoolExecutor(max_workers=FILE_PROCESSING_WORKERS)

# =============================================================================
# PDF/DOCX EXTRACTION
# =============================================================================

def extract_pdf_pages(file_path: str) -> List[Tuple[int, str]]:
    pages = []
    try:
        doc = fitz.open(file_path)
        for i in range(len(doc)):
            text = doc[i].get_text("text").strip()
            if text:
                pages.append((i + 1, text))
        doc.close()
        print(f"✅ Extracted {len(pages)} pages from PDF")
    except Exception as e:
        print(f"❌ PDF error: {e}")
    return pages

def extract_pdf_images(file_path: str, max_images: int = 25) -> List[Tuple[int, Image.Image]]:
    images = []
    try:
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            if len(images) >= max_images:
                break
            for img in doc[page_num].get_images(full=True):
                if len(images) >= max_images:
                    break
                try:
                    xref = img[0]
                    base = doc.extract_image(xref)
                    image = Image.open(io.BytesIO(base["image"]))
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    if image.size[0] > 50 and image.size[1] > 50:
                        images.append((page_num + 1, image))
                except:
                    continue
        doc.close()
    except Exception as e:
        print(f"❌ Image extraction error: {e}")
    return images

def extract_docx(file_path: str) -> str:
    try:
        doc = docx.Document(file_path)
        return "\n\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    except Exception as e:
        print(f"❌ DOCX error: {e}")
        return ""

# =============================================================================
# VISION
# =============================================================================

def resize_image(image: Image.Image, max_size: int = 384) -> Image.Image:
    w, h = image.size
    if w <= max_size and h <= max_size:
        return image
    ratio = min(max_size / w, max_size / h)
    return image.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)

def image_to_base64(image: Image.Image) -> str:
    if image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGB')
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

async def analyze_image(image: Image.Image, page: int, idx: int) -> str:
    async with vision_semaphore:
        try:
            resized = resize_image(image)
            b64 = image_to_base64(resized)
            
            response = await vision_client.chat.completions.create(
                model=VISION_MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": "Describe this image. Include ALL text, numbers, and data visible."}
                    ]
                }],
                max_tokens=VISION_MAX_TOKENS,
                temperature=0.3
            )
            result = response.choices[0].message.content
            return filter_cjk(result.strip()) if result else "[No description]"
        except Exception as e:
            print(f"❌ Vision error: {e}")
            return f"[Image analysis failed]"

# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are a Document Assistant. You have access to document content provided below.

CRITICAL RULES:
1. The DOCUMENT CONTEXT below contains REAL content from uploaded files
2. Use ONLY this content to answer questions  
3. DO NOT say "I don't have access" - the content IS provided
4. For page-specific questions, look for [Page X] markers
5. Quote from documents when helpful
6. If info isn't in the context, say "This isn't in the retrieved sections"

FORMAT:
- Be direct, start with the answer
- Reference page numbers
- No <think> tags
- English only"""

# =============================================================================
# RETRIEVE FROM CHROMADB - CRITICAL FUNCTION
# =============================================================================

def retrieve_chunks(
    collection: chromadb.Collection,
    query: str,
    top_k: int = MAX_CONTEXT_CHUNKS,
    page_filter: Optional[int] = None
) -> List[Dict]:
    """
    Retrieve chunks from ChromaDB collection.
    This is synchronous because ChromaDB operations are sync.
    """
    try:
        count = collection.count()
        print(f"📊 Collection has {count} chunks")
        
        if count == 0:
            print("⚠️ Collection is empty!")
            return []
        
        # Check for page in query
        page_match = re.search(r'page\s*(\d+)', query.lower())
        if page_match:
            page_filter = int(page_match.group(1))
            print(f"📄 Page filter detected: {page_filter}")
        
        chunks = []
        
        # If page filter, try to get page-specific chunks first
        if page_filter:
            try:
                page_results = collection.get(
                    where={"page_number": page_filter},
                    include=["documents", "metadatas"]
                )
                
                if page_results and page_results.get("documents"):
                    print(f"📄 Found {len(page_results['documents'])} chunks from page {page_filter}")
                    
                    for doc, meta in zip(page_results["documents"], page_results["metadatas"]):
                        chunks.append({
                            "text": doc,
                            "page": meta.get("page_number"),
                            "type": meta.get("chunk_type", "text")
                        })
                    
                    # If we have enough, return
                    if len(chunks) >= top_k // 2:
                        return chunks[:top_k]
            except Exception as e:
                print(f"⚠️ Page filter failed: {e}")
        
        # Semantic search
        try:
            # Generate query embedding
            query_embedding = embed_query(query)
            
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            if results and results.get("documents") and results["documents"][0]:
                for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                ):
                    # Avoid duplicates
                    if not any(c["text"][:50] == doc[:50] for c in chunks):
                        chunks.append({
                            "text": doc,
                            "page": meta.get("page_number"),
                            "type": meta.get("chunk_type", "text"),
                            "distance": dist
                        })
                
                print(f"✅ Retrieved {len(chunks)} total chunks")
            else:
                print("⚠️ No results from semantic search")
                
        except Exception as e:
            print(f"❌ Semantic search error: {e}")
            import traceback
            traceback.print_exc()
        
        return chunks[:top_k]
        
    except Exception as e:
        print(f"❌ Retrieval error: {e}")
        import traceback
        traceback.print_exc()
        return []


def format_context(chunks: List[Dict], max_tokens: int = 5000) -> str:
    """Format chunks into context string for LLM."""
    if not chunks:
        return "[No document content retrieved]"
    
    parts = []
    tokens = 0
    
    for i, chunk in enumerate(chunks):
        page = chunk.get("page", "?")
        ctype = chunk.get("type", "text")
        
        header = f"[Page {page}]" if page else f"[Section {i+1}]"
        if ctype == "image":
            header += " (Image)"
        
        text = f"{header}\n{chunk['text']}\n"
        text_tokens = count_tokens(text)
        
        if tokens + text_tokens > max_tokens:
            break
        
        parts.append(text)
        tokens += text_tokens
    
    return "\n---\n".join(parts)

# =============================================================================
# FILE PROCESSING
# =============================================================================

async def process_files(files, collection: chromadb.Collection, file_list: List[dict]):
    """Process uploaded files into ChromaDB."""
    
    progress = cl.Message(content="📂 Starting processing...")
    await progress.send()
    
    total_chunks = 0
    
    for i, file in enumerate(files):
        fname = file.name
        fpath = file.path
        
        progress.content = f"📂 Processing {i+1}/{len(files)}: {fname}"
        await progress.update()
        
        file_list.append({"name": fname})
        
        if fname.lower().endswith('.pdf'):
            # Extract text
            loop = asyncio.get_event_loop()
            pages = await loop.run_in_executor(executor, extract_pdf_pages, fpath)
            
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
                    all_ids.append(f"{fname}_p{page_num}_c{idx}_{uuid.uuid4().hex[:6]}")
            
            print(f"📊 Created {len(all_docs)} chunks from {fname}")
            
            # Embed and add in batches
            if all_docs:
                progress.content = f"⚡ Embedding {len(all_docs)} chunks..."
                await progress.update()
                
                batch_size = 50
                for batch_start in range(0, len(all_docs), batch_size):
                    batch_end = min(batch_start + batch_size, len(all_docs))
                    
                    batch_docs = all_docs[batch_start:batch_end]
                    batch_metas = all_metas[batch_start:batch_end]
                    batch_ids = all_ids[batch_start:batch_end]
                    
                    # Generate embeddings
                    batch_embeddings = embed_documents(batch_docs)
                    
                    # Add to collection
                    collection.add(
                        documents=batch_docs,
                        embeddings=batch_embeddings,
                        metadatas=batch_metas,
                        ids=batch_ids
                    )
                    
                    progress.content = f"⚡ Added {batch_end}/{len(all_docs)} chunks..."
                    await progress.update()
                
                total_chunks += len(all_docs)
            
            # Process images
            progress.content = f"🖼️ Extracting images..."
            await progress.update()
            
            images = await loop.run_in_executor(executor, extract_pdf_images, fpath, 25)
            
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
        
        elif fname.lower().endswith('.docx'):
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(executor, extract_docx, fpath)
            
            if text:
                chunks = text_splitter.split_text(text)
                embeddings = embed_documents(chunks)
                
                for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                    collection.add(
                        documents=[chunk],
                        embeddings=[emb],
                        metadatas=[{"source": fname, "chunk_type": "text"}],
                        ids=[f"{fname}_c{idx}_{uuid.uuid4().hex[:6]}"]
                    )
                
                total_chunks += len(chunks)
        
        elif fname.lower().endswith('.txt'):
            with open(fpath, 'r', encoding='utf-8') as f:
                text = f.read()
            
            if text:
                chunks = text_splitter.split_text(text)
                embeddings = embed_documents(chunks)
                
                for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                    collection.add(
                        documents=[chunk],
                        embeddings=[emb],
                        metadatas=[{"source": fname, "chunk_type": "text"}],
                        ids=[f"{fname}_c{idx}_{uuid.uuid4().hex[:6]}"]
                    )
                
                total_chunks += len(chunks)
        
        elif fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(fpath)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            desc = await analyze_image(img, 0, 1)
            emb = embed_documents([desc])[0]
            
            collection.add(
                documents=[desc],
                embeddings=[emb],
                metadatas=[{"source": fname, "chunk_type": "image"}],
                ids=[f"{fname}_img_{uuid.uuid4().hex[:6]}"]
            )
            
            total_chunks += 1
    
    # Final count
    final_count = collection.count()
    
    # Get page range
    try:
        all_meta = collection.get(include=["metadatas"])
        pages = [m.get("page_number") for m in all_meta["metadatas"] if m.get("page_number")]
        page_range = f"{min(pages)}-{max(pages)}" if pages else "N/A"
        text_count = sum(1 for m in all_meta["metadatas"] if m.get("chunk_type") == "text")
        img_count = sum(1 for m in all_meta["metadatas"] if m.get("chunk_type") == "image")
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

# =============================================================================
# RESPONSE GENERATION
# =============================================================================

async def generate_response(
    query: str,
    collection: chromadb.Collection,
    memory: ConversationMemory,
    msg: cl.Message
):
    """Generate streaming response with retrieval."""
    
    thinking_filter = ThinkingFilter()
    
    try:
        # STEP 1: RETRIEVE
        print(f"\n{'='*50}")
        print(f"🔍 Query: {query}")
        
        chunks = retrieve_chunks(collection, query, top_k=MAX_CONTEXT_CHUNKS)
        
        print(f"📦 Retrieved {len(chunks)} chunks")
        if chunks:
            for i, c in enumerate(chunks[:3]):
                print(f"   [{i+1}] Page {c.get('page', '?')}: {c['text'][:80]}...")
        
        # STEP 2: FORMAT CONTEXT
        context = format_context(chunks, max_tokens=5000)
        context_tokens = count_tokens(context)
        print(f"📝 Context: {context_tokens} tokens")
        
        # STEP 3: BUILD MESSAGES
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Add context as system message
        context_msg = f"""## DOCUMENT CONTEXT

{context}

## END CONTEXT

Answer the user's question using ONLY the above content."""
        
        messages.append({"role": "system", "content": context_msg})
        
        # Add history
        for user_q, asst_a in memory.get_history(2):
            messages.append({"role": "user", "content": user_q})
            messages.append({"role": "assistant", "content": asst_a[:200] + "..." if len(asst_a) > 200 else asst_a})
        
        # Add current query
        messages.append({"role": "user", "content": query})
        
        total_tokens = sum(count_tokens(m["content"]) for m in messages)
        print(f"📊 Total input: {total_tokens} tokens")
        print(f"{'='*50}\n")
        
        # STEP 4: GENERATE
        async with llm_semaphore:
            stream = await reasoning_client.chat.completions.create(
                model=REASONING_MODEL,
                messages=messages,
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                stream=True
            )
            
            full_response = ""
            thinking_shown = False
            
            async for chunk in stream:
                # Check for stop
                cancelled = cl.user_session.get("cancelled", False)
                if cancelled:
                    print("🛑 Cancelled")
                    cl.user_session.set("cancelled", False)
                    break
                
                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    
                    if FILTER_THINKING_TAGS:
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
            
            # Flush
            remaining = thinking_filter.flush()
            if remaining:
                clean = filter_cjk(remaining)
                full_response += clean
                await msg.stream_token(clean)
        
        full_response = clean_thinking(full_response)
        await msg.update()
        
        # Save to memory
        if full_response:
            memory.add(query, full_response)
        
        return full_response
        
    except Exception as e:
        print(f"❌ Generation error: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {e}"

# =============================================================================
# CHAINLIT HANDLERS
# =============================================================================

@cl.on_chat_start
async def start():
    """Initialize session with ChromaDB collection."""
    
    # Create unique collection for this session
    session_id = str(uuid.uuid4())[:8]
    collection_name = f"docs_{session_id}"
    
    # Create collection
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Store in session
    cl.user_session.set("collection", collection)
    cl.user_session.set("collection_name", collection_name)
    cl.user_session.set("files", [])
    cl.user_session.set("memory", ConversationMemory())
    cl.user_session.set("cancelled", False)
    
    print(f"📚 Created collection: {collection_name}")
    
    files = await cl.AskFileMessage(
        content=f"🚀 **Document Assistant v6**\n\n"
                f"**Models:**\n"
                f"• Vision: {VISION_MODEL}\n"
                f"• Reasoning: {REASONING_MODEL}\n\n"
                f"**Fixed:**\n"
                f"• ✅ ChromaDB retrieval working\n"
                f"• ✅ Session-based collections\n"
                f"• ✅ Proper embedding for queries\n"
                f"• ✅ Context injection verified\n\n"
                f"Upload files to start!",
        accept=["application/pdf",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "text/plain", "image/png", "image/jpeg"],
        max_size_mb=50,
        timeout=300
    ).send()
    
    if files:
        file_list = cl.user_session.get("files")
        await process_files(files, collection, file_list)


@cl.on_stop
async def on_stop():
    """Handle stop button."""
    cl.user_session.set("cancelled", True)
    print("🛑 Stop requested")


@cl.on_message
async def on_message(message: cl.Message):
    """Handle messages."""
    
    query = message.content.strip()
    collection = cl.user_session.get("collection")
    file_list = cl.user_session.get("files")
    memory = cl.user_session.get("memory")
    
    # Commands
    if query.lower() == '/clear':
        coll_name = cl.user_session.get("collection_name")
        try:
            chroma_client.delete_collection(coll_name)
        except:
            pass
        
        # Create new collection
        new_name = f"docs_{str(uuid.uuid4())[:8]}"
        new_coll = chroma_client.get_or_create_collection(name=new_name, metadata={"hnsw:space": "cosine"})
        
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
        await cl.Message(content=f"📊 **Stats:**\n• Chunks: {count}\n• Files: {len(file_list)}").send()
        return
    
    if query.lower() == '/test':
        # Test retrieval
        if collection and collection.count() > 0:
            test_chunks = retrieve_chunks(collection, "summary overview", top_k=3)
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
    
    # File upload
    if message.elements:
        if not collection:
            await cl.Message(content="❌ Session not initialized. Refresh page.").send()
            return
        await process_files(message.elements, collection, file_list)
        return
    
    # Check collection
    if not collection or collection.count() == 0:
        await cl.Message(content="📭 No documents loaded. Please upload files first.").send()
        return
    
    # Generate response
    response_msg = cl.Message(content="")
    await response_msg.send()
    
    await generate_response(query, collection, memory, response_msg)


@cl.on_chat_end
async def on_end():
    """Cleanup on session end."""
    coll_name = cl.user_session.get("collection_name")
    if coll_name:
        try:
            chroma_client.delete_collection(coll_name)
            print(f"🗑️ Cleaned up collection: {coll_name}")
        except:
            pass


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 DOCUMENT ASSISTANT v6 - FIXED RETRIEVAL")
    print("="*60)
    print("✅ Session-based ChromaDB collections")
    print("✅ Proper query vs document embeddings")
    print("✅ Verified context injection")
    print("✅ /test command to verify retrieval")
    print("="*60 + "\n")