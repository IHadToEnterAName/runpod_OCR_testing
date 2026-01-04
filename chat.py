import chainlit as cl
import ollama
import json
import os
from pathlib import Path
import base64
from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image
import io
import fitz  # PyMuPDF
import docx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

# --- OLLAMA ENVIRONMENT CONFIGURATION ---
# Set these before importing/using Ollama to optimize performance
os.environ['OLLAMA_NUM_PARALLEL'] = '4'  # Max parallel requests (reduced from 8)
os.environ['OLLAMA_MAX_LOADED_MODELS'] = '3'  # Keep 3 models in memory
os.environ['OLLAMA_KEEP_ALIVE'] = '30m'  # Keep models loaded for 30 minutes
os.environ['OLLAMA_FLASH_ATTENTION'] = '1'  # Enable flash attention
os.environ['OLLAMA_HOST'] = '0.0.0.0:11434'  # Ollama server host

# --- CONFIGURATION ---
# Ollama model names
LLM_MODEL = "deepseek-r1"
VISION_MODEL = "qwen-vl-2.5"
EMBEDDING_MODEL = "nomic-embed-text"

# Performance settings - REDUCED for stability and deadlock prevention
EMBEDDING_BATCH_SIZE = 256  # Reduced from 512
VISION_CONCURRENT_LIMIT = 3  # Reduced from 8 to prevent GPU contention
MAX_CONTEXT_CHUNKS = 16
FILE_PROCESSING_WORKERS = 8  # Reduced from 16
LLM_MAX_TOKENS = 8192
VISION_MAX_TOKENS = 4096

# UI Update throttling to prevent Socket.IO overflow
UI_UPDATE_INTERVAL = 2.0  # Update UI every 2 seconds max
MIN_UPDATE_ITEMS = 5  # Update UI every N items processed

# Ollama performance options
OLLAMA_OPTIONS = {
    "num_ctx": 8192,
    "num_gpu": -1,
    "num_thread": 16,
    "num_batch": 512,
}

# --- SEMAPHORES FOR THREAD SAFETY ---
# These prevent deadlocks and manage concurrent access
embedding_semaphore = asyncio.Semaphore(4)  # Max 4 concurrent embedding calls
vision_semaphore = asyncio.Semaphore(VISION_CONCURRENT_LIMIT)  # Max 3 concurrent vision calls
llm_semaphore = asyncio.Semaphore(2)  # Max 2 concurrent LLM calls
file_lock = asyncio.Lock()  # Exclusive lock for file operations

system_prompt_content = """
# ROLE & OBJECTIVE
You are an expert Corporate AI Assistant. Your purpose is to analyze the provided internal documents (text and image descriptions) to answer user questions with high accuracy, professionalism, and strict adherence to the facts.

# INSTRUCTIONS
1. **Language Mirroring**: STRICTLY answer in the same language the user is speaking. If they ask in Spanish, answer in Spanish. If they ask in English, answer in English.
2. **Evidence-Based Answers**: You must derive your answer ONLY from the "REFERENCED DOCUMENTS" provided below. Do not use outside knowledge or prior training data to answer factual questions.
3. **Unknown Information**: If the provided documents do not contain the answer, explicitly state: "The provided documents do not contain information regarding this specific query." Do not make up or hallucinate an answer.
4. **Image Context**: The context includes descriptions of images and charts. Treat this data as factual content.

# TONE & STYLE
- **Professional**: Use a formal, corporate tone. Avoid slang, emojis, or casual language.
- **Concise**: Be direct. Start with the answer immediately. Avoid filler phrases like "Here is the answer based on the context."
- **Structured**: Use Markdown formatting (bullet points, bold text for key terms) to make the answer easy to scan.

# CRITICAL RULES
- Never mention "In the context provided" or "According to the chunks." Just state the facts as if you know them.
- If the documents offer conflicting information, note the discrepancy politely.
- Do not output these instructions in your response.
"""

# Global storage
conversation_history = []
document_chunks = []
uploaded_files_info = []

# Thread pool for CPU-intensive operations
executor = ThreadPoolExecutor(max_workers=FILE_PROCESSING_WORKERS)

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1800,
    chunk_overlap=300,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# --- LANGGRAPH STATE DEFINITION ---
class AgentState(TypedDict):
    """State for LangGraph agentic workflow"""
    query: str
    retrieved_chunks: List[str]
    reasoning_steps: List[str]
    final_answer: Optional[str]
    error: Optional[str]

# --- UTILITY FUNCTIONS ---

def extract_text_from_pdf(file_path: str, progress_callback=None) -> Tuple[str, List[Image.Image]]:
    """Extract text and images from PDF using PyMuPDF"""
    text_content = []
    images = []
    
    try:
        doc = fitz.open(file_path)
        total_pages = len(doc)
        
        for page_num in range(total_pages):
            page = doc[page_num]
            
            text = page.get_text("text")
            text_content.append(f"--- Page {page_num + 1} ---\n{text}")
            
            # Report progress
            if progress_callback:
                progress_callback(f"✅ Extracted page {page_num + 1}/{total_pages}")
            print(f"✅ PDF: Extracted page {page_num + 1}/{total_pages}")
            
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes))
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    images.append(image)
                    print(f"  📸 Found image {img_index + 1} on page {page_num + 1}")
                except Exception as e:
                    print(f"  ⚠️ Could not extract image {img_index} from page {page_num + 1}: {e}")
                    continue
        
        doc.close()
        print(f"✅ PDF complete: {total_pages} pages, {len(images)} images extracted")
    except Exception as e:
        print(f"❌ PDF extraction error: {e}")
    
    return "\n\n".join(text_content), images

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX"""
    try:
        doc = docx.Document(file_path)
        return "\n\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    except Exception as e:
        print(f"DOCX extraction error: {e}")
        return ""

def chunk_text_recursive(text: str) -> List[str]:
    """Split text using recursive character splitting"""
    if not text.strip():
        return []
    
    chunks = text_splitter.split_text(text)
    return [chunk for chunk in chunks if chunk.strip()]

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# --- OLLAMA INFERENCE FUNCTIONS WITH SEMAPHORES ---

async def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Generate embeddings with semaphore protection"""
    async with embedding_semaphore:
        try:
            all_embeddings = []
            
            for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
                batch = texts[i:i + EMBEDDING_BATCH_SIZE]
                
                loop = asyncio.get_event_loop()
                
                def embed_batch():
                    batch_embeddings = []
                    for text in batch:
                        response = ollama.embeddings(
                            model=EMBEDDING_MODEL,
                            prompt=text
                        )
                        batch_embeddings.append(response['embedding'])
                    return batch_embeddings
                
                embeddings = await loop.run_in_executor(executor, embed_batch)
                all_embeddings.extend(embeddings)
                
                # Small delay to prevent overwhelming the GPU
                await asyncio.sleep(0.1)
            
            return all_embeddings
        except Exception as e:
            print(f"Embedding Error: {e}")
            import traceback
            traceback.print_exc()
            return []

async def analyze_image_with_vision(image: Image.Image, image_num: int = 0, total_images: int = 0, prompt: str = "Describe this image in detail, including any text, charts, graphs, or diagrams. Be precise with numbers and data.") -> str:
    """Use Llama 3.2 Vision with semaphore protection"""
    async with vision_semaphore:
        try:
            print(f"🖼️  Processing image {image_num}/{total_images}...")
            
            img_base64 = image_to_base64(image)
            
            loop = asyncio.get_event_loop()
            
            def vision_inference():
                response = ollama.generate(
                    model=VISION_MODEL,
                    prompt=prompt,
                    images=[img_base64],
                    options={
                        "num_ctx": VISION_MAX_TOKENS,
                        "num_gpu": -1,
                        "num_thread": 8,
                        "temperature": 0.6,
                    }
                )
                return response['response']
            
            description = await loop.run_in_executor(executor, vision_inference)
            
            print(f"✅ Completed image {image_num}/{total_images}")
            
            # Delay between vision calls to prevent GPU contention
            await asyncio.sleep(0.5)
            
            return description.strip()
            
        except Exception as e:
            print(f"❌ Vision Error on image {image_num}/{total_images}: {e}")
            import traceback
            traceback.print_exc()
            return "Error analyzing image."

async def analyze_images_concurrent(images: List[Image.Image], progress_msg=None) -> List[str]:
    """Analyze images with controlled concurrency (3 concurrent)"""
    all_descriptions = []
    total_images = len(images)
    last_update_time = time.time()
    
    print(f"\n🖼️  Starting analysis of {total_images} images...")
    
    # Process 3 images at a time (reduced from 8)
    for i in range(0, len(images), VISION_CONCURRENT_LIMIT):
        batch = images[i:i + VISION_CONCURRENT_LIMIT]
        batch_num = i // VISION_CONCURRENT_LIMIT + 1
        total_batches = (total_images + VISION_CONCURRENT_LIMIT - 1) // VISION_CONCURRENT_LIMIT
        
        print(f"\n📦 Batch {batch_num}/{total_batches}: Processing {len(batch)} images concurrently...")
        
        # Update UI only every UI_UPDATE_INTERVAL seconds
        current_time = time.time()
        if progress_msg and (current_time - last_update_time >= UI_UPDATE_INTERVAL or batch_num == 1 or batch_num == total_batches):
            progress_msg.content = f"🖼️ Analyzing images: Batch {batch_num}/{total_batches} ({i+1}-{min(i+len(batch), total_images)}/{total_images})"
            await progress_msg.update()
            last_update_time = current_time
        
        # Create tasks with image numbers
        batch_tasks = [
            analyze_image_with_vision(img, i+j+1, total_images) 
            for j, img in enumerate(batch)
        ]
        descriptions = await asyncio.gather(*batch_tasks)
        all_descriptions.extend(descriptions)
        
        print(f"✅ Batch {batch_num}/{total_batches} complete")
        
        # Delay between batches
        await asyncio.sleep(1.0)
    
    # Final UI update
    if progress_msg:
        progress_msg.content = f"✅ Completed analysis of {total_images} images"
        await progress_msg.update()
    
    print(f"\n🎉 All {total_images} images analyzed!\n")
    return all_descriptions

async def generate_completion(messages_list: List[Dict], temperature: float = 0.6) -> Dict:
    """Generate completion with semaphore protection"""
    async with llm_semaphore:
        try:
            formatted_messages = []
            for msg in messages_list:
                formatted_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            loop = asyncio.get_event_loop()
            
            def llm_inference():
                response = ollama.chat(
                    model=LLM_MODEL,
                    messages=formatted_messages,
                    options={
                        **OLLAMA_OPTIONS,
                        "temperature": temperature,
                        "num_predict": 2048,
                    }
                )
                return response
            
            response = await loop.run_in_executor(executor, llm_inference)
            
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": response['message']['content']
                        }
                    }
                ]
            }
        except Exception as e:
            print(f"Generation Error: {e}")
            import traceback
            traceback.print_exc()
            return None

# --- LANGGRAPH AGENTIC WORKFLOW ---

async def retrieve_context_node(state: AgentState) -> AgentState:
    """Node 1: Retrieve relevant context chunks"""
    query = state['query']
    state['reasoning_steps'].append(f"🔍 Retrieving relevant context for: '{query[:50]}...'")
    
    if not document_chunks:
        state['retrieved_chunks'] = []
        state['reasoning_steps'].append("⚠️ No documents loaded")
        return state
    
    # Retrieve chunks
    chunks = await retrieve_relevant_chunks(query, top_k=MAX_CONTEXT_CHUNKS)
    state['retrieved_chunks'] = chunks
    state['reasoning_steps'].append(f"✅ Retrieved {len(chunks)} relevant chunks")
    
    return state

async def reasoning_node(state: AgentState) -> AgentState:
    """Node 2: Analyze if retrieved context is sufficient"""
    chunks = state['retrieved_chunks']
    
    if not chunks:
        state['reasoning_steps'].append("🤔 No context available - will provide general response")
    elif len(chunks) < 3:
        state['reasoning_steps'].append("🤔 Limited context found - may need clarification")
    else:
        state['reasoning_steps'].append(f"🤔 Analyzing {len(chunks)} chunks to formulate answer")
    
    return state

async def answer_generation_node(state: AgentState) -> AgentState:
    """Node 3: Generate final answer"""
    state['reasoning_steps'].append("✍️ Generating response...")
    
    messages = [{"role": "system", "content": system_prompt_content}]
    
    if state['retrieved_chunks']:
        context_text = "\n\n---DOCUMENT CONTEXT---\n" + "\n\n".join(state['retrieved_chunks']) + "\n---END CONTEXT---\n"
        messages.append({"role": "system", "content": context_text})
    
    for user_msg, assistant_msg in conversation_history[-5:]:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    
    messages.append({"role": "user", "content": state['query']})
    
    response = await generate_completion(messages)
    
    if response and 'choices' in response:
        state['final_answer'] = response['choices'][0]['message']['content']
        state['reasoning_steps'].append("✅ Answer generated successfully")
    else:
        state['error'] = "Failed to generate response"
        state['reasoning_steps'].append("❌ Error generating answer")
    
    return state

# Build LangGraph workflow
def build_agent_graph():
    """Build the agentic reasoning graph"""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("retrieve", retrieve_context_node)
    workflow.add_node("reason", reasoning_node)
    workflow.add_node("answer", answer_generation_node)
    
    # Define edges
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "reason")
    workflow.add_edge("reason", "answer")
    workflow.add_edge("answer", END)
    
    return workflow.compile()

# Initialize agent graph
agent_graph = build_agent_graph()

# --- RETRIEVAL FUNCTIONS ---

def cosine_similarity_batch(query_embedding: List[float], chunk_embeddings: np.ndarray) -> np.ndarray:
    """Vectorized cosine similarity calculation"""
    query_vec = np.array(query_embedding)
    query_norm = query_vec / np.linalg.norm(query_vec)
    
    chunk_norms = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
    chunk_embeddings_normalized = chunk_embeddings / chunk_norms
    
    similarities = np.dot(chunk_embeddings_normalized, query_norm)
    
    return similarities

async def retrieve_relevant_chunks(query: str, top_k: int = 5) -> List[str]:
    """Retrieve most relevant document chunks"""
    if not document_chunks:
        return []
    
    query_embeddings = await generate_embeddings_batch([query])
    if not query_embeddings or len(query_embeddings) == 0:
        return []
    
    query_embedding = query_embeddings[0]
    
    chunk_embeddings_array = np.array([chunk['embedding'] for chunk in document_chunks])
    
    similarities = cosine_similarity_batch(query_embedding, chunk_embeddings_array)
    
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    return [document_chunks[i]['text'] for i in top_indices]

# --- FILE MANAGEMENT FUNCTIONS ---

async def clear_all_documents():
    """Clear all documents with file lock"""
    async with file_lock:
        global document_chunks, uploaded_files_info
        document_chunks = []
        uploaded_files_info = []
        await cl.Message(content="🗑️ All documents have been cleared from memory.").send()

async def show_uploaded_files():
    """Display list of currently uploaded files"""
    async with file_lock:
        if not uploaded_files_info:
            await cl.Message(content="📭 No documents are currently loaded.").send()
        else:
            file_list = "\n".join([f"📄 {i+1}. {file['name']}" for i, file in enumerate(uploaded_files_info)])
            await cl.Message(content=f"📚 **Currently loaded documents:**\n{file_list}").send()

# --- FILE PROCESSING FUNCTION ---

async def process_files(files):
    """Process uploaded files with file lock"""
    async with file_lock:
        global document_chunks, uploaded_files_info
        
        await cl.Message(content="🚀 Processing with production-ready Ollama stack...").send()
        
        all_text = []
        all_images = []
        
        # Create a live progress message
        file_progress_msg = cl.Message(content="📂 Starting file processing...")
        await file_progress_msg.send()
        
        last_update_time = time.time()
        
        # Process files in parallel
        async def process_single_file(file, file_num, total_files):
            file_path = file.path
            file_name = file.name
            
            print(f"\n📄 [{file_num}/{total_files}] Processing: {file_name}")
            
            # Update UI only periodically
            nonlocal last_update_time
            current_time = time.time()
            if current_time - last_update_time >= UI_UPDATE_INTERVAL or file_num == 1 or file_num == total_files:
                file_progress_msg.content = f"📂 Processing file {file_num}/{total_files}: {file_name}"
                await file_progress_msg.update()
                last_update_time = current_time
            
            uploaded_files_info.append({"name": file_name, "path": file_path})
            
            if file_name.lower().endswith('.pdf'):
                print(f"📕 Extracting PDF: {file_name}")
                loop = asyncio.get_event_loop()
                
                def extract_with_progress():
                    return extract_text_from_pdf(file_path)
                
                text, images = await loop.run_in_executor(executor, extract_with_progress)
                print(f"✅ PDF complete: {file_name}")
                return text, images
            elif file_name.lower().endswith('.docx'):
                print(f"📘 Extracting DOCX: {file_name}")
                loop = asyncio.get_event_loop()
                text = await loop.run_in_executor(executor, extract_text_from_docx, file_path)
                print(f"✅ DOCX complete: {file_name}")
                return text, []
            elif file_name.lower().endswith('.txt'):
                print(f"📝 Reading TXT: {file_name}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                print(f"✅ TXT complete: {file_name}")
                return text, []
            elif file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                print(f"🖼️  Loading image: {file_name}")
                img = Image.open(file_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                print(f"✅ Image loaded: {file_name}")
                return "", [img]
            return "", []
        
        total_files = len(files)
        results = await asyncio.gather(*[
            process_single_file(file, i+1, total_files) 
            for i, file in enumerate(files)
        ])
        
        for text, images in results:
            if text:
                all_text.append(text)
            all_images.extend(images)
        
        file_progress_msg.content = f"✅ All {total_files} files extracted!"
        await file_progress_msg.update()
        
        # --- TEXT PROCESSING ---
        full_text = "\n\n".join(all_text)
        if full_text.strip():
            chunks = chunk_text_recursive(full_text)
            print(f"\n📊 Created {len(chunks)} text chunks")
            
            progress_msg = cl.Message(content="⚡ Generating embeddings...")
            await progress_msg.send()
            
            last_embed_update = time.time()
            
            for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
                batch = chunks[i:i + EMBEDDING_BATCH_SIZE]
                batch_num = i // EMBEDDING_BATCH_SIZE + 1
                total_batches = (len(chunks) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE
                
                print(f"⚡ Embedding batch {batch_num}/{total_batches}...")
                
                embeddings = await generate_embeddings_batch(batch)
                
                if embeddings:
                    for j, embedding in enumerate(embeddings):
                        if embedding:
                            document_chunks.append({
                                'text': batch[j],
                                'embedding': embedding
                            })
                
                progress = min(i + EMBEDDING_BATCH_SIZE, len(chunks))
                
                # Update UI periodically
                current_time = time.time()
                if current_time - last_embed_update >= UI_UPDATE_INTERVAL or batch_num == 1 or batch_num == total_batches:
                    progress_msg.content = f"⚡ Embeddings: {progress}/{len(chunks)} (batch {batch_num}/{total_batches})"
                    await progress_msg.update()
                    last_embed_update = current_time
                
                print(f"✅ Batch {batch_num}/{total_batches} complete")
            
            progress_msg.content = f"✅ Processed {len(chunks)} text chunks"
            await progress_msg.update()
        
        # --- IMAGE PROCESSING ---
        if all_images:
            vision_msg = cl.Message(content=f"🖼️ Starting image analysis...")
            await vision_msg.send()
            
            descriptions = await analyze_images_concurrent(all_images[:100], vision_msg)
            image_descriptions = [f"Image {i+1}: {desc}" for i, desc in enumerate(descriptions)]
            
            vision_msg.content = f"✅ Analyzed {len(descriptions)} images"
            await vision_msg.update()
            
            if image_descriptions:
                combined_images_text = "\n\n".join(image_descriptions)
                img_chunks = chunk_text_recursive(combined_images_text)
                
                print(f"\n📊 Created {len(img_chunks)} image description chunks")
                
                embed_img_msg = cl.Message(content="⚡ Embedding image descriptions...")
                await embed_img_msg.send()
                
                last_img_embed_update = time.time()
                
                for i in range(0, len(img_chunks), EMBEDDING_BATCH_SIZE):
                    batch = img_chunks[i:i + EMBEDDING_BATCH_SIZE]
                    batch_num = i // EMBEDDING_BATCH_SIZE + 1
                    total_batches = (len(img_chunks) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE
                    
                    embeddings = await generate_embeddings_batch(batch)
                    if embeddings:
                        for j, embedding in enumerate(embeddings):
                            if embedding:
                                document_chunks.append({
                                    'text': batch[j],
                                    'embedding': embedding
                                })
                    
                    progress = min(i + EMBEDDING_BATCH_SIZE, len(img_chunks))
                    
                    # Update UI periodically
                    current_time = time.time()
                    if current_time - last_img_embed_update >= UI_UPDATE_INTERVAL or batch_num == 1 or batch_num == total_batches:
                        embed_img_msg.content = f"⚡ Image embeddings: {progress}/{len(img_chunks)} (batch {batch_num}/{total_batches})"
                        await embed_img_msg.update()
                        last_img_embed_update = current_time
                
                embed_img_msg.content = f"✅ Embedded {len(img_chunks)} image description chunks"
                await embed_img_msg.update()
        
        await cl.Message(
            content=f"🎉 **Processing Complete!**\n\n"
                    f"📊 **Results:**\n"
                    f"• Chunks indexed: **{len(document_chunks)}**\n"
                    f"• Files processed: **{len(uploaded_files_info)}**\n"
                    f"• Text chunks: **{len([c for c in document_chunks if 'Image' not in c['text'][:10]])}**\n"
                    f"• Image descriptions: **{len([c for c in document_chunks if 'Image' in c['text'][:10]])}**\n\n"
                    f"⚙️ **Configuration:**\n"
                    f"• Vision concurrency: **{VISION_CONCURRENT_LIMIT}**\n"
                    f"• Embedding batch: **{EMBEDDING_BATCH_SIZE}**\n"
                    f"• Ollama parallel: **4** requests\n"
                    f"• Keep-alive: **30 minutes**\n\n"
                    f"💡 **Commands:**\n"
                    f"• `/clear` - Delete all documents\n"
                    f"• `/files` - View loaded documents"
        ).send()

# --- CHAINLIT EVENT HANDLERS ---

@cl.on_chat_start
async def start():
    files = await cl.AskFileMessage(
        content="Welcome to Production-Ready RAG! 🚀\n\n"
                "**Enhanced Features:**\n"
                f"✨ LLM: {LLM_MODEL}\n"
                f"🔍 Vision: {VISION_MODEL} ({VISION_CONCURRENT_LIMIT} concurrent)\n"
                f"📊 Embeddings: {EMBEDDING_MODEL}\n"
                f"🤖 LangGraph agentic reasoning\n"
                f"🔒 Semaphore-based thread safety\n"
                f"⚡ Ollama parallel requests: 4\n"
                f"🕐 Model keep-alive: 30 minutes\n\n"
                "**File Management:**\n"
                "• Upload files anytime\n"
                "• `/clear` to delete all documents\n"
                "• `/files` to see loaded documents\n\n"
                "Upload a file or click 'Skip' to chat!",
        accept=["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", 
                "text/plain", "image/png", "image/jpeg", "image/jpg", "image/gif", "image/bmp"],
        max_size_mb=20,
        timeout=180,
    ).send()
    
    if files:
        await process_files(files)

@cl.on_message
async def on_message(message: cl.Message):
    global document_chunks, uploaded_files_info
    
    # --- HANDLE COMMANDS ---
    if message.content.strip().lower() == '/clear':
        await clear_all_documents()
        return
    
    if message.content.strip().lower() == '/files':
        await show_uploaded_files()
        return
    
    # --- HANDLE FILE UPLOADS ---
    if message.elements:
        await process_files(message.elements)
        return
    
    # --- HANDLE REGULAR MESSAGES WITH LANGGRAPH ---
    
    thinking_msg = cl.Message(content="🤔 Reasoning through your question...")
    await thinking_msg.send()
    
    # Initialize agent state
    initial_state: AgentState = {
        "query": message.content,
        "retrieved_chunks": [],
        "reasoning_steps": [],
        "final_answer": None,
        "error": None
    }
    
    # Run the agentic workflow
    final_state = await agent_graph.ainvoke(initial_state)
    
    # Display reasoning steps (optional, for transparency)
    reasoning_text = "\n".join(final_state['reasoning_steps'])
    print(f"Reasoning Steps:\n{reasoning_text}")
    
    if final_state['final_answer']:
        thinking_msg.content = final_state['final_answer']
        await thinking_msg.update()
        conversation_history.append((message.content, final_state['final_answer']))
    else:
        error_msg = final_state.get('error', 'Unknown error')
        thinking_msg.content = f"❌ Error: {error_msg}\n\nReasoning:\n{reasoning_text}"
        await thinking_msg.update()