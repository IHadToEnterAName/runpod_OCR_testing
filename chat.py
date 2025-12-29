import chainlit as cl
import httpx
import json
from pathlib import Path
import base64
from typing import List, Dict, Tuple
import numpy as np
from PIL import Image
import io
import fitz  # PyMuPDF
import docx
from langchain_text_splitters import RecursiveCharacterTextSplitter
import asyncio
from concurrent.futures import ThreadPoolExecutor

# --- CONFIGURATION ---
model = "mistral"
vision_model = "llama3.2-vision"  # Modern: llama3.2-vision or moondream2
embedding_model = "nomic-embed-text"  # ollama pull nomic-embed-text
base_url = "" #TODO: PUT PROXY LINK FOR OLLAMA HERE

# Performance settings optimized for 32GB VRAM (targeting ~20-24GB usage, leaving 8-12GB free)
EMBEDDING_BATCH_SIZE = 300  # Balanced batch size for GPU utilization
VISION_CONCURRENT_LIMIT = 12  # Balanced concurrent vision processing
MAX_CONTEXT_CHUNKS = 12  # Balanced context chunks
FILE_PROCESSING_WORKERS = 12  # Balanced parallel file processing

system_prompt_content = """You are a helpful AI assistant for corporate use. 
Answer formally and answer in the same language you are questioned with.
When provided with document context, use it to answer questions accurately."""

# Global storage
conversation_history = []
document_chunks = []  # Store text chunks with embeddings
uploaded_files_info = []  # Track uploaded files

# Thread pool for CPU-intensive operations - more workers for 32GB RAM
executor = ThreadPoolExecutor(max_workers=FILE_PROCESSING_WORKERS)

# Initialize recursive text splitter with optimized chunk sizes for 32GB VRAM
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1800,  # Balanced larger chunks
    chunk_overlap=300,  # Better context preservation
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# --- UTILITY FUNCTIONS ---

def extract_text_from_pdf(file_path: str) -> Tuple[str, List[Image.Image]]:
    """Extract text and images from PDF using PyMuPDF (faster and more accurate)"""
    text_content = []
    images = []
    
    try:
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Extract text with better layout preservation
            text = page.get_text("text")
            text_content.append(f"--- Page {page_num + 1} ---\n{text}")
            
            # Extract images from page
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes))
                    images.append(image)
                except Exception as e:
                    print(f"Could not extract image {img_index} from page {page_num + 1}: {e}")
                    continue
        
        doc.close()
    except Exception as e:
        print(f"PDF extraction error: {e}")
    
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
    """Split text using recursive character splitting for better semantic preservation"""
    if not text.strip():
        return []
    
    chunks = text_splitter.split_text(text)
    return [chunk for chunk in chunks if chunk.strip()]

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# --- API FUNCTIONS ---

async def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Generate embeddings in massive batches using /api/embed endpoint - optimized for 32GB VRAM"""
    url = f"{base_url}/api/embed"
    headers = {"Content-Type": "application/json"}
    
    data = {
        "model": embedding_model,
        "input": texts,  # Send large batch of texts
        "options": {
            "num_gpu": -1,  # Use all available GPUs
            "num_thread": 24,  # Balanced CPU threads
            "num_batch": 1536,  # Balanced batch processing
        }
    }
    
    timeout = httpx.Timeout(180.0)  # Longer timeout for massive batches
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.post(url, headers=headers, json=data)
            if response.status_code == 200:
                result = response.json()
                return result.get('embeddings', [])
            else:
                print(f"Embedding Error ({response.status_code}): {response.text}")
                return []
        except Exception as e:
            print(f"Embedding Error: {e}")
            return []

async def analyze_image_with_vision(image: Image.Image, prompt: str = "Describe this image in detail, including any text, charts, graphs, or diagrams. Be precise with numbers and data.") -> str:
    """Use modern vision model to analyze images - optimized for 32GB VRAM"""
    url = f"{base_url}/api/generate"
    headers = {"Content-Type": "application/json"}
    
    # Convert image to base64
    img_base64 = image_to_base64(image)
    
    data = {
        "model": vision_model,
        "prompt": prompt,
        "images": [img_base64],
        "stream": False,
        "options": {
            "num_gpu": -1,  # Use all GPUs
            "num_thread": 24,  # Balanced threads
            "num_ctx": 12288,  # Balanced context for vision processing
            "num_batch": 768,  # Balanced batch size
        }
    }
    
    timeout = httpx.Timeout(240.0)  # Longer timeout for high-quality processing
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.post(url, headers=headers, json=data)
            if response.status_code == 200:
                return response.json().get('response', 'Could not analyze image.')
            else:
                print(f"Vision API Error ({response.status_code}): {response.text}")
                return "Error analyzing image."
        except Exception as e:
            print(f"Vision Error: {e}")
            return "Error analyzing image."

async def analyze_images_concurrent(images: List[Image.Image]) -> List[str]:
    """Analyze multiple images concurrently - optimized for 32GB VRAM (8 at a time)"""
    all_descriptions = []
    
    # Process in batches of VISION_CONCURRENT_LIMIT (8 images simultaneously)
    for i in range(0, len(images), VISION_CONCURRENT_LIMIT):
        batch = images[i:i + VISION_CONCURRENT_LIMIT]
        batch_tasks = [analyze_image_with_vision(img) for img in batch]
        descriptions = await asyncio.gather(*batch_tasks)
        all_descriptions.extend(descriptions)
    
    return all_descriptions

async def generate_completion(messages_list: List[Dict], model: str):
    """Generate chat completion - optimized for 32GB VRAM"""
    url = f"{base_url}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    
    data = {
        "model": model,
        "messages": messages_list,
        "temperature": 0.6,
        "options": {
            "num_gpu": -1,  # Use all GPUs
            "num_thread": 24,  # Balanced threads
            "num_ctx": 24576,  # Balanced larger context window
            "num_batch": 1536,  # Balanced batch size
            "num_predict": 2048,  # Allow longer responses
        }
    }
    
    timeout = httpx.Timeout(180.0)  # Longer timeout for quality responses
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.post(url, headers=headers, json=data)
            
            if response.status_code != 200:
                print(f"API Error ({response.status_code}): {response.text}")
                return None
            
            return response.json()
            
        except json.JSONDecodeError:
            print("CRITICAL ERROR: Server returned text/html instead of JSON.")
            print("Raw Response:", response.text)
            return None
        except Exception as e:
            print(f"Connection Error: {e}")
            return None

def cosine_similarity_batch(query_embedding: List[float], chunk_embeddings: np.ndarray) -> np.ndarray:
    """Vectorized cosine similarity calculation for better performance"""
    query_vec = np.array(query_embedding)
    
    # Normalize query vector
    query_norm = query_vec / np.linalg.norm(query_vec)
    
    # Normalize all chunk embeddings
    chunk_norms = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
    chunk_embeddings_normalized = chunk_embeddings / chunk_norms
    
    # Compute all similarities at once (vectorized)
    similarities = np.dot(chunk_embeddings_normalized, query_norm)
    
    return similarities

async def retrieve_relevant_chunks(query: str, top_k: int = 5) -> List[str]:
    """Retrieve most relevant document chunks using vectorized similarity"""
    if not document_chunks:
        return []
    
    # Generate query embedding (batch of 1)
    query_embeddings = await generate_embeddings_batch([query])
    if not query_embeddings or len(query_embeddings) == 0:
        return []
    
    query_embedding = query_embeddings[0]
    
    # Convert all chunk embeddings to numpy array for vectorized operations
    chunk_embeddings_array = np.array([chunk['embedding'] for chunk in document_chunks])
    
    # Calculate all similarities at once (much faster than loop)
    similarities = cosine_similarity_batch(query_embedding, chunk_embeddings_array)
    
    # Get top_k indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Return corresponding chunks
    return [document_chunks[i]['text'] for i in top_indices]

# --- FILE MANAGEMENT FUNCTIONS ---

async def clear_all_documents():
    """Clear all uploaded documents and their chunks"""
    global document_chunks, uploaded_files_info
    document_chunks = []
    uploaded_files_info = []
    await cl.Message(content="🗑️ All documents have been cleared from memory.").send()

async def show_uploaded_files():
    """Display list of currently uploaded files"""
    if not uploaded_files_info:
        await cl.Message(content="📭 No documents are currently loaded.").send()
    else:
        file_list = "\n".join([f"📄 {i+1}. {file['name']}" for i, file in enumerate(uploaded_files_info)])
        await cl.Message(content=f"📚 **Currently loaded documents:**\n{file_list}").send()

# --- FILE PROCESSING FUNCTION ---

async def process_files(files):
    """Process uploaded files with maximum parallelism - optimized for 32GB VRAM"""
    global document_chunks, uploaded_files_info
    
    await cl.Message(content="🚀 Processing with balanced GPU acceleration (32GB VRAM optimization targeting 20-24GB usage)...").send()
    
    all_text = []
    all_images = []
    
    # Process files in parallel using thread pool (8 workers)
    async def process_single_file(file):
        file_path = file.path
        file_name = file.name
        
        uploaded_files_info.append({"name": file_name, "path": file_path})
        
        if file_name.lower().endswith('.pdf'):
            loop = asyncio.get_event_loop()
            text, images = await loop.run_in_executor(executor, extract_text_from_pdf, file_path)
            return text, images
        elif file_name.lower().endswith('.docx'):
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(executor, extract_text_from_docx, file_path)
            return text, []
        elif file_name.lower().endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read(), []
        elif file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            img = Image.open(file_path)
            return "", [img]
        return "", []
    
    # Process all files concurrently
    results = await asyncio.gather(*[process_single_file(file) for file in files])
    
    for text, images in results:
        if text:
            all_text.append(text)
        all_images.extend(images)
    
    # --- TEXT PROCESSING WITH MASSIVE BATCHES (200 chunks at once) ---
    full_text = "\n\n".join(all_text)
    if full_text.strip():
        chunks = chunk_text_recursive(full_text)
        
        progress_msg = cl.Message(content="⚡ Generating embeddings with 32GB VRAM optimization...")
        await progress_msg.send()
        
        # Use massive batch size for lightning-fast processing
        for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
            batch = chunks[i:i + EMBEDDING_BATCH_SIZE]
            embeddings = await generate_embeddings_batch(batch)
            
            if embeddings:
                for j, embedding in enumerate(embeddings):
                    if embedding:
                        document_chunks.append({
                            'text': batch[j],
                            'embedding': embedding
                        })
            
            progress = min(i + EMBEDDING_BATCH_SIZE, len(chunks))
            progress_msg.content = f"⚡ Embeddings: {progress}/{len(chunks)} | Batch: {EMBEDDING_BATCH_SIZE} chunks | VRAM: ~20-24GB"
            await progress_msg.update()
        
        progress_msg.content = f"✅ Processed {len(chunks)} text chunks in massive batches"
        await progress_msg.update()
    
    # --- CONCURRENT IMAGE PROCESSING (8 images simultaneously) ---
    if all_images:
        vision_msg = cl.Message(content="🖼️ Analyzing images: 8 concurrent streams with 32GB VRAM...")
        await vision_msg.send()
        
        # Process up to 50 images with 8 concurrent streams
        descriptions = await analyze_images_concurrent(all_images[:50])
        image_descriptions = [f"Image {i+1}: {desc}" for i, desc in enumerate(descriptions)]
        
        vision_msg.content = f"✅ Analyzed {len(descriptions)} images (8 concurrent streams)"
        await vision_msg.update()
        
        if image_descriptions:
            combined_images_text = "\n\n".join(image_descriptions)
            img_chunks = chunk_text_recursive(combined_images_text)
            
            # Process image embeddings in massive batches
            for i in range(0, len(img_chunks), EMBEDDING_BATCH_SIZE):
                batch = img_chunks[i:i + EMBEDDING_BATCH_SIZE]
                embeddings = await generate_embeddings_batch(batch)
                if embeddings:
                    for j, embedding in enumerate(embeddings):
                        if embedding:
                            document_chunks.append({
                                'text': batch[j],
                                'embedding': embedding
                            })
    
    await cl.Message(
        content=f"🎉 **Processing Complete!**\n\n"
                f"📊 **Performance Stats:**\n"
                f"• Total chunks indexed: **{len(document_chunks)}**\n"
                f"• Files processed: **{len(uploaded_files_info)}**\n"
                f"• Embedding batch size: **{EMBEDDING_BATCH_SIZE}** (massive!)\n"
                f"• Vision concurrent streams: **{VISION_CONCURRENT_LIMIT}**\n"
                f"• File workers: **{FILE_PROCESSING_WORKERS}**\n"
                f"• Target VRAM usage: **~24-26GB** (leaving 6-8GB free)\n"
                f"• Context window: **16K tokens**\n\n"
                f"💡 **Commands:**\n"
                f"• `/clear` - Delete all documents\n"
                f"• `/files` - View loaded documents\n"
                f"• Upload files anytime to add more"
    ).send()

# --- CHAINLIT EVENT HANDLERS ---

@cl.on_chat_start
async def start():
    # Enable file uploads
    files = await cl.AskFileMessage(
        content="Welcome! I can help you with:\n"
                "📄 Upload documents (PDF, DOCX, TXT)\n"
                "🖼️ Analyze images with OCR/Vision AI\n"
                "💬 Answer questions based on your documents\n\n"
                "**Using Modern Stack:**\n"
                "✨ Llama 3.2 Vision for superior OCR\n"
                "⚡ Batch embeddings for faster processing\n"
                "📚 PyMuPDF for accurate text extraction\n\n"
                "**File Management:**\n"
                "• Upload files anytime during conversation\n"
                "• Type `/clear` to delete all documents\n"
                "• Type `/files` to see loaded documents\n\n"
                "Please upload a file to get started, or click 'Skip' to chat without documents.",
        accept=["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", 
                "text/plain", "image/png", "image/jpeg", "image/jpg", "image/gif", "image/bmp"],
        max_size_mb=20,
        timeout=180,
    ).send()
    
    if files:
        # Process uploaded files immediately
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
    
    # --- HANDLE FILE UPLOADS (via attachment) ---
    if message.elements:
        await process_files(message.elements)
        return
    
    # --- HANDLE REGULAR MESSAGES ---
    
    # Retrieve relevant context if documents are loaded
    context_chunks = []
    if document_chunks and message.content:
        context_chunks = await retrieve_relevant_chunks(message.content, top_k=MAX_CONTEXT_CHUNKS)
    
    # Build messages list
    messages = [{"role": "system", "content": system_prompt_content}]
    
    # Add document context if available
    if context_chunks:
        context_text = "\n\n---DOCUMENT CONTEXT---\n" + "\n\n".join(context_chunks) + "\n---END CONTEXT---\n"
        messages.append({"role": "system", "content": context_text})
    
    # Add conversation history
    for user_msg, assistant_msg in conversation_history[-5:]:  # Keep last 5 exchanges
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    
    # Add current message
    messages.append({"role": "user", "content": message.content})
    
    # Get response
    response = await generate_completion(messages, model)
    
    if response and 'choices' in response:
        assistant_response = response['choices'][0]['message']['content']
        await cl.Message(content=assistant_response).send()
        conversation_history.append((message.content, assistant_response))
    else:
        error_msg = "Error: The API returned an invalid response. Check terminal logs."
        await cl.Message(content=error_msg).send()