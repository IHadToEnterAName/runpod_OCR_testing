"""
API Routes
===========
REST endpoints for document management.

POST   /api/documents/upload          - Upload and chunk a document
GET    /api/documents/chunks          - Retrieve relevant chunks by query
DELETE /api/documents/{document_id}   - Delete a document and its chunks
"""

import os
import shutil
import tempfile
import uuid

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query

from config.settings import get_config
from storage.visual_store import get_visual_store
from api.models import UploadResponse, ChunksResponse, ChunkMetadata, DeleteResponse
from api.document_registry import get_document_registry, DocumentRecord

config = get_config()
router = APIRouter(prefix="/api/documents", tags=["Documents"])

# Persistent directory for storing uploaded files (needed for index rebuilds)
UPLOAD_STORE = os.path.join(config.byaldi.index_path, "api_uploads")
os.makedirs(UPLOAD_STORE, exist_ok=True)

# Shared Byaldi index name for all API documents
API_INDEX = "api_documents"

# Supported file extensions
SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.docx', '.json', '.png', '.jpg', '.jpeg', '.webp'}


# =============================================================================
# HELPERS
# =============================================================================

def _get_file_type(filename: str) -> str:
    """Determine file type from extension."""
    ext = os.path.splitext(filename)[1].lower()
    if ext == '.pdf':
        return 'pdf'
    if ext in ('.png', '.jpg', '.jpeg', '.webp'):
        return 'image'
    if ext == '.docx':
        return 'docx'
    if ext == '.txt':
        return 'txt'
    if ext == '.json':
        return 'json'
    return 'unknown'


def _convert_text_to_pdf(text: str) -> str:
    """Render text into a multi-page PDF. Returns path to temp PDF."""
    import fitz

    doc = fitz.open()
    page_width, page_height = 595, 842  # A4
    margin = 50
    font_size = 11
    line_height = font_size * 1.4
    max_chars = 80

    # Word wrap
    lines = []
    for paragraph in text.split('\n'):
        if not paragraph.strip():
            lines.append('')
            continue
        words = paragraph.split()
        current = ''
        for word in words:
            if len(current) + len(word) + 1 <= max_chars:
                current = current + ' ' + word if current else word
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)

    lines_per_page = int((page_height - 2 * margin) / line_height)
    page = None
    y = margin

    for i, line in enumerate(lines):
        if page is None or i % lines_per_page == 0:
            page = doc.new_page(width=page_width, height=page_height)
            y = margin
        page.insert_text((margin, y), line, fontsize=font_size, fontname="helv")
        y += line_height

    tmp = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
    doc.save(tmp.name)
    doc.close()
    return tmp.name


def _convert_docx_to_pdf(docx_path: str) -> str:
    """Convert DOCX to PDF. Returns path to temp PDF or None."""
    try:
        from docx import Document
        doc = Document(docx_path)
        text = "\n\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        if not text.strip():
            return None
        return _convert_text_to_pdf(text)
    except Exception as e:
        print(f"DOCX conversion failed: {e}")
        return None


def _convert_json_to_pdf(json_path: str) -> str:
    """Convert JSON file to a readable PDF. Returns path to temp PDF or None."""
    import json as json_lib
    try:
        with open(json_path, 'r', encoding='utf-8', errors='replace') as f:
            data = json_lib.load(f)
        text = json_lib.dumps(data, indent=2, ensure_ascii=False)
        if not text.strip():
            return None
        return _convert_text_to_pdf(text)
    except Exception as e:
        print(f"JSON conversion failed: {e}")
        return None


def _convert_txt_to_pdf(txt_path: str) -> str:
    """Convert TXT to PDF. Returns path to temp PDF or None."""
    try:
        with open(txt_path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        if not text.strip():
            return None
        return _convert_text_to_pdf(text)
    except Exception as e:
        print(f"TXT conversion failed: {e}")
        return None


def _prepare_for_indexing(file_path: str, file_type: str) -> str:
    """
    Convert file to PDF if needed for Byaldi indexing.
    Returns path to the file ready for indexing (PDF or image).
    Caller must clean up temp files.
    """
    if file_type == 'pdf' or file_type == 'image':
        return file_path  # Already indexable

    converters = {
        'docx': _convert_docx_to_pdf,
        'txt': _convert_txt_to_pdf,
        'json': _convert_json_to_pdf,
    }

    converter = converters.get(file_type)
    if converter:
        result = converter(file_path)
        if result:
            return result

    raise ValueError(f"Cannot convert {file_type} to indexable format")


def _rebuild_index_without(document_id: str):
    """
    Rebuild the Byaldi index excluding a specific document.
    This is needed because Byaldi doesn't support removing individual documents.
    """
    registry = get_document_registry()
    store = get_visual_store()

    remaining = registry.get_all_except(document_id)

    # Delete the current index
    store.delete_index(API_INDEX)

    if not remaining:
        return

    # Re-index remaining documents
    for i, record in enumerate(remaining):
        if not os.path.exists(record.file_path):
            print(f"Warning: file missing for {record.document_id}, skipping")
            continue

        indexable_path = _prepare_for_indexing(record.file_path, record.file_type)
        temp_file = indexable_path if indexable_path != record.file_path else None

        try:
            if i == 0:
                store.create_index(API_INDEX, indexable_path, record.document_name)
            else:
                store.add_to_index(API_INDEX, indexable_path, record.document_name)

            # Update the byaldi_doc_id since it changes after rebuild
            record.byaldi_doc_id = i
        finally:
            if temp_file:
                os.unlink(temp_file)

    # Save updated file metadata for Byaldi
    file_list = [
        {"name": r.document_name, "pages": r.chunk_count, "type": r.file_type}
        for r in remaining
    ]
    store.save_file_metadata(API_INDEX, file_list)


# =============================================================================
# ROUTES
# =============================================================================

@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    document_id: str = Form(None),
    document_name: str = Form(None),
):
    """
    Upload a document (PDF, TXT, Word, JSON, or image).

    The server splits the document into chunks (pages) and stores them,
    attaching metadata to each chunk: documentId, documentName, chunkIndex.

    - **file**: The document file to upload
    - **document_id**: Optional custom document ID (auto-generated if omitted)
    - **document_name**: Optional display name (defaults to filename)
    """
    # Validate file type
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    # Generate IDs
    doc_id = document_id or str(uuid.uuid4())
    doc_name = document_name or file.filename
    file_type = _get_file_type(file.filename)

    registry = get_document_registry()

    # Check for duplicate document_id
    if registry.get(doc_id):
        raise HTTPException(
            status_code=409,
            detail=f"Document with ID '{doc_id}' already exists. Delete it first or use a different ID."
        )

    # Save the uploaded file persistently
    file_dir = os.path.join(UPLOAD_STORE, doc_id)
    os.makedirs(file_dir, exist_ok=True)
    stored_path = os.path.join(file_dir, file.filename)

    with open(stored_path, 'wb') as f:
        content = await file.read()
        f.write(content)

    # Prepare file for indexing (convert to PDF if needed)
    temp_pdf = None
    try:
        indexable_path = _prepare_for_indexing(stored_path, file_type)
        temp_pdf = indexable_path if indexable_path != stored_path else None

        # Index with Byaldi
        store = get_visual_store()
        is_first = registry.is_empty()

        if is_first:
            chunk_count = store.create_index(API_INDEX, indexable_path, doc_name)
        else:
            # Ensure the existing index is loaded
            if store.index_exists_on_disk(API_INDEX):
                store.load_existing_index(API_INDEX)
            chunk_count = store.add_to_index(API_INDEX, indexable_path, doc_name)

        # Register the document
        byaldi_doc_id = registry.next_byaldi_doc_id()
        record = DocumentRecord(
            document_id=doc_id,
            document_name=doc_name,
            file_path=stored_path,
            chunk_count=chunk_count,
            byaldi_doc_id=byaldi_doc_id,
            file_type=file_type,
        )
        registry.add(record)

        # Update Byaldi file metadata
        file_list = [
            {"name": r.document_name, "pages": r.chunk_count, "type": r.file_type}
            for r in registry.list_all()
        ]
        store.save_file_metadata(API_INDEX, file_list)

        return UploadResponse(
            document_id=doc_id,
            document_name=doc_name,
            chunk_count=chunk_count,
            message=f"Document uploaded and split into {chunk_count} chunk(s)."
        )

    except Exception as e:
        # Cleanup on failure
        if os.path.exists(file_dir):
            shutil.rmtree(file_dir)
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

    finally:
        if temp_pdf:
            os.unlink(temp_pdf)


@router.get("/chunks", response_model=ChunksResponse)
async def get_chunks(
    query: str = Query(..., description="Search query to find relevant chunks"),
    document_id: str = Query(None, description="Filter by specific document ID"),
    top_k: int = Query(5, ge=1, le=50, description="Number of chunks to return"),
):
    """
    Retrieve document chunks based on a user query.

    Returns the most relevant chunks along with their metadata
    (documentId, documentName, chunkIndex) for search, RAG, or answering questions.

    - **query**: Natural language query to search for relevant content
    - **document_id**: Optional filter to search within a specific document
    - **top_k**: Maximum number of chunks to return (default: 5)
    """
    registry = get_document_registry()

    if registry.is_empty():
        return ChunksResponse(query=query, chunks=[], total_results=0)

    store = get_visual_store()

    # Ensure the index is loaded
    if not store.has_documents(API_INDEX):
        if store.index_exists_on_disk(API_INDEX):
            store.load_existing_index(API_INDEX)
        else:
            return ChunksResponse(query=query, chunks=[], total_results=0)

    # Determine document filter
    doc_filter = None
    if document_id:
        record = registry.get(document_id)
        if not record:
            raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")
        doc_filter = record.document_name

    # Search
    results = store.search(
        index_name=API_INDEX,
        query=query,
        top_k=top_k,
        document_filter=doc_filter,
    )

    # Map results to chunks with document metadata
    chunks = []
    for result in results.results:
        # Find the document record by name
        doc_record = None
        for rec in registry.list_all():
            if rec.document_name.lower() == result.document_name.lower():
                doc_record = rec
                break

        chunks.append(ChunkMetadata(
            document_id=doc_record.document_id if doc_record else "unknown",
            document_name=result.document_name,
            chunk_index=result.page_number,
            score=result.score,
            image_base64=result.image_base64,
        ))

    return ChunksResponse(
        query=query,
        chunks=chunks,
        total_results=len(chunks),
    )


@router.delete("/{document_id}", response_model=DeleteResponse)
async def delete_document(document_id: str):
    """
    Delete a document and all of its related chunks.

    Removes the stored content and its metadata (documentId, documentName, chunkIndex)
    so it will no longer appear in search or retrieval results.

    - **document_id**: The ID of the document to delete
    """
    registry = get_document_registry()
    record = registry.get(document_id)

    if not record:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")

    doc_name = record.document_name

    # Rebuild the Byaldi index without this document
    _rebuild_index_without(document_id)

    # Remove from registry
    registry.remove(document_id)

    # Delete the stored file
    file_dir = os.path.join(UPLOAD_STORE, document_id)
    if os.path.exists(file_dir):
        shutil.rmtree(file_dir)

    return DeleteResponse(
        document_id=document_id,
        message=f"Document '{doc_name}' and all its chunks have been deleted."
    )
