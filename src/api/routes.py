"""
API Routes
===========
REST endpoints for document management.

POST   /api/documents/upload          - Upload and chunk a document
GET    /api/documents/chunks          - Retrieve relevant chunks by query
DELETE /api/documents/{document_id}   - Delete a document and its chunks
"""

import asyncio
import os
import shutil
import subprocess
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

# Chunked upload size (1 MB) - prevents loading entire file into memory
UPLOAD_CHUNK_SIZE = 1024 * 1024

# Supported file extensions
SUPPORTED_EXTENSIONS = {
    '.pdf', '.txt', '.docx', '.json',
    '.png', '.jpg', '.jpeg', '.webp',
    '.xlsx', '.xls',
}

# Lock to prevent concurrent Byaldi index modifications
_processing_lock = asyncio.Lock()


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
    if ext in ('.xlsx', '.xls'):
        return 'excel'
    if ext == '.txt':
        return 'txt'
    if ext == '.json':
        return 'json'
    return 'unknown'


async def _save_upload_chunked(upload_file: UploadFile, dest_path: str):
    """Stream uploaded file to disk in chunks instead of loading it all into memory."""
    with open(dest_path, 'wb') as f:
        while True:
            chunk = await upload_file.read(UPLOAD_CHUNK_SIZE)
            if not chunk:
                break
            f.write(chunk)


# =============================================================================
# IMAGE / CHART DETECTION
# =============================================================================

def _docx_has_visuals(docx_path: str) -> bool:
    """Check if a DOCX contains embedded images, drawings, or shapes."""
    try:
        from docx import Document
        doc = Document(docx_path)
        if doc.inline_shapes:
            return True
        for rel in doc.part.rels.values():
            if "image" in rel.reltype:
                return True
        return False
    except Exception:
        return True  # Assume visuals on error -> use LibreOffice


def _excel_has_visuals(excel_path: str) -> bool:
    """Check if an Excel file contains images or charts."""
    try:
        from openpyxl import load_workbook
        wb = load_workbook(excel_path, data_only=True)
        for ws in wb.worksheets:
            if ws._images:
                wb.close()
                return True
            if ws._charts:
                wb.close()
                return True
        wb.close()
        return False
    except Exception:
        return True  # Assume visuals on error -> use LibreOffice


# =============================================================================
# LIBREOFFICE RENDERER (fallback for files with images/charts)
# =============================================================================

def _render_with_libreoffice(file_path: str) -> str:
    """
    Render DOCX/Excel to a high-fidelity PDF using LibreOffice headless.

    Used ONLY when the file contains images, charts, or other visual elements
    that cannot be preserved via text extraction.

    Returns path to a temporary PDF (caller must clean up).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            "soffice",
            "--headless",
            "--norestore",
            "--convert-to", "pdf",
            "--outdir", tmpdir,
            file_path,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"LibreOffice conversion failed: {result.stderr or result.stdout}"
            )

        stem = os.path.splitext(os.path.basename(file_path))[0]
        pdf_path = os.path.join(tmpdir, f"{stem}.pdf")
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(
                "LibreOffice did not produce PDF output"
            )

        # Move to a persistent temp file (tmpdir gets deleted on exit)
        final = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
        final.close()
        shutil.move(pdf_path, final.name)
        return final.name


# =============================================================================
# TEXT-BASED EXTRACTORS (fast path for data-only files)
# =============================================================================

def _convert_text_to_pdf(text: str) -> str:
    """Render plain text into a multi-page PDF. Returns path to temp PDF."""
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


def _extract_docx_to_pdf(docx_path: str) -> str:
    """
    Extract text + tables from DOCX in document order and render to PDF.

    Iterates body elements in order so paragraphs and tables appear
    in the same sequence as the original document.
    Returns path to temp PDF, or None on failure.
    """
    try:
        from docx import Document
        from docx.table import Table as DocxTable
        from docx.text.paragraph import Paragraph
        from docx.oxml.ns import qn

        doc = Document(docx_path)
        parts = []

        for child in doc.element.body:
            if child.tag == qn('w:p'):
                para = Paragraph(child, doc)
                text = para.text.strip()
                if text:
                    parts.append(text)

            elif child.tag == qn('w:tbl'):
                table = DocxTable(child, doc)
                rows = []
                for row in table.rows:
                    cells = [cell.text.strip() for cell in row.cells]
                    rows.append(cells)

                if rows:
                    # Calculate column widths for alignment
                    col_count = max(len(r) for r in rows)
                    col_widths = [0] * col_count
                    for row in rows:
                        for j, cell in enumerate(row):
                            if j < col_count:
                                col_widths[j] = max(col_widths[j], len(cell))

                    # Cap column widths so tables fit in 80 chars
                    total = sum(col_widths) + (col_count * 3)
                    if total > 76:
                        scale = 76 / total
                        col_widths = [max(3, int(w * scale)) for w in col_widths]

                    # Render table rows
                    for i, row in enumerate(rows):
                        cells = []
                        for j, cell in enumerate(row):
                            w = col_widths[j] if j < len(col_widths) else 10
                            cells.append(cell[:w].ljust(w))
                        parts.append(" | ".join(cells))

                        # Add separator after first row (header)
                        if i == 0:
                            sep = ["-" * (col_widths[j] if j < len(col_widths) else 10)
                                   for j in range(len(cells))]
                            parts.append("-+-".join(sep))

                    parts.append("")  # blank line after table

        text = "\n".join(parts)
        if not text.strip():
            return None
        return _convert_text_to_pdf(text)

    except Exception as e:
        print(f"DOCX text extraction failed: {e}")
        return None


def _extract_excel_to_pdf(excel_path: str) -> str:
    """
    Parse Excel cell data and render as clean text tables.

    Each sheet becomes a section. Rows are formatted as pipe-separated tables
    with aligned columns. Empty rows/columns are skipped.
    Returns path to temp PDF, or None on failure.
    """
    try:
        from openpyxl import load_workbook

        wb = load_workbook(excel_path, data_only=True)
        parts = []

        for ws in wb.worksheets:
            # Collect non-empty rows
            rows = []
            for row in ws.iter_rows(values_only=True):
                cells = [str(c) if c is not None else "" for c in row]
                if any(c.strip() for c in cells):
                    rows.append(cells)

            if not rows:
                continue

            # Sheet header
            parts.append(f"Sheet: {ws.title}")
            parts.append("=" * min(len(f"Sheet: {ws.title}"), 40))
            parts.append("")

            # Normalize column count
            col_count = max(len(r) for r in rows)
            for r in rows:
                while len(r) < col_count:
                    r.append("")

            # Calculate column widths
            col_widths = [0] * col_count
            for row in rows:
                for j, cell in enumerate(row):
                    col_widths[j] = max(col_widths[j], len(cell))

            # Cap widths so rows fit reasonably (max 76 chars content)
            total = sum(col_widths) + (col_count * 3)
            if total > 76:
                scale = 76 / total
                col_widths = [max(3, int(w * scale)) for w in col_widths]

            # Render rows
            for i, row in enumerate(rows):
                cells = []
                for j, cell in enumerate(row):
                    w = col_widths[j] if j < len(col_widths) else 10
                    cells.append(cell[:w].ljust(w))
                parts.append(" | ".join(cells))

                # Separator after first row (header)
                if i == 0:
                    sep = ["-" * (col_widths[j] if j < len(col_widths) else 10)
                           for j in range(len(cells))]
                    parts.append("-+-".join(sep))

            parts.append("")  # blank line between sheets

        wb.close()

        text = "\n".join(parts)
        if not text.strip():
            return None
        return _convert_text_to_pdf(text)

    except Exception as e:
        print(f"Excel text extraction failed: {e}")
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


# =============================================================================
# INDEXING PREPARATION (hybrid routing)
# =============================================================================

def _prepare_for_indexing(file_path: str, file_type: str) -> str:
    """
    Prepare a file for Byaldi indexing using the best strategy:

    - PDF / image: used directly
    - DOCX: text+table extraction (fast) unless it has images -> LibreOffice
    - Excel: cell data extraction (fast) unless it has charts/images -> LibreOffice
    - TXT / JSON: text layout to PDF

    Returns path to the file ready for indexing.
    Caller must clean up temp files (when returned path != input path).
    """
    if file_type in ('pdf', 'image'):
        return file_path

    # --- DOCX: extract text+tables, fall back to LibreOffice if images ---
    if file_type == 'docx':
        if _docx_has_visuals(file_path):
            print(f"DOCX has images/shapes -> using LibreOffice rendering")
            return _render_with_libreoffice(file_path)
        result = _extract_docx_to_pdf(file_path)
        if result:
            print(f"DOCX text+tables extracted -> fast path")
            return result
        # Extraction failed, fall back to LibreOffice
        print(f"DOCX extraction failed -> falling back to LibreOffice")
        return _render_with_libreoffice(file_path)

    # --- Excel: parse cell data, fall back to LibreOffice if charts/images ---
    if file_type == 'excel':
        if _excel_has_visuals(file_path):
            print(f"Excel has images/charts -> using LibreOffice rendering")
            return _render_with_libreoffice(file_path)
        result = _extract_excel_to_pdf(file_path)
        if result:
            print(f"Excel cell data extracted -> fast path")
            return result
        # Extraction failed, fall back to LibreOffice
        print(f"Excel extraction failed -> falling back to LibreOffice")
        return _render_with_libreoffice(file_path)

    # --- Plain text formats ---
    converters = {
        'txt': _convert_txt_to_pdf,
        'json': _convert_json_to_pdf,
    }
    converter = converters.get(file_type)
    if converter:
        result = converter(file_path)
        if result:
            return result

    raise ValueError(f"Cannot convert {file_type} to indexable format")


def _index_document(stored_path: str, file_type: str, doc_id: str, doc_name: str) -> int:
    """
    Synchronous document processing: convert + index with Byaldi.
    Runs inside asyncio.to_thread() to avoid blocking the event loop.
    """
    registry = get_document_registry()
    store = get_visual_store()

    indexable_path = _prepare_for_indexing(stored_path, file_type)
    temp_file = indexable_path if indexable_path != stored_path else None

    try:
        is_first = registry.is_empty()

        if is_first:
            chunk_count = store.create_index(API_INDEX, indexable_path, doc_name)
        else:
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

        return chunk_count

    finally:
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)


def _search_chunks(query: str, top_k: int, doc_filter: str = None):
    """
    Synchronous Byaldi search.
    Runs inside asyncio.to_thread() to avoid blocking the event loop.
    """
    store = get_visual_store()

    if not store.has_documents(API_INDEX):
        if store.index_exists_on_disk(API_INDEX):
            store.load_existing_index(API_INDEX)
        else:
            return None

    return store.search(
        index_name=API_INDEX,
        query=query,
        top_k=top_k,
        document_filter=doc_filter,
    )


def _delete_and_rebuild(document_id: str):
    """
    Synchronous index rebuild + file cleanup.
    Runs inside asyncio.to_thread() to avoid blocking the event loop.
    """
    _rebuild_index_without(document_id)


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
            if temp_file and os.path.exists(temp_file):
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
    Upload a document (PDF, Word, Excel, TXT, JSON, or image).

    The server splits the document into chunks (pages/sheets) and indexes them
    for visual similarity search, preserving all original formatting.

    - **file**: The document file to upload
    - **document_id**: Optional custom document ID (auto-generated if omitted)
    - **document_name**: Optional display name (defaults to filename)
    """
    # Validate file type (fast, stays on event loop)
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    doc_id = document_id or str(uuid.uuid4())
    doc_name = document_name or file.filename
    file_type = _get_file_type(file.filename)

    registry = get_document_registry()

    if registry.get(doc_id):
        raise HTTPException(
            status_code=409,
            detail=f"Document with ID '{doc_id}' already exists. Delete it first or use a different ID."
        )

    # Save the uploaded file in chunks (no full-file memory spike)
    file_dir = os.path.join(UPLOAD_STORE, doc_id)
    os.makedirs(file_dir, exist_ok=True)
    stored_path = os.path.join(file_dir, file.filename)

    try:
        await _save_upload_chunked(file, stored_path)

        # Run all blocking work (conversion + GPU indexing) in a thread
        # Lock ensures only one index modification at a time
        async with _processing_lock:
            chunk_count = await asyncio.to_thread(
                _index_document, stored_path, file_type, doc_id, doc_name
            )

        return UploadResponse(
            document_id=doc_id,
            document_name=doc_name,
            chunk_count=chunk_count,
            message=f"Document uploaded and split into {chunk_count} chunk(s)."
        )

    except HTTPException:
        raise
    except Exception as e:
        # Cleanup on failure
        if os.path.exists(file_dir):
            shutil.rmtree(file_dir)
        registry.remove(doc_id)
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


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

    # Determine document filter
    doc_filter = None
    if document_id:
        record = registry.get(document_id)
        if not record:
            raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")
        doc_filter = record.document_name

    # Run GPU search in a thread to avoid blocking event loop
    results = await asyncio.to_thread(_search_chunks, query, top_k, doc_filter)

    if results is None:
        return ChunksResponse(query=query, chunks=[], total_results=0)

    # Map results to chunks with document metadata
    chunks = []
    for result in results.results:
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

    # Run blocking rebuild + cleanup in a thread with lock
    async with _processing_lock:
        await asyncio.to_thread(_delete_and_rebuild, document_id)

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
