"""
File Processor Module
======================
Processes uploaded files into Byaldi visual indexes.
Replaces text extraction + chunking + embedding with visual indexing.
"""

import os
import shutil
import tempfile
from typing import List, Dict

import chainlit as cl

from config.settings import get_config
from storage.visual_store import get_visual_store
from processing.page_screenshotter import get_pdf_page_count

config = get_config()

# Persistent storage for Chainlit-uploaded files (needed for /delete rebuild)
CHAINLIT_UPLOAD_STORE = os.path.join(config.byaldi.index_path, "chainlit_uploads")
os.makedirs(CHAINLIT_UPLOAD_STORE, exist_ok=True)

# =============================================================================
# FILE PROCESSING
# =============================================================================

async def process_files(
    files,
    index_name: str,
    file_list: List[Dict],
    is_first_upload: bool = True
):
    """
    Process uploaded files into a Byaldi visual index.

    For PDFs: Byaldi handles page screenshots and ColQwen2 embedding internally.
    For images: Indexed as single-page documents.

    If a file with the same name already exists, the old version is replaced
    (requires a full index rebuild since Byaldi can't remove individual docs).

    Args:
        files: Chainlit uploaded file objects
        index_name: Byaldi index name (session-unique)
        file_list: List to track uploaded files
        is_first_upload: True if this is the first file for this index
    """
    store = get_visual_store()
    total_pages = 0

    progress = cl.Message(content="Processing documents...")
    await progress.send()

    # --- Duplicate detection: replace old versions of same-name files ---
    incoming_names_lower = {f.name.lower() for f in files}
    duplicates = [f for f in file_list if f["name"].lower() in incoming_names_lower]

    if duplicates:
        dup_names = [d["name"] for d in duplicates]
        print(f"Duplicate upload detected: {dup_names} — rebuilding index")
        progress.content = f"Replacing: {', '.join(dup_names)} — rebuilding index..."
        await progress.update()

        # Remove old entries from file_list
        dup_names_lower = {d["name"].lower() for d in duplicates}
        remaining = [f for f in file_list if f["name"].lower() not in dup_names_lower]
        file_list.clear()
        file_list.extend(remaining)

        # Delete old stored copies
        for d in duplicates:
            old_path = os.path.join(CHAINLIT_UPLOAD_STORE, d["name"])
            if os.path.exists(old_path):
                os.unlink(old_path)

        # Delete index and rebuild from remaining stored files
        store.delete_index(index_name)

        rebuilt = []
        for idx, f_meta in enumerate(remaining):
            file_path = f_meta.get("path", "")
            if not file_path or not os.path.exists(file_path):
                alt = os.path.join(CHAINLIT_UPLOAD_STORE, f_meta["name"])
                if os.path.exists(alt):
                    file_path = alt
            if not file_path or not os.path.exists(file_path):
                print(f"Cannot re-index '{f_meta['name']}': file not found, skipping")
                continue
            try:
                progress.content = f"Re-indexing ({idx + 1}/{len(remaining)}): {f_meta['name']}..."
                await progress.update()
                if not rebuilt:
                    store.create_index(index_name, file_path, f_meta["name"])
                else:
                    store.add_to_index(index_name, file_path, f_meta["name"])
                rebuilt.append(f_meta)
            except Exception as e:
                print(f"Failed to re-index {f_meta['name']}: {e}")

        # Update file_list to only successfully rebuilt files
        file_list.clear()
        file_list.extend(rebuilt)

        # After rebuild, new files will be added on top
        is_first_upload = not rebuilt

    # --- Process each incoming file ---
    for i, file in enumerate(files):
        fname = file.name
        fpath = file.path

        progress.content = f"Processing ({i + 1}/{len(files)}): {fname}"
        await progress.update()

        try:
            pages = _index_single_file(
                store, index_name, fname, fpath,
                is_first=(is_first_upload and i == 0 and not file_list),
            )
            if pages is None:
                await cl.Message(content=f"Unsupported or failed: {fname}").send()
                continue

            # Save persistent copy (overwrite if exists)
            stored_path = os.path.join(CHAINLIT_UPLOAD_STORE, fname)
            shutil.copy2(fpath, stored_path)

            total_pages += pages
            file_list.append({
                "name": fname,
                "pages": pages,
                "type": _detect_file_type(fname),
                "path": stored_path,
            })
            print(f"Indexed {fname} ({pages} pages)")

        except Exception as e:
            print(f"Error processing {fname}: {e}")
            import traceback
            traceback.print_exc()
            await cl.Message(content=f"Error processing {fname}: {str(e)}").send()

    # Persist file metadata alongside the index
    if total_pages > 0:
        store.save_file_metadata(index_name, file_list)

        file_names = ", ".join([f["name"] for f in file_list])
        await cl.Message(
            content=f"Indexed {total_pages} pages from {len(file_list)} file(s): {file_names}\n\n"
                    f"You can now ask questions about your documents."
        ).send()

    return total_pages


def _detect_file_type(fname: str) -> str:
    """Get file type string from filename."""
    lower = fname.lower()
    if lower.endswith('.pdf'):
        return 'pdf'
    if lower.endswith(('.png', '.jpg', '.jpeg', '.webp')):
        return 'image'
    if lower.endswith('.docx'):
        return 'docx'
    if lower.endswith('.txt'):
        return 'txt'
    return 'unknown'


def _index_single_file(store, index_name: str, fname: str, fpath: str,
                        is_first: bool) -> int:
    """
    Index a single file into Byaldi. Handles format conversion.

    Returns page count, or None if unsupported.
    """
    if fname.lower().endswith('.pdf'):
        page_count = get_pdf_page_count(fpath)
        if is_first:
            return store.create_index(index_name, fpath, fname)
        return store.add_to_index(index_name, fpath, fname)

    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        if is_first:
            return store.create_index(index_name, fpath, fname)
        return store.add_to_index(index_name, fpath, fname)

    if fname.lower().endswith('.docx'):
        pdf_path = _convert_docx_to_pdf(fpath)
        if not pdf_path:
            return None
        try:
            if is_first:
                return store.create_index(index_name, pdf_path, fname)
            return store.add_to_index(index_name, pdf_path, fname)
        finally:
            os.unlink(pdf_path)

    if fname.lower().endswith('.txt'):
        pdf_path = _convert_txt_to_pdf(fpath)
        if not pdf_path:
            return None
        try:
            if is_first:
                return store.create_index(index_name, pdf_path, fname)
            return store.add_to_index(index_name, pdf_path, fname)
        finally:
            os.unlink(pdf_path)

    return None


# =============================================================================
# FORMAT CONVERTERS
# =============================================================================

def _convert_docx_to_pdf(docx_path: str) -> str:
    """
    Convert DOCX to PDF using PyMuPDF text rendering.
    Returns path to temporary PDF file, or None on failure.
    """
    try:
        from docx import Document

        doc = Document(docx_path)
        full_text = "\n\n".join([p.text for p in doc.paragraphs if p.text.strip()])

        if not full_text.strip():
            return None

        return _text_to_pdf(full_text)

    except Exception as e:
        print(f"DOCX to PDF conversion failed: {e}")
        return None


def _convert_txt_to_pdf(txt_path: str) -> str:
    """
    Convert TXT to PDF for visual indexing.
    Returns path to temporary PDF file, or None on failure.
    """
    try:
        with open(txt_path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()

        if not text.strip():
            return None

        return _text_to_pdf(text)

    except Exception as e:
        print(f"TXT to PDF conversion failed: {e}")
        return None


def _text_to_pdf(text: str) -> str:
    """
    Render text content into a multi-page PDF using PyMuPDF.
    Returns path to the temporary PDF file.
    """
    import fitz

    doc = fitz.open()

    # A4 dimensions
    page_width = 595
    page_height = 842
    margin = 50
    font_size = 11
    line_height = font_size * 1.4

    # Word wrap
    max_chars_per_line = 80
    lines = []
    for paragraph in text.split('\n'):
        if not paragraph.strip():
            lines.append('')
            continue
        words = paragraph.split()
        current_line = ''
        for word in words:
            if len(current_line) + len(word) + 1 <= max_chars_per_line:
                current_line = current_line + ' ' + word if current_line else word
            else:
                lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)

    # Render across pages
    lines_per_page = int((page_height - 2 * margin) / line_height)
    page = None
    y_pos = margin

    for i, line in enumerate(lines):
        if page is None or i % lines_per_page == 0:
            page = doc.new_page(width=page_width, height=page_height)
            y_pos = margin

        page.insert_text(
            (margin, y_pos),
            line,
            fontsize=font_size,
            fontname="helv"
        )
        y_pos += line_height

    tmp = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
    doc.save(tmp.name)
    doc.close()

    return tmp.name
