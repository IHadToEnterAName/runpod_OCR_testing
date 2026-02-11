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

    for i, file in enumerate(files):
        fname = file.name
        fpath = file.path

        progress.content = f"Processing ({i + 1}/{len(files)}): {fname}"
        await progress.update()

        try:
            # Persistent PDF storage dir (for text extraction in long-doc summarization)
            persistent_pdf_dir = os.path.join(config.byaldi.index_path, index_name, "pdfs")
            os.makedirs(persistent_pdf_dir, exist_ok=True)

            if fname.lower().endswith('.pdf'):
                # Get page count for progress reporting
                page_count = get_pdf_page_count(fpath)
                progress.content = f"Indexing {fname} ({page_count} pages)..."
                await progress.update()

                # Copy PDF to persistent storage for text extraction later
                persistent_pdf_path = os.path.join(persistent_pdf_dir, fname)
                if not os.path.exists(persistent_pdf_path):
                    shutil.copy2(fpath, persistent_pdf_path)

                # Index the PDF - Byaldi handles page screenshots internally
                if is_first_upload and i == 0:
                    pages = store.create_index(index_name, fpath, fname)
                else:
                    pages = store.add_to_index(index_name, fpath, fname)

                total_pages += pages
                file_list.append({
                    "name": fname,
                    "pages": pages,
                    "type": "pdf",
                    "path": persistent_pdf_path
                })
                print(f"Indexed PDF: {fname} ({pages} pages)")

            elif fname.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                # Index single image
                if is_first_upload and i == 0:
                    pages = store.create_index(index_name, fpath, fname)
                else:
                    pages = store.add_to_index(index_name, fpath, fname)

                total_pages += pages
                file_list.append({
                    "name": fname,
                    "pages": 1,
                    "type": "image"
                })
                print(f"Indexed image: {fname}")

            elif fname.lower().endswith('.docx'):
                # Convert DOCX to PDF first, then index
                pdf_path = _convert_docx_to_pdf(fpath)
                if pdf_path:
                    if is_first_upload and i == 0:
                        pages = store.create_index(index_name, pdf_path, fname)
                    else:
                        pages = store.add_to_index(index_name, pdf_path, fname)

                    # Copy converted PDF to persistent storage before unlinking
                    persistent_pdf_path = os.path.join(
                        persistent_pdf_dir, fname.rsplit('.', 1)[0] + '.pdf'
                    )
                    shutil.copy2(pdf_path, persistent_pdf_path)

                    total_pages += pages
                    file_list.append({
                        "name": fname,
                        "pages": pages,
                        "type": "docx",
                        "path": persistent_pdf_path
                    })
                    print(f"Indexed DOCX (via PDF): {fname} ({pages} pages)")
                    os.unlink(pdf_path)
                else:
                    await cl.Message(
                        content=f"Could not convert {fname} to PDF for visual indexing."
                    ).send()

            elif fname.lower().endswith('.txt'):
                # Convert text to PDF for visual indexing
                pdf_path = _convert_txt_to_pdf(fpath)
                if pdf_path:
                    if is_first_upload and i == 0:
                        pages = store.create_index(index_name, pdf_path, fname)
                    else:
                        pages = store.add_to_index(index_name, pdf_path, fname)

                    # Copy converted PDF to persistent storage before unlinking
                    persistent_pdf_path = os.path.join(
                        persistent_pdf_dir, fname.rsplit('.', 1)[0] + '.pdf'
                    )
                    shutil.copy2(pdf_path, persistent_pdf_path)

                    total_pages += pages
                    file_list.append({
                        "name": fname,
                        "pages": pages,
                        "type": "txt",
                        "path": persistent_pdf_path
                    })
                    print(f"Indexed TXT (via PDF): {fname} ({pages} pages)")
                    os.unlink(pdf_path)
            else:
                await cl.Message(
                    content=f"Unsupported file type: {fname}"
                ).send()

        except Exception as e:
            print(f"Error processing {fname}: {e}")
            import traceback
            traceback.print_exc()
            await cl.Message(
                content=f"Error processing {fname}: {str(e)}"
            ).send()

    # Persist file metadata alongside the index
    if total_pages > 0:
        store.save_file_metadata(index_name, file_list)

        file_names = ", ".join([f["name"] for f in file_list])
        await cl.Message(
            content=f"Indexed {total_pages} pages from {len(file_list)} file(s): {file_names}\n\n"
                    f"You can now ask questions about your documents."
        ).send()

    return total_pages


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
