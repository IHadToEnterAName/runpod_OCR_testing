"""
Text Extractor Module
=====================
Extract text from PDF pages using PyMuPDF (fitz).
Falls back to pytesseract OCR for scanned pages with minimal text.
Used ONLY for long-document summarization path.
"""

import fitz  # PyMuPDF
from typing import List, Tuple

from config.settings import get_config

config = get_config()


def extract_text_from_pdf(
    pdf_path: str,
    min_chars_per_page: int = None,
    enable_ocr_fallback: bool = None,
) -> List[Tuple[int, str]]:
    """
    Extract text from all pages of a PDF.

    Uses PyMuPDF's page.get_text() as the primary method.
    For pages where PyMuPDF returns very little text (scanned PDFs),
    falls back to pytesseract OCR if enabled.

    Args:
        pdf_path: Path to the PDF file.
        min_chars_per_page: Minimum character count to consider extraction
            successful. Defaults to config.long_doc.min_chars_per_page.
        enable_ocr_fallback: Whether to use pytesseract for scanned pages.
            Defaults to config.long_doc.enable_ocr_fallback.

    Returns:
        List of (page_number, text) tuples. Page numbers are 1-indexed.
    """
    if min_chars_per_page is None:
        min_chars_per_page = config.long_doc.min_chars_per_page
    if enable_ocr_fallback is None:
        enable_ocr_fallback = config.long_doc.enable_ocr_fallback

    doc = fitz.open(pdf_path)
    pages: List[Tuple[int, str]] = []
    ocr_needed: List[int] = []  # 0-indexed pages needing OCR

    for i in range(len(doc)):
        text = doc[i].get_text().strip()
        page_num = i + 1

        if len(text) >= min_chars_per_page:
            pages.append((page_num, text))
        else:
            ocr_needed.append(i)
            pages.append((page_num, text))

    doc.close()

    # OCR fallback for scanned pages
    if ocr_needed and enable_ocr_fallback:
        pages = _ocr_fallback(pdf_path, pages, ocr_needed, min_chars_per_page)

    total = len(pages)
    non_empty = sum(1 for _, t in pages if len(t) >= min_chars_per_page)
    print(f"Text extraction: {non_empty}/{total} pages have text, "
          f"{len(ocr_needed)} needed OCR fallback")

    return pages


def _ocr_fallback(
    pdf_path: str,
    pages: List[Tuple[int, str]],
    ocr_page_indices: List[int],
    min_chars: int,
) -> List[Tuple[int, str]]:
    """
    Apply pytesseract OCR to pages where PyMuPDF extraction yielded
    insufficient text.
    """
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        print("pytesseract not installed. Skipping OCR fallback. "
              "Install with: pip install pytesseract")
        return pages

    doc = fitz.open(pdf_path)

    for page_idx in ocr_page_indices:
        try:
            # Render page at 300 DPI for better OCR accuracy
            pix = doc[page_idx].get_pixmap(dpi=300)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

            ocr_text = pytesseract.image_to_string(img).strip()

            if len(ocr_text) >= min_chars:
                page_num = page_idx + 1
                for j, (pn, _) in enumerate(pages):
                    if pn == page_num:
                        pages[j] = (page_num, ocr_text)
                        break
                print(f"  OCR extracted {len(ocr_text)} chars from page {page_num}")
        except Exception as e:
            print(f"  OCR failed for page {page_idx + 1}: {e}")

    doc.close()
    return pages


def chunk_pages_by_tokens(
    pages: List[Tuple[int, str]],
    chunk_token_limit: int = None,
) -> List[dict]:
    """
    Group extracted page texts into chunks that fit within a token budget.
    Respects page boundaries -- never splits a page across chunks.

    Args:
        pages: List of (page_number, text) tuples from extract_text_from_pdf.
        chunk_token_limit: Max tokens per chunk.
            Defaults to config.long_doc.chunk_token_limit.

    Returns:
        List of chunk dicts with keys:
        text, page_start, page_end, page_count, token_count
    """
    from utils.helpers import count_tokens

    if chunk_token_limit is None:
        chunk_token_limit = config.long_doc.chunk_token_limit

    chunks = []
    current_texts = []
    current_token_count = 0
    current_page_start = None
    current_page_end = None
    current_page_count = 0

    for page_num, text in pages:
        if not text.strip():
            continue

        page_tokens = count_tokens(text)

        # Single page exceeds limit — becomes its own chunk
        if page_tokens > chunk_token_limit:
            if current_texts:
                chunks.append({
                    "text": "\n\n".join(current_texts),
                    "page_start": current_page_start,
                    "page_end": current_page_end,
                    "page_count": current_page_count,
                    "token_count": current_token_count,
                })
                current_texts = []
                current_token_count = 0
                current_page_start = None
                current_page_count = 0

            chunks.append({
                "text": f"[Page {page_num}]\n{text}",
                "page_start": page_num,
                "page_end": page_num,
                "page_count": 1,
                "token_count": page_tokens,
            })
            continue

        # Would adding this page exceed the limit?
        if current_token_count + page_tokens > chunk_token_limit and current_texts:
            chunks.append({
                "text": "\n\n".join(current_texts),
                "page_start": current_page_start,
                "page_end": current_page_end,
                "page_count": current_page_count,
                "token_count": current_token_count,
            })
            current_texts = []
            current_token_count = 0
            current_page_start = None
            current_page_count = 0

        current_texts.append(f"[Page {page_num}]\n{text}")
        current_token_count += page_tokens
        if current_page_start is None:
            current_page_start = page_num
        current_page_end = page_num
        current_page_count += 1

    # Flush final chunk
    if current_texts:
        chunks.append({
            "text": "\n\n".join(current_texts),
            "page_start": current_page_start,
            "page_end": current_page_end,
            "page_count": current_page_count,
            "token_count": current_token_count,
        })

    print(f"Chunking: {len(chunks)} chunks from "
          f"{sum(c['page_count'] for c in chunks)} pages")
    return chunks
