"""
Page Screenshotter Module
==========================
Convert PDF pages to high-resolution images using PyMuPDF.
Used as a utility for direct page image access when needed.
Byaldi handles its own page screenshots during indexing.
"""

import fitz  # PyMuPDF
from PIL import Image
from typing import List, Tuple

from config.settings import get_config

config = get_config()


def screenshot_pdf_pages(
    file_path: str,
    dpi: int = None
) -> List[Tuple[int, Image.Image]]:
    """
    Convert each PDF page to a PIL Image using PyMuPDF.

    Args:
        file_path: Path to the PDF file
        dpi: Resolution in DPI (defaults to config.visual_rag.image_dpi)

    Returns:
        List of (page_number, PIL.Image) tuples (1-indexed page numbers)
    """
    if dpi is None:
        dpi = config.visual_rag.image_dpi

    pages = []
    doc = fitz.open(file_path)

    for i in range(len(doc)):
        pix = doc[i].get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        pages.append((i + 1, img))

    doc.close()
    return pages


def screenshot_single_page(
    file_path: str,
    page_number: int,
    dpi: int = None
) -> Image.Image:
    """
    Convert a single PDF page to a PIL Image.

    Args:
        file_path: Path to the PDF file
        page_number: 1-indexed page number
        dpi: Resolution in DPI

    Returns:
        PIL Image of the page
    """
    if dpi is None:
        dpi = config.visual_rag.image_dpi

    doc = fitz.open(file_path)
    page_idx = page_number - 1

    if page_idx < 0 or page_idx >= len(doc):
        doc.close()
        raise ValueError(f"Page {page_number} out of range (document has {len(doc)} pages)")

    pix = doc[page_idx].get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()

    return img


def get_pdf_page_count(file_path: str) -> int:
    """Get the number of pages in a PDF file."""
    doc = fitz.open(file_path)
    count = len(doc)
    doc.close()
    return count
