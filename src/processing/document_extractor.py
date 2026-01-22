"""
Document Processing Module
===========================
Enhanced document extraction with Structural Layout Awareness.
Detects columns, tables, headers, and maintains reading order.
"""

import fitz  # PyMuPDF
import docx
from PIL import Image
import io
import re
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field

# =============================================================================
# LAYOUT DATA STRUCTURES
# =============================================================================

@dataclass
class TextBlock:
    """Represents a distinct text block with position metadata."""
    text: str
    x0: float  # Left coordinate
    y0: float  # Top coordinate
    x1: float  # Right coordinate
    y1: float  # Bottom coordinate
    block_type: str = "text"  # text, header, footer, table, caption

@dataclass
class TableData:
    """Represents a detected table with cells."""
    rows: List[List[str]]
    x0: float
    y0: float
    x1: float
    y1: float

@dataclass
class PageLayout:
    """Complete layout description for a page."""
    page_number: int
    layout_type: str  # single-column, multi-column, table-heavy, form, mixed
    num_columns: int
    headers: List[str]
    footers: List[str]
    text_blocks: List[TextBlock]
    tables: List[TableData]
    reading_order: List[int]  # Indices into text_blocks for reading order
    raw_text: str
    structured_text: str

# =============================================================================
# LAYOUT DETECTION FUNCTIONS
# =============================================================================

def _detect_columns(blocks: List[Dict], page_width: float) -> int:
    """
    Detect number of columns by analyzing x-coordinate clustering.
    """
    if not blocks:
        return 1

    # Get left edges of all blocks
    x_coords = [b.get('bbox', [0])[0] for b in blocks if b.get('type') == 0]

    if len(x_coords) < 3:
        return 1

    # Simple column detection: check if there are distinct x-clusters
    x_coords.sort()
    gaps = []
    for i in range(1, len(x_coords)):
        gap = x_coords[i] - x_coords[i-1]
        if gap > page_width * 0.2:  # 20% of page width = significant gap
            gaps.append(gap)

    # If we have significant gaps, we likely have multiple columns
    if len(gaps) >= len(x_coords) // 4:
        return 2 if len(gaps) < len(x_coords) // 2 else 3

    return 1

def _identify_headers_footers(
    blocks: List[Dict],
    page_height: float,
    threshold: float = 0.1
) -> Tuple[List[str], List[str]]:
    """
    Identify header and footer text blocks based on position.
    """
    headers = []
    footers = []

    header_zone = page_height * threshold
    footer_zone = page_height * (1 - threshold)

    for block in blocks:
        if block.get('type') != 0:  # Only text blocks
            continue

        bbox = block.get('bbox', [0, 0, 0, 0])
        y_top = bbox[1]
        y_bottom = bbox[3]
        text = ""

        # Extract text from lines and spans
        for line in block.get('lines', []):
            for span in line.get('spans', []):
                text += span.get('text', '') + " "

        text = text.strip()
        if not text:
            continue

        if y_top < header_zone:
            headers.append(text)
        elif y_bottom > footer_zone:
            footers.append(text)

    return headers, footers

def _detect_tables(page: fitz.Page) -> List[TableData]:
    """
    Detect and extract tables from a PDF page.
    Uses PyMuPDF's table detection when available.
    """
    tables = []

    try:
        # Try to use PyMuPDF's built-in table finder (v1.23+)
        found_tables = page.find_tables()
        for table in found_tables:
            rows = []
            for row in table.extract():
                cleaned_row = [cell if cell else "" for cell in row]
                rows.append(cleaned_row)

            if rows:
                bbox = table.bbox
                tables.append(TableData(
                    rows=rows,
                    x0=bbox[0],
                    y0=bbox[1],
                    x1=bbox[2],
                    y1=bbox[3]
                ))
    except AttributeError:
        # Fallback: Older PyMuPDF without find_tables
        pass
    except Exception as e:
        print(f"⚠️ Table detection error: {e}")

    return tables

def _sort_blocks_reading_order(
    blocks: List[TextBlock],
    num_columns: int
) -> List[int]:
    """
    Sort text blocks into logical reading order.
    For multi-column layouts, read top-to-bottom within each column.
    """
    if not blocks:
        return []

    if num_columns == 1:
        # Simple: sort by y-coordinate (top to bottom)
        indexed = [(i, b) for i, b in enumerate(blocks)]
        indexed.sort(key=lambda x: (x[1].y0, x[1].x0))
        return [i for i, _ in indexed]

    # Multi-column: group by x-coordinate, then sort within each group
    page_width = max(b.x1 for b in blocks) - min(b.x0 for b in blocks)
    column_width = page_width / num_columns

    columns = [[] for _ in range(num_columns)]

    for i, block in enumerate(blocks):
        # Determine which column this block belongs to
        col_idx = int((block.x0 - min(b.x0 for b in blocks)) / column_width)
        col_idx = min(col_idx, num_columns - 1)
        columns[col_idx].append((i, block))

    # Sort each column by y-coordinate
    reading_order = []
    for col in columns:
        col.sort(key=lambda x: x[1].y0)
        reading_order.extend([i for i, _ in col])

    return reading_order

def _format_table_as_text(table: TableData) -> str:
    """
    Format a table as readable text with row/column structure.
    """
    if not table.rows:
        return ""

    lines = ["[TABLE START]"]

    for i, row in enumerate(table.rows):
        if i == 0:
            # Assume first row is header
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
            lines.append("|" + "|".join(["---"] * len(row)) + "|")
        else:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")

    lines.append("[TABLE END]")
    return "\n".join(lines)

# =============================================================================
# ENHANCED PDF EXTRACTION WITH LAYOUT AWARENESS
# =============================================================================

def analyze_page_layout(page: fitz.Page, page_num: int) -> PageLayout:
    """
    Analyze a PDF page's structural layout before extracting text.
    """
    page_width = page.rect.width
    page_height = page.rect.height

    # Get raw block data from PyMuPDF
    blocks_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
    blocks_raw = blocks_dict.get("blocks", [])

    # Detect layout characteristics
    num_columns = _detect_columns(blocks_raw, page_width)
    headers, footers = _identify_headers_footers(blocks_raw, page_height)
    tables = _detect_tables(page)

    # Convert raw blocks to TextBlock objects
    text_blocks = []
    for block in blocks_raw:
        if block.get('type') != 0:  # Skip image blocks
            continue

        bbox = block.get('bbox', [0, 0, 0, 0])
        text = ""

        for line in block.get('lines', []):
            for span in line.get('spans', []):
                text += span.get('text', '')
            text += "\n"

        text = text.strip()
        if text:
            # Determine block type
            block_type = "text"
            if text in headers:
                block_type = "header"
            elif text in footers:
                block_type = "footer"

            text_blocks.append(TextBlock(
                text=text,
                x0=bbox[0],
                y0=bbox[1],
                x1=bbox[2],
                y1=bbox[3],
                block_type=block_type
            ))

    # Determine reading order
    reading_order = _sort_blocks_reading_order(text_blocks, num_columns)

    # Determine layout type
    has_tables = len(tables) > 0
    if has_tables and len(tables) > len(text_blocks) // 2:
        layout_type = "table-heavy"
    elif num_columns > 1:
        layout_type = "multi-column"
    elif has_tables:
        layout_type = "mixed"
    else:
        layout_type = "single-column"

    # Build raw text (simple extraction)
    raw_text = page.get_text("text").strip()

    # Build structured text with layout preservation
    structured_parts = []

    # Add headers
    if headers:
        structured_parts.append("[HEADER]")
        structured_parts.extend(headers)
        structured_parts.append("[/HEADER]\n")

    # Add content blocks in reading order
    table_positions = [(t.y0, t) for t in tables]
    table_positions.sort(key=lambda x: x[0])
    table_idx = 0

    structured_parts.append("[CONTENT]")

    for idx in reading_order:
        block = text_blocks[idx]

        # Check if we need to insert a table here
        while table_idx < len(table_positions):
            table_y, table = table_positions[table_idx]
            if table_y < block.y0:
                structured_parts.append(_format_table_as_text(table))
                table_idx += 1
            else:
                break

        if block.block_type not in ("header", "footer"):
            if num_columns > 1:
                # Mark column blocks
                col_idx = int((block.x0 / page_width) * num_columns) + 1
                structured_parts.append(f"[COLUMN {col_idx}]")

            structured_parts.append(block.text)

    # Add remaining tables
    while table_idx < len(table_positions):
        _, table = table_positions[table_idx]
        structured_parts.append(_format_table_as_text(table))
        table_idx += 1

    structured_parts.append("[/CONTENT]")

    # Add footers
    if footers:
        structured_parts.append("\n[FOOTER]")
        structured_parts.extend(footers)
        structured_parts.append("[/FOOTER]")

    structured_text = "\n".join(structured_parts)

    return PageLayout(
        page_number=page_num,
        layout_type=layout_type,
        num_columns=num_columns,
        headers=headers,
        footers=footers,
        text_blocks=text_blocks,
        tables=tables,
        reading_order=reading_order,
        raw_text=raw_text,
        structured_text=structured_text
    )

def extract_pdf_pages_with_layout(
    file_path: str,
    preserve_structure: bool = True
) -> List[Tuple[int, str, PageLayout]]:
    """
    Extract text from PDF pages with structural layout analysis.

    Returns:
        List of (page_number, text, PageLayout) tuples
    """
    pages = []
    try:
        doc = fitz.open(file_path)
        for i in range(len(doc)):
            page = doc[i]
            layout = analyze_page_layout(page, i + 1)

            # Use structured text if preserving structure, otherwise raw
            text = layout.structured_text if preserve_structure else layout.raw_text

            if text:
                pages.append((i + 1, text, layout))

        doc.close()
        print(f"✅ Extracted {len(pages)} pages with layout analysis from PDF")
    except Exception as e:
        print(f"❌ PDF error: {e}")
    return pages

# =============================================================================
# PDF EXTRACTION (Original Logic - Preserved for compatibility)
# =============================================================================

def extract_pdf_pages(file_path: str) -> List[Tuple[int, str]]:
    """
    Extract text from PDF pages.
    Original logic from your code preserved exactly.
    """
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

def extract_pdf_images(
    file_path: str, 
    max_images: int = 25
) -> List[Tuple[int, Image.Image]]:
    """
    Extract images from PDF.
    Original logic from your code preserved exactly.
    """
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
                    
                    # Convert to RGB if needed
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    # Filter small images
                    if image.size[0] > 50 and image.size[1] > 50:
                        images.append((page_num + 1, image))
                except:
                    continue
        
        doc.close()
    except Exception as e:
        print(f"❌ Image extraction error: {e}")
    
    return images

# =============================================================================
# DOCX EXTRACTION (Original Logic)
# =============================================================================

def extract_docx(file_path: str) -> str:
    """
    Extract text from DOCX file.
    Original logic from your code preserved exactly.
    """
    try:
        doc = docx.Document(file_path)
        return "\n\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    except Exception as e:
        print(f"❌ DOCX error: {e}")
        return ""

# =============================================================================
# TEXT FILE EXTRACTION (Original Logic)
# =============================================================================

def extract_txt(file_path: str) -> str:
    """
    Extract text from TXT file.
    Original logic preserved.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"❌ TXT error: {e}")
        return ""
