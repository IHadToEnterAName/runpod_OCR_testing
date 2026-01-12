"""
Document Processing Module
===========================
All document extraction functions from original code preserved exactly.
"""

import fitz  # PyMuPDF
import docx
from PIL import Image
import io
from typing import List, Tuple

# =============================================================================
# PDF EXTRACTION (Original Logic)
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
