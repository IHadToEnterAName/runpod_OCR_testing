"""
Vision Processing Module
=========================
Image analysis using vision model with Multi-Stage Verification.
Enhanced with self-correction and confidence evaluation.
"""

import asyncio
import base64
import io
import re
from PIL import Image
from openai import AsyncOpenAI
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from config.settings import get_config
from utils.helpers import filter_cjk

# =============================================================================
# CONFIGURATION
# =============================================================================

config = get_config()

# Initialize vision client
vision_client = AsyncOpenAI(
    base_url=config.models.vision_base_url,
    api_key="EMPTY"
)

# Initialize reasoning client for self-correction
reasoning_client = AsyncOpenAI(
    base_url=config.models.reasoning_base_url,
    api_key="EMPTY"
)

# Semaphore for concurrency control (original logic)
vision_semaphore = asyncio.Semaphore(config.performance.vision_concurrent_limit)
reasoning_semaphore = asyncio.Semaphore(config.performance.llm_concurrent_limit)

# =============================================================================
# VERIFICATION DATA STRUCTURES
# =============================================================================

@dataclass
class OCRResult:
    """Result from OCR with confidence metadata."""
    text: str
    confidence: float  # 0.0 to 1.0
    corrections_made: int
    original_text: str
    layout_description: str

# =============================================================================
# IMAGE PROCESSING (Original Logic)
# =============================================================================

def resize_image(image: Image.Image, max_size: int = 384) -> Image.Image:
    """
    Resize image to max dimension.
    Original logic from your code preserved exactly.
    """
    w, h = image.size
    if w <= max_size and h <= max_size:
        return image
    
    ratio = min(max_size / w, max_size / h)
    new_size = (int(w * ratio), int(h * ratio))
    
    return image.resize(new_size, Image.Resampling.LANCZOS)

def image_to_base64(image: Image.Image) -> str:
    """
    Convert PIL Image to base64 string.
    Original logic from your code preserved exactly.
    """
    # Convert to RGB if needed
    if image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGB')
    
    # Save to buffer
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=70)
    
    # Encode to base64
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# =============================================================================
# MULTI-STAGE VERIFICATION PROMPTS
# =============================================================================

LAYOUT_ANALYSIS_PROMPT = """Analyze this image's structural layout BEFORE extracting text.

Describe in order:
1. LAYOUT TYPE: (single-column, multi-column, table, form, mixed)
2. STRUCTURAL ELEMENTS: List headers, footers, sidebars, captions
3. DATA BLOCKS: Identify distinct text regions (numbered 1, 2, 3...)
4. TABLE DETECTION: If tables exist, describe rows/columns
5. READING ORDER: Specify the logical reading sequence

Then extract ALL text, maintaining the structure you identified.
Separate distinct blocks with [BLOCK N] markers.
Include ALL text, numbers, symbols, and data visible."""

CONFIDENCE_EVALUATION_PROMPT = """You are an OCR quality evaluator. Analyze this transcription for errors.

ORIGINAL OCR OUTPUT:
{ocr_text}

Evaluate:
1. CONFIDENCE SCORE (0-100): Rate transcription accuracy
2. SUSPICIOUS PATTERNS: List words/numbers that look malformed
3. COMMON OCR ERRORS: Check for l/1, O/0, rn/m confusion
4. LOGICAL INCONSISTENCIES: Numbers that don't add up, broken words
5. MISSING CONTENT: Gaps or incomplete sentences

Output format:
CONFIDENCE: [score]
SUSPICIOUS: [list]
CORRECTIONS_NEEDED: [yes/no]"""

SELF_CORRECTION_PROMPT = """You are an OCR error corrector. Fix the transcription using reasoning.

ORIGINAL OCR OUTPUT:
{ocr_text}

IDENTIFIED ISSUES:
{issues}

LAYOUT CONTEXT:
{layout}

Apply corrections:
1. Fix character substitution errors (l→1, O→0, rn→m)
2. Complete truncated words using context
3. Fix number sequences that don't match patterns
4. Restore logical sentence structure
5. Preserve original meaning - don't add information

Output ONLY the corrected text, maintaining the original structure and [BLOCK N] markers."""

# =============================================================================
# STAGE 1: LAYOUT-AWARE INITIAL OCR
# =============================================================================

async def _stage1_layout_ocr(image: Image.Image, b64: str) -> Tuple[str, str]:
    """
    Stage 1: Analyze layout and extract text with structure awareness.
    Returns (extracted_text, layout_description)
    """
    response = await vision_client.chat.completions.create(
        model=config.models.vision_model,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                },
                {
                    "type": "text",
                    "text": LAYOUT_ANALYSIS_PROMPT
                }
            ]
        }],
        max_tokens=config.tokens.vision_max_tokens,
        temperature=0.2  # Lower temp for more consistent structure detection
    )

    result = response.choices[0].message.content or ""

    # Extract layout description (first few lines typically)
    lines = result.strip().split('\n')
    layout_lines = []
    text_lines = []
    in_text = False

    for line in lines:
        if '[BLOCK' in line.upper() or in_text:
            in_text = True
            text_lines.append(line)
        else:
            layout_lines.append(line)

    layout_desc = '\n'.join(layout_lines[:10])  # First 10 lines as layout
    extracted_text = '\n'.join(text_lines) if text_lines else result

    return filter_cjk(extracted_text.strip()), filter_cjk(layout_desc.strip())

# =============================================================================
# STAGE 2: CONFIDENCE EVALUATION
# =============================================================================

async def _stage2_evaluate_confidence(ocr_text: str) -> Tuple[float, str, bool]:
    """
    Stage 2: Evaluate OCR output confidence and identify issues.
    Returns (confidence_score, issues_description, needs_correction)
    """
    async with reasoning_semaphore:
        try:
            response = await reasoning_client.chat.completions.create(
                model=config.models.reasoning_model,
                messages=[{
                    "role": "user",
                    "content": CONFIDENCE_EVALUATION_PROMPT.format(ocr_text=ocr_text)
                }],
                max_tokens=512,
                temperature=0.1  # Very low for consistent evaluation
            )

            result = response.choices[0].message.content or ""
            result = filter_cjk(result)

            # Parse confidence score
            confidence = 0.75  # Default
            confidence_match = re.search(r'CONFIDENCE:\s*(\d+)', result, re.IGNORECASE)
            if confidence_match:
                confidence = int(confidence_match.group(1)) / 100.0

            # Check if corrections needed
            needs_correction = 'CORRECTIONS_NEEDED: YES' in result.upper() or confidence < 0.8

            # Extract issues
            issues = result

            return confidence, issues, needs_correction

        except Exception as e:
            print(f"⚠️ Confidence evaluation skipped: {e}")
            return 0.75, "", False

# =============================================================================
# STAGE 3: SELF-CORRECTION
# =============================================================================

async def _stage3_self_correct(
    ocr_text: str,
    issues: str,
    layout: str
) -> Tuple[str, int]:
    """
    Stage 3: Apply reasoning-based corrections to OCR output.
    Returns (corrected_text, corrections_count)
    """
    async with reasoning_semaphore:
        try:
            response = await reasoning_client.chat.completions.create(
                model=config.models.reasoning_model,
                messages=[{
                    "role": "user",
                    "content": SELF_CORRECTION_PROMPT.format(
                        ocr_text=ocr_text,
                        issues=issues,
                        layout=layout
                    )
                }],
                max_tokens=config.tokens.vision_max_tokens,
                temperature=0.2
            )

            corrected = response.choices[0].message.content or ocr_text
            corrected = filter_cjk(corrected.strip())

            # Count corrections (simple heuristic: character differences)
            corrections = sum(1 for a, b in zip(ocr_text, corrected) if a != b)
            corrections += abs(len(ocr_text) - len(corrected))

            return corrected, corrections

        except Exception as e:
            print(f"⚠️ Self-correction skipped: {e}")
            return ocr_text, 0

# =============================================================================
# VISION ANALYSIS WITH MULTI-STAGE VERIFICATION
# =============================================================================

async def analyze_image(
    image: Image.Image,
    page: int,
    idx: int,
    enable_verification: bool = True
) -> str:
    """
    Analyze image using vision model with optional multi-stage verification.

    Pipeline:
    1. Layout-aware initial OCR
    2. Confidence evaluation
    3. Self-correction (if confidence < threshold)

    Args:
        image: PIL Image to analyze
        page: Page number
        idx: Image index on page
        enable_verification: If True, run full verification pipeline

    Returns:
        Extracted and verified text
    """
    async with vision_semaphore:
        try:
            # Resize image
            resized = resize_image(image)

            # Convert to base64
            b64 = image_to_base64(resized)

            # === STAGE 1: Layout-Aware OCR ===
            ocr_text, layout_desc = await _stage1_layout_ocr(resized, b64)

            if not ocr_text or ocr_text == "[No description]":
                return "[No description]"

            if not enable_verification:
                # Return early if verification disabled
                return ocr_text

            # === STAGE 2: Confidence Evaluation ===
            confidence, issues, needs_correction = await _stage2_evaluate_confidence(ocr_text)

            print(f"📊 OCR Confidence (p{page}, img{idx}): {confidence:.0%}")

            if not needs_correction:
                # High confidence - return as-is
                return ocr_text

            # === STAGE 3: Self-Correction ===
            corrected_text, corrections = await _stage3_self_correct(
                ocr_text, issues, layout_desc
            )

            if corrections > 0:
                print(f"✏️ Applied {corrections} corrections (p{page}, img{idx})")

            return corrected_text

        except Exception as e:
            print(f"❌ Vision error: {e}")
            return f"[Image analysis failed]"

async def analyze_image_detailed(
    image: Image.Image,
    page: int,
    idx: int
) -> OCRResult:
    """
    Analyze image and return detailed OCR result with metadata.
    Use this for quality analysis and validation.
    """
    async with vision_semaphore:
        try:
            resized = resize_image(image)
            b64 = image_to_base64(resized)

            # Stage 1
            ocr_text, layout_desc = await _stage1_layout_ocr(resized, b64)
            original_text = ocr_text

            if not ocr_text:
                return OCRResult(
                    text="[No description]",
                    confidence=0.0,
                    corrections_made=0,
                    original_text="",
                    layout_description=""
                )

            # Stage 2
            confidence, issues, needs_correction = await _stage2_evaluate_confidence(ocr_text)

            corrections = 0
            if needs_correction:
                # Stage 3
                ocr_text, corrections = await _stage3_self_correct(
                    ocr_text, issues, layout_desc
                )

            return OCRResult(
                text=ocr_text,
                confidence=confidence,
                corrections_made=corrections,
                original_text=original_text,
                layout_description=layout_desc
            )

        except Exception as e:
            print(f"❌ Vision error: {e}")
            return OCRResult(
                text="[Image analysis failed]",
                confidence=0.0,
                corrections_made=0,
                original_text="",
                layout_description=""
            )

# =============================================================================
# LEGACY FUNCTION (for backwards compatibility)
# =============================================================================

async def analyze_image_simple(
    image: Image.Image,
    page: int,
    idx: int
) -> str:
    """
    Simple image analysis without verification pipeline.
    Original logic preserved for comparison/fallback.
    """
    async with vision_semaphore:
        try:
            resized = resize_image(image)
            b64 = image_to_base64(resized)

            response = await vision_client.chat.completions.create(
                model=config.models.vision_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                        },
                        {
                            "type": "text",
                            "text": "Describe this image. Include ALL text, numbers, and data visible."
                        }
                    ]
                }],
                max_tokens=config.tokens.vision_max_tokens,
                temperature=0.3
            )

            result = response.choices[0].message.content
            return filter_cjk(result.strip()) if result else "[No description]"

        except Exception as e:
            print(f"❌ Vision error: {e}")
            return f"[Image analysis failed]"
