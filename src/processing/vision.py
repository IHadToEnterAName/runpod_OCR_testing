"""
Vision Processing Module
=========================
Image analysis using vision model.
Original logic from your code preserved exactly.
"""

import asyncio
import base64
import io
from PIL import Image
from openai import AsyncOpenAI
from typing import Optional
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

# Semaphore for concurrency control (original logic)
vision_semaphore = asyncio.Semaphore(config.performance.vision_concurrent_limit)

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
# VISION ANALYSIS (Original Logic)
# =============================================================================

async def analyze_image(
    image: Image.Image, 
    page: int, 
    idx: int
) -> str:
    """
    Analyze image using vision model.
    Original logic from your code preserved exactly.
    """
    async with vision_semaphore:
        try:
            # Resize image
            resized = resize_image(image)
            
            # Convert to base64
            b64 = image_to_base64(resized)
            
            # Call vision API
            response = await vision_client.chat.completions.create(
                model=config.models.vision_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}"
                            }
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
            
            # Extract result
            result = response.choices[0].message.content
            
            # Filter CJK characters
            if result:
                return filter_cjk(result.strip())
            else:
                return "[No description]"
            
        except Exception as e:
            print(f"❌ Vision error: {e}")
            return f"[Image analysis failed]"
