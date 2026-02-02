"""
Grounding Utility Module
=========================
Parse and draw bounding boxes from Qwen3-VL's visual grounding output.
Handles the 0-1000 normalized coordinate system.
"""

import re
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont


def parse_bounding_boxes(text: str) -> List[Tuple[int, int, int, int]]:
    """
    Parse bounding box coordinates from Qwen3-VL output.

    Supports multiple formats:
    - <box>[ymin, xmin, ymax, xmax]</box>
    - {"bbox_2d": [ymin, xmin, ymax, xmax]}
    - [[ymin, xmin, ymax, xmax]]

    All coordinates are normalized to 0-1000.

    Returns:
        List of (ymin, xmin, ymax, xmax) tuples
    """
    boxes = []

    # Pattern 1: <box>[y1, x1, y2, x2]</box>
    pattern1 = re.findall(
        r'<box>\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\s*</box>',
        text
    )
    for match in pattern1:
        boxes.append(tuple(int(x) for x in match))

    # Pattern 2: {"bbox_2d": [y1, x1, y2, x2]}
    pattern2 = re.findall(
        r'"bbox_2d":\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]',
        text
    )
    for match in pattern2:
        boxes.append(tuple(int(x) for x in match))

    # Pattern 3: standalone [y1, x1, y2, x2] (only if no other patterns matched)
    if not boxes:
        pattern3 = re.findall(
            r'\[(\d{1,4}),\s*(\d{1,4}),\s*(\d{1,4}),\s*(\d{1,4})\]',
            text
        )
        for match in pattern3:
            coords = tuple(int(x) for x in match)
            # Only accept if all values are in 0-1000 range
            if all(0 <= c <= 1000 for c in coords):
                boxes.append(coords)

    return boxes


def draw_bounding_boxes(
    image: Image.Image,
    boxes: List[Tuple[int, int, int, int]],
    labels: Optional[List[str]] = None,
    color: str = "red",
    width: int = 3
) -> Image.Image:
    """
    Draw bounding boxes on a page image.

    Coordinates are Qwen3-VL's 0-1000 normalized format: [ymin, xmin, ymax, xmax].
    These are mapped to the actual image pixel dimensions.

    Args:
        image: PIL Image to draw on (copy is made)
        boxes: List of (ymin, xmin, ymax, xmax) in 0-1000 normalized coords
        labels: Optional labels to draw above each box
        color: Box outline color
        width: Box outline width in pixels

    Returns:
        New PIL Image with bounding boxes drawn
    """
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    for i, (ymin, xmin, ymax, xmax) in enumerate(boxes):
        # Scale from 0-1000 normalized to actual pixel coordinates
        left = int(xmin * w / 1000)
        top = int(ymin * h / 1000)
        right = int(xmax * w / 1000)
        bottom = int(ymax * h / 1000)

        draw.rectangle([left, top, right, bottom], outline=color, width=width)

        if labels and i < len(labels):
            # Draw label above the box
            label_y = max(0, top - 15)
            draw.text((left, label_y), labels[i], fill=color)

    return img


def extract_grounded_response(text: str) -> Tuple[str, List[Tuple[int, int, int, int]]]:
    """
    Separate the text response from bounding box annotations.

    Args:
        text: Full model response potentially containing <box> tags

    Returns:
        Tuple of (clean_text, bounding_boxes)
    """
    # Extract boxes
    boxes = parse_bounding_boxes(text)

    # Remove box tags from text for clean display
    clean_text = re.sub(r'<box>\s*\[\d+,\s*\d+,\s*\d+,\s*\d+\]\s*</box>', '', text)
    clean_text = clean_text.strip()

    return clean_text, boxes
