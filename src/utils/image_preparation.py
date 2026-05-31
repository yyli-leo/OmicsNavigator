"""
Image preparation utilities for VisualProfiler agent.

Adapted from interpretation_module/src/processing/image.py
"""

import os
import time
import uuid
import base64
import mimetypes
import tempfile
from pathlib import Path
from typing import Dict, Any
from PIL import Image, ImageDraw


def encode_image(image_path: Path) -> str:
    """Read image and convert to base64."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_mime_type(image_path: Path) -> str:
    """Get MIME type of image."""
    mime_type, _ = mimetypes.guess_type(image_path)
    return mime_type or "image/png"


def load_rendered_image_bytes(region_id: str, data_root: Path) -> bytes:
    """
    Load the pre-rendered image bytes for a region (thread-safe).

    Args:
        region_id: Region identifier (e.g., "s255")
        data_root: Root directory containing the data

    Returns:
        Raw image bytes

    Raises:
        FileNotFoundError: If rendered image not found
    """
    region_dir = data_root / region_id
    rendered_path = region_dir / f"{region_id}_rendered.png"
    if not rendered_path.exists():
        raise FileNotFoundError(f"Rendered image not found: {rendered_path}")
    with open(rendered_path, "rb") as f:
        return f.read()


def add_bounding_box_pil(image: Image.Image, roi: Dict[str, Any], color='red', width=15) -> Image.Image:
    """
    Add a bounding box to a PIL Image.

    Args:
        image: PIL Image object
        roi: ROI dictionary with 'xyranges' key containing (x0, x1, y0, y1)
        color: Box color (default: red)
        width: Box line width (default: 15)

    Returns:
        PIL Image with bounding box
    """
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    img_with_bbox = image.copy()
    draw = ImageDraw.Draw(img_with_bbox)

    # Handle both ROI object and dict
    if hasattr(roi, 'xyranges'):
        x0, x1, y0, y1 = roi.xyranges
    elif isinstance(roi, dict) and 'xyranges' in roi:
        x0, x1, y0, y1 = roi['xyranges']
    elif isinstance(roi, dict) and 'roi_obj' in roi:
        x0, x1, y0, y1 = roi['roi_obj'].xyranges
    else:
        raise ValueError(f"Invalid ROI format: {roi}")

    x_min, x_max = min(x0, x1), max(x0, x1)
    y_min, y_max = min(y0, y1), max(y0, y1)
    draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=width)
    return img_with_bbox


def crop_roi(image: Image.Image, roi: Dict[str, Any]) -> Image.Image:
    """
    Crop the ROI region from an image.

    Args:
        image: PIL Image object
        roi: ROI dictionary with 'xyranges' key

    Returns:
        Cropped PIL Image
    """
    # Handle both ROI object and dict
    if hasattr(roi, 'xyranges'):
        x0, x1, y0, y1 = roi.xyranges
    elif isinstance(roi, dict) and 'xyranges' in roi:
        x0, x1, y0, y1 = roi['xyranges']
    elif isinstance(roi, dict) and 'roi_obj' in roi:
        x0, x1, y0, y1 = roi['roi_obj'].xyranges
    else:
        raise ValueError(f"Invalid ROI format: {roi}")

    x_min, x_max = min(x0, x1), max(x0, x1)
    y_min, y_max = min(y0, y1), max(y0, y1)
    return image.crop((x_min, y_min, x_max, y_max))


def prepare_images_for_llm(rendered_image: Image.Image, roi: Dict[str, Any], scale_factor=0.25) -> Dict[str, Any]:
    """
    Prepare images for VisualProfiler LLM input.

    Creates two images:
    1. Full slide image with red bounding box (downscaled)
    2. Zoomed ROI view (cropped, original resolution)

    Args:
        rendered_image: PIL Image of the full rendered slide
        roi: ROI dictionary with patch_name and xyranges
        scale_factor: Scale factor for full image (default: 0.25)

    Returns:
        Dictionary with paths and metadata:
        - full_path: Path to downscaled full image with bbox
        - small_path: Path to cropped ROI view
        - temp_dir: Temporary directory path
        - full_size: (width, height) of full image
        - small_size: (width, height) of small image
    """
    # Get ROI object from dict if needed
    roi_obj = roi.get('roi_obj') if isinstance(roi, dict) else roi

    # Add bounding box to full image
    full_image = add_bounding_box_pil(rendered_image, roi_obj, color='red', width=15)

    # Downscale full image
    new_width = int(full_image.width * scale_factor)
    new_height = int(full_image.height * scale_factor)
    full_image_scaled = full_image.resize((new_width, new_height), resample=Image.Resampling.BILINEAR)

    # Crop small image from original (no bbox)
    small_image = crop_roi(rendered_image, roi_obj)

    # Save to temporary files for LLM
    temp_dir = Path(tempfile.gettempdir()) / f"roi_pipeline_{os.getpid()}"
    temp_dir.mkdir(exist_ok=True)

    # Get patch name
    if isinstance(roi, dict):
        patch_name = roi.get('patch_name', 'roi')
    else:
        patch_name = getattr(roi, 'patch_name', 'roi')

    # Use unique filenames to avoid conflicts
    unique_id = str(uuid.uuid4())[:8]
    full_path = temp_dir / f"{patch_name}_{unique_id}_full.png"
    small_path = temp_dir / f"{patch_name}_{unique_id}_small.png"

    # Save with explicit flush
    full_image_scaled.save(full_path, optimize=False)
    small_image.save(small_path, optimize=False)

    # Ensure files are written to disk
    time.sleep(0.05)

    return {
        'full_path': full_path,
        'small_path': small_path,
        'temp_dir': temp_dir,
        'full_size': (new_width, new_height),
        'small_size': small_image.size,
    }


def cleanup_temp_images(img_data: Dict[str, Any]) -> None:
    """
    Clean up temporary image files.

    Args:
        img_data: Dictionary returned from prepare_images_for_llm
    """
    if img_data and 'full_path' in img_data:
        try:
            if img_data['full_path'].exists():
                img_data['full_path'].unlink()
        except:
            pass
    if img_data and 'small_path' in img_data:
        try:
            if img_data['small_path'].exists():
                img_data['small_path'].unlink()
        except:
            pass
