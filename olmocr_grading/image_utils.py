"""
Image loading, resizing, and optional preprocessing (deskew / denoise /
contrast enhancement) for scanned handwritten answer sheets.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def load_image(path: str) -> Image.Image:
    """Load an image file from disk as an RGB PIL Image."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    if p.suffix.lower() not in SUPPORTED_IMAGE_EXTS:
        raise ValueError(
            f"Unsupported image extension '{p.suffix}'. Supported: {sorted(SUPPORTED_IMAGE_EXTS)}"
        )
    try:
        img = Image.open(p)
        # Respect EXIF rotation (common with phone-scanned answer sheets)
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")
    except Exception as e:
        raise IOError(f"Failed to open image '{path}': {e}") from e


def resize_to_longest_dim(img: Image.Image, target_longest_dim: int) -> Image.Image:
    """
    Resize so the longest side equals target_longest_dim, preserving aspect
    ratio. olmOCR-2-7B-1025 was trained expecting the longest dimension to
    be 1288px -- see the model card's "Usage" section.
    """
    w, h = img.size
    longest = max(w, h)
    if longest == target_longest_dim:
        return img
    scale = target_longest_dim / float(longest)
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return img.resize(new_size, Image.LANCZOS)


def preprocess_image(
    img: Image.Image,
    deskew: bool = True,
    denoise: bool = True,
    enhance_contrast: bool = True,
) -> Image.Image:
    """
    Optional preprocessing pass to help the OCR model with photographed /
    scanned handwritten pages. Each step is independently toggleable since
    aggressive preprocessing can sometimes hurt VLM-based OCR (unlike
    classical OCR engines, these models were trained mostly on natural
    scans, so keep this conservative).
    """
    try:
        import cv2
    except ImportError as e:
        raise ImportError(
            "opencv-python is required for --preprocess. Install with: pip install opencv-python"
        ) from e

    arr = np.array(img)  # RGB
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    if denoise:
        bgr = cv2.fastNlMeansDenoisingColored(bgr, None, h=7, hColor=7, templateWindowSize=7, searchWindowSize=21)

    if enhance_contrast:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        lab = cv2.merge((l_channel, a_channel, b_channel))
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if deskew:
        bgr = _deskew(bgr)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _deskew(bgr: "np.ndarray") -> "np.ndarray":
    """Estimate and correct small rotational skew using the minAreaRect
    of foreground (ink) pixels. Conservative: only corrects small angles
    (< 15 deg) to avoid flipping a page that's simply landscape-oriented."""
    import cv2

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] < 50:
        return bgr  # not enough foreground to estimate skew reliably

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < 0.1 or abs(angle) > 15:
        return bgr  # skip: negligible skew, or angle estimate looks unreliable

    (h, w) = bgr.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        bgr, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated
