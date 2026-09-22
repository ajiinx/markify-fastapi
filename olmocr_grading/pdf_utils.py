"""
PDF -> per-page image rendering.

Uses PyMuPDF (fitz) rather than poppler/pdf2image so the pipeline has no
external system binary dependency -- just `pip install pymupdf`, which
works the same way on macOS (Apple Silicon), Windows, and Linux.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from PIL import Image

from .image_utils import resize_to_longest_dim

logger = logging.getLogger(__name__)


def pdf_to_images(
    pdf_path: str,
    target_longest_dim: int = 1288,
    render_dpi: int = 300,
) -> List[Image.Image]:
    """
    Convert every page of a PDF into a PIL Image, downscaled so the
    longest side is target_longest_dim (matching what olmOCR-2-7B-1025
    expects -- see the model card).

    render_dpi controls the initial rasterization resolution before
    downscaling; higher values give the resize step more detail to work
    with for pages with small handwriting, at the cost of memory/time.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise ImportError(
            "PyMuPDF is required to read PDFs. Install with: pip install pymupdf"
        ) from e

    p = Path(pdf_path)
    if not p.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    images: List[Image.Image] = []
    try:
        doc = fitz.open(str(p))
    except Exception as e:
        raise IOError(f"Failed to open PDF '{pdf_path}': {e}") from e

    try:
        if doc.page_count == 0:
            raise ValueError(f"PDF '{pdf_path}' has no pages.")

        zoom = render_dpi / 72.0  # PDF base unit is 72 dpi
        matrix = fitz.Matrix(zoom, zoom)

        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            img = resize_to_longest_dim(img, target_longest_dim)
            images.append(img)
    finally:
        doc.close()

    logger.info("Rendered %d page(s) from %s", len(images), pdf_path)
    return images
