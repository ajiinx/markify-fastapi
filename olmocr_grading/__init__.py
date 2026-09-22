"""
olmocr_grading
==============

A small, reusable library for running allenai/olmOCR-2-7B-1025 over
handwritten student answer sheets (images or PDFs) and producing
evaluation-friendly output for an automated grading pipeline.

Public API:
    - OCRConfig            (config.py)   configuration dataclass + loader
    - OlmOCREngine          (ocr_engine.py) model wrapper / inference
    - pdf_to_images          (pdf_utils.py)  PDF -> list[PIL.Image]
    - load_image, preprocess_image (image_utils.py)
    - save_txt, save_json    (output_writer.py)
"""

from .config import OCRConfig, load_config

from .pdf_utils import pdf_to_images
from .image_utils import load_image, preprocess_image, resize_to_longest_dim
from .output_writer import save_txt, save_json

__all__ = [
    "OCRConfig",
    "load_config",

    "pdf_to_images",
    "load_image",
    "preprocess_image",
    "resize_to_longest_dim",
    "save_txt",
    "save_json",
]
