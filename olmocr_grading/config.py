"""
Configuration for the OCR pipeline.

Values here are the defaults. They can be overridden by:
    1. A YAML or JSON config file (--config path/to/config.yaml)
    2. CLI flags (highest priority, applied last in main.py)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class OCRConfig:
    # --- Model ---
    model_id: str = "allenai/olmOCR-2-7B-1025"
    # Official model card loads the processor from the base Qwen model rather
    # than the olmOCR checkpoint itself (olmOCR-2-7B-1025 doesn't ship its
    # own processor files). Kept configurable in case that changes.
    processor_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    device: str = "auto"  # "auto" | "cuda" | "mps" | "cpu"
    dtype: str = "auto"  # "auto" | "bfloat16" | "float16" | "float32"
    attn_implementation: Optional[str] = None  # e.g. "flash_attention_2" on supported CUDA setups

    # --- Rendering ---
    target_longest_dim: int = 1288  # per olmOCR-2-7B-1025 model card
    pdf_render_dpi: int = 300  # used before downscaling to target_longest_dim

    # --- Preprocessing (optional enhancement) ---
    preprocess: bool = False
    deskew: bool = True
    denoise: bool = True
    enhance_contrast: bool = True

    # --- Generation ---
    max_new_tokens: int = 4096
    do_sample: bool = False  # deterministic decoding is safer for grading
    temperature: float = 0.1  # only used when do_sample=True
    # Repetition controls: quantized Qwen2.5-VL-family models can fall into
    # repetition loops on dense handwritten pages. These mitigate that.
    repetition_penalty: float = 1.15
    no_repeat_ngram_size: int = 6

    # --- Prompting ---
    # If True and the `olmocr` pip package is installed, use the stock
    # olmOCR toolkit prompt (build_no_anchoring_v4_yaml_prompt) instead of
    # the grading-specific custom prompt in prompts.py.
    use_stock_prompt: bool = False

    # --- Confidence (optional enhancement) ---
    compute_confidence: bool = False

    # --- Misc ---
    log_level: str = "INFO"

    def as_dict(self) -> dict:
        return asdict(self)


def load_config(path: Optional[str]) -> OCRConfig:
    """
    Load a config file (YAML or JSON) and merge it over the defaults.
    Unknown keys in the file are ignored with a warning so typos don't
    silently do nothing.
    """
    cfg = OCRConfig()
    if not path:
        return cfg

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    text = p.read_text(encoding="utf-8")
    data: dict
    if p.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # lazy import: only required if user passes a YAML config
        except ImportError as e:
            raise ImportError(
                "PyYAML is required to read .yaml config files. Install with: pip install pyyaml"
            ) from e
        data = yaml.safe_load(text) or {}
    elif p.suffix.lower() == ".json":
        data = json.loads(text) or {}
    else:
        raise ValueError(f"Unsupported config format '{p.suffix}'. Use .yaml, .yml, or .json")

    valid_keys = {f.name for f in fields(OCRConfig)}
    for key, value in data.items():
        if key not in valid_keys:
            logger.warning("Ignoring unknown config key '%s' in %s", key, path)
            continue
        setattr(cfg, key, value)

    return cfg
