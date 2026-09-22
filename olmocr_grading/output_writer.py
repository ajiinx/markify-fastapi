"""
Write OCR results to disk as plain text or JSON, matching the schema
expected by the downstream answer-sheet grading system.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

from model_client import PageResult

logger = logging.getLogger(__name__)


def save_txt(pages: List[PageResult], output_path: str) -> None:
    """
    Save as plain text. Single-page documents are written as-is;
    multi-page documents get a clear page separator so question numbers
    that repeat across pages aren't ambiguous to a downstream parser.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if len(pages) == 1:
        content = pages[0].text
    else:
        blocks = []
        for p in pages:
            blocks.append(f"===== Page {p.page} =====\n{p.text}")
        content = "\n\n".join(blocks)

    out.write_text(content, encoding="utf-8")
    logger.info("Saved text output to %s", output_path)


def save_json(document_name: str, pages: List[PageResult], output_path: str) -> None:
    """Save in the evaluation-friendly JSON schema:
    {document_name, page_count, pages: [{page, text}, ...]}
    (confidence is included per page only when it was computed).
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    page_entries = []
    for p in pages:
        entry = {"page": p.page, "text": p.text}
        if p.confidence is not None:
            entry["confidence"] = round(p.confidence, 4)
        page_entries.append(entry)

    payload = {
        "document_name": document_name,
        "page_count": len(pages),
        "pages": page_entries,
    }

    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved JSON output to %s", output_path)
