"""
LLM-based marking-scheme extraction.

The regex-based table parser (main.process_marking_scheme) assumes a
fairly specific PDF-table-to-markdown shape: a question's own number
lands on the FIRST row of its block, continuation rows follow it, and
a Marks-column cell's stacked values ("2<br>3") all belong to that same
row's text. In practice, marking-scheme PDFs are inconsistent about
this - some tables put a question's earlier value-point row BEFORE the
row carrying its number, some split value points across rows in ways
that don't line up 1:1 with the Marks column, etc. Getting this wrong
silently drops or misattributes marks, which then quietly corrupts
every downstream max_marks/grading total.

This module re-derives the same {number, question_text, points:
[{text, marks}]} structure by giving the WHOLE raw markdown to the
already-loaded Qwen2.5-VL model (in text-only mode, same as
llm_segmentation.py) and asking it to read the document as a whole
rather than row-by-row - it isn't tripped up by which table row a
number physically landed on.

Self-verification: most marking-scheme headers state their own grand
total (e.g. "4 Questions x 5 Marks = 20 Marks"). detect_stated_total_marks()
extracts that number when present, and extract_marking_scheme_llm()
requires the model's own output to sum to exactly that total - a
concrete, cheap correctness check that doesn't depend on trusting the
model's arithmetic on faith. Output that fails this check (or fails to
parse/validate at all) is rejected (returns None) so the caller can
fall back to the regex parser instead of silently trusting a wrong
extraction.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

logger = logging.getLogger("autoassess.llm_marking_scheme")

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

# Matches the common "N Questions x M Marks = TOTAL Marks" header style,
# e.g. "(Theory Paper - 4 Questions x 5 Marks = 20 Marks)". Falls back to
# a looser "= TOTAL Marks" match (see detect_stated_total_marks) for
# headers phrased slightly differently.
_TOTAL_HEADER_RE = re.compile(
    r"=\s*(\d+(?:\.5|½)?)\s*\*{0,2}\s*Marks",
    re.IGNORECASE,
)

_INSTRUCTIONS = """You are extracting a marking scheme (model-answer key) from the markdown of a scanned/converted table, for an automated grading system.

The markdown below came from converting a PDF table and may have inconsistent row/column alignment - a question's value points can be split across multiple rows in ways that don't line up 1:1 with the Marks column, a question's number can land on a different row than its own first value point, and a Marks-column cell can stack multiple numbers together (e.g. "2<br>3" meaning two separate value points worth 2 and 3 marks). Read the WHOLE document as a human examiner would, not row-by-row, and reconstruct each question's true value points and marks regardless of which raw table row/cell the text or number happened to land in.

Every marks number that appears anywhere in the document belongs to exactly one value point of exactly one question - never drop one, never duplicate one, never invent one that isn't in the document.
{total_hint}
Output ONLY a JSON array and NOTHING else (no markdown fences, no commentary):
[
  {{"number": 1, "question_text": "...", "points": [{{"text": "...", "marks": 5}}]}},
  {{"number": 2, "question_text": "...", "points": [{{"text": "...", "marks": 2}}, {{"text": "...", "marks": 3}}]}}
]

Strict rules:
1. "number" is the question's number (integer), each appearing exactly once, in ascending order.
2. "question_text" is the question's own statement/prompt if it is present in the document, else "".
3. "points" is the ordered list of that question's value points. "marks" is a plain number (use 0.5 for a half-mark "½", never the symbol itself).
4. Do not merge two different questions' value points together, and do not split one value point's marks across two entries.

=== SOURCE DOCUMENT START ===
{document}
=== SOURCE DOCUMENT END ===

JSON array:"""


def detect_stated_total_marks(markdown: str) -> Optional[float]:
    """
    Best-effort extraction of a marking scheme's own stated grand total
    (e.g. the "20" in "4 Questions x 5 Marks = 20 Marks"), used as a
    self-check on extracted marks. Returns None if no such header could
    be confidently found - callers must treat that as "unverifiable",
    not "zero".
    """

    if not markdown:
        return None

    match = _TOTAL_HEADER_RE.search(markdown)
    if not match:
        return None

    try:
        return float(match.group(1).replace("½", ".5"))
    except ValueError:
        return None


def _format_marks(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _extract_json_array(raw: str) -> Optional[list]:
    cleaned = _FENCE_RE.sub("", raw.strip()).strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    match = _JSON_ARRAY_RE.search(cleaned)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            return None

    return None


def _validate_and_normalize(parsed: list) -> Optional[list[dict]]:
    if not parsed:
        return None

    questions: list[dict] = []

    for item in parsed:
        if not isinstance(item, dict):
            return None

        number = item.get("number")
        if isinstance(number, bool) or not isinstance(number, int):
            return None

        question_text = item.get("question_text", "")
        if not isinstance(question_text, str):
            return None

        raw_points = item.get("points")
        if not isinstance(raw_points, list) or not raw_points:
            return None

        points: list[dict] = []
        for raw_point in raw_points:
            if not isinstance(raw_point, dict):
                return None

            text = raw_point.get("text")
            marks = raw_point.get("marks")

            if not isinstance(text, str) or not text.strip():
                return None
            if isinstance(marks, bool) or not isinstance(marks, (int, float)):
                return None
            if marks < 0:
                return None

            points.append(
                {
                    "question_text": question_text,
                    "text": text.strip(),
                    "marks": _format_marks(marks),
                }
            )

        questions.append(
            {
                "number": number,
                "question_text": question_text,
                "points": points,
            }
        )

    # Every question number must be unique - a duplicate means the model
    # split/duplicated a question, which is exactly the kind of error
    # this extraction is meant to avoid.
    numbers = [q["number"] for q in questions]
    if len(numbers) != len(set(numbers)):
        return None

    return questions


def _sum_all_marks(questions: list[dict]) -> float:
    total = 0.0
    for question in questions:
        for point in question["points"]:
            try:
                total += float(point["marks"])
            except (ValueError, TypeError):
                pass
    return total


def extract_marking_scheme_llm(
    markdown: str,
    engine,
    stated_total: Optional[float] = None,
    max_new_tokens: int = 2048,
) -> Optional[list[dict]]:
    """
    Extract {number, question_text, points: [{text, marks}]} per
    question directly from the raw marking-scheme markdown using the
    pre-loaded Qwen2.5-VL / olmOCR model in text-only mode.

    `stated_total`, when given (see detect_stated_total_marks), is used
    both as a hint IN the prompt and as a hard post-hoc check: if the
    model's own output doesn't sum to exactly this total, the result is
    rejected (returns None) rather than trusted. Without a stated_total
    the result is accepted as long as it's structurally valid - there's
    nothing to cross-check it against.

    Returns None (never raises) on any failure - missing engine,
    generation error, unparsable/invalid output, or a total that
    disagrees with `stated_total` - so the caller can fall back to the
    regex-based parser.
    """

    if not markdown or not markdown.strip():
        return None

    if engine is None or not engine.is_ready():
        logger.info("LLM engine not available; skipping LLM marking-scheme extraction.")
        return None

    total_hint = ""
    if stated_total is not None:
        total_hint = (
            "\nIMPORTANT: according to this document's own header, it "
            f"totals EXACTLY {_format_marks(stated_total)} marks across "
            "its questions. Your output's marks must sum to exactly "
            "that total - use it to check your own extraction before "
            "answering.\n"
        )

    prompt = _INSTRUCTIONS.format(total_hint=total_hint, document=markdown)

    try:
        raw_output = engine.generate_text(prompt, max_new_tokens=max_new_tokens)
    except Exception:
        logger.exception("LLM marking-scheme extraction generation failed.")
        return None

    parsed = _extract_json_array(raw_output)
    if parsed is None:
        logger.warning(
            "LLM marking-scheme output was not valid JSON (first 200 chars: %r).",
            raw_output[:200],
        )
        return None

    questions = _validate_and_normalize(parsed)
    if questions is None:
        logger.warning("LLM marking-scheme output failed validation: %r", parsed)
        return None

    if stated_total is not None:
        extracted_total = _sum_all_marks(questions)
        if abs(extracted_total - stated_total) > 1e-6:
            logger.warning(
                "LLM marking-scheme extraction totals %s but the document "
                "states %s; rejecting (falling back to regex parsing).",
                extracted_total,
                stated_total,
            )
            return None

    logger.info(
        "LLM marking-scheme extraction succeeded: %d questions.",
        len(questions),
    )
    return questions