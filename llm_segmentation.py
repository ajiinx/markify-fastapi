"""
LLM-based question segmentation.

Uses the already-loaded olmOCR / Qwen2.5-VL model (see olmocr_grading.ocr_engine)
in pure text mode to split a full OCR'd/markdown document into per-question
segments. This replaces brittle regex-marker matching with a model that can
understand messy handwritten-answer OCR output (missing markers, OCR'd "Q"
read as "A", inconsistent numbering, run-on answers, etc.).

Output contract is identical to segmentation.segment_document():

    [
        {"question_id": "Q1", "text": "..."},
        ...
    ]

Safety net: every LLM call is validated before being trusted. If the model
is unavailable, the call errors out, the output isn't valid JSON, or the
returned segments don't plausibly cover the source document, we fall back
to the existing regex-based segmenter rather than risk silently dropping
or corrupting student answer content.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from segmentation import segment_document as _regex_segment_document

logger = logging.getLogger("autoassess.llm_segmentation")

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)
_QID_RE = re.compile(r"^Q?\s*(\d+)$", re.IGNORECASE)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

_INSTRUCTIONS = """You are a precise document segmentation assistant for a student answer-sheet grading system.

You will be given the full OCR'd text of ONE scanned answer sheet, with each line prefixed by its line number in square brackets, like "[12] some text". It may contain multiple question/answer pairs, inconsistent or OCR-garbled markers (e.g. "Q" misread as "A", "Ans(1)", "Answer 3", "Q.1", plain numbering, or no marker at all for the first question), messy handwriting artifacts, and spelling/grammar mistakes from the student.

Your ONLY job is to find where each new question/answer BEGINS and report its line number. Do not grade, correct, summarize, paraphrase, or reproduce any answer content.

Output format - a JSON object containing a "segments" array and NOTHING else (no markdown fences, no commentary):
{
  "segments": [
    {"question_id": "Q1", "start_line": 1},
    {"question_id": "Q2", "start_line": 14}
  ]
}

Strict rules:
1. "start_line" is the line number (the number in [brackets]) where that question's marker/heading line begins. This must be the FIRST line belonging to that question, i.e. the line containing its Q/Question/Ans/Answer marker or number.
2. "question_id" must be exactly "Q" followed by the question's NUMBER, e.g. "Q1", "Q2", "Q10" - use the question number even if the student wrote "A3" or "Ans 3" (that still means answer/question 3).
3. Start a new entry at every genuine question boundary, however it is marked (Q1, Q.1, Question 1, A1/Ans1/Answer 1, plain "1)" numbering restarting a new question, etc.). Use judgment: a numbered sub-point WITHIN one answer (e.g. a numbered list inside a single answer, like types or steps) is NOT a new question boundary - only report a boundary when a genuinely new question/answer begins.
4. List entries in ascending order of start_line, one per question, with no duplicates.
5. If the whole document is genuinely a single question/answer with no boundaries, return one element with question_id "Q1" and start_line equal to the first line number in the document.
6. Never invent a question/start_line that isn't actually in the document.

Now find the question boundaries in the following document.

=== SOURCE DOCUMENT START ===
{document}
=== SOURCE DOCUMENT END ===

JSON object:"""


def _number_lines(text: str) -> tuple[str, list[str]]:
    """Prefix each line with a 1-based line number for the model to
    reference, and return the raw (unprefixed) line list for slicing."""
    lines = text.split("\n")
    numbered = "\n".join(f"[{i + 1}] {line}" for i, line in enumerate(lines))
    return numbered, lines


def _build_reference_block(expected_questions: Optional[list[int]]) -> str:
    if not expected_questions:
        return ""

    numbers_str = ", ".join(str(n) for n in expected_questions)
    return (
        "\nIMPORTANT REFERENCE INFO: according to the official model answer key for this "
        f"paper, this answer sheet should contain answers to EXACTLY these {len(expected_questions)} "
        f"question numbers, in this order: {numbers_str}. Trust this reference over any single "
        "garbled digit in the noisy student OCR - handwritten-digit OCR frequently misreads numbers "
        "(e.g. a '3' read as '8', a '1' read as '7', a '4' read as '9'). If a marker line's digit "
        "doesn't match the next expected number, but everything else about it (its position right "
        "after the previous answer ends, starting a clearly new topic/answer) indicates it IS the "
        "next expected question, label it with the EXPECTED number, not the misread digit. "
        f"You MUST return exactly {len(expected_questions)} boundaries total, one per expected "
        "question number, in the given order - never more, never fewer, and never invent a boundary "
        "at a sub-point/bullet within an answer just because it happens to be numbered.\n"
    )


def _build_prompt(numbered_text: str, expected_questions: Optional[list[int]] = None) -> str:
    prompt = _INSTRUCTIONS.replace("{document}", numbered_text)
    reference_block = _build_reference_block(expected_questions)
    if reference_block:
        prompt = prompt.replace(
            "Now find the question boundaries in the following document.",
            reference_block + "\nNow find the question boundaries in the following document.",
        )
    return prompt


def _extract_json_array(raw: str) -> Optional[list]:
    cleaned = _FENCE_RE.sub("", raw.strip()).strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and "segments" in parsed:
            if isinstance(parsed["segments"], list):
                return parsed["segments"]
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


def _normalize_boundaries(parsed: list, num_lines: int) -> Optional[list[dict]]:
    if not parsed:
        return None

    boundaries: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            return None

        qid = item.get("question_id")
        start_line = item.get("start_line")

        if not isinstance(qid, str):
            return None
        if isinstance(start_line, bool):  # bool is an int subclass; exclude explicitly
            return None
        if not isinstance(start_line, int):
            return None

        match = _QID_RE.match(qid.strip())
        if not match:
            return None

        # Convert 1-based model line number to a 0-based index, clamped
        # into range in case the model is off by a line or two.
        line_index = max(0, min(start_line - 1, num_lines - 1))

        boundaries.append({"question_id": f"Q{int(match.group(1))}", "line_index": line_index})

    if not boundaries:
        return None

    # Sort by position and drop duplicate/out-of-order boundaries so the
    # slicing below always produces non-overlapping, forward-moving spans.
    boundaries.sort(key=lambda b: b["line_index"])
    deduped: list[dict[str, Any]] = []
    seen_lines = set()
    for b in boundaries:
        if b["line_index"] in seen_lines:
            continue
        seen_lines.add(b["line_index"])
        deduped.append(b)

    return deduped


def _reconcile_with_expected(
    boundaries: list[dict],
    expected_questions: list[int],
) -> Optional[list[dict]]:
    """
    When we have a trusted question-number sequence from the model-answer
    key, require the LLM's boundary count to match it exactly and then
    relabel boundaries positionally with the expected numbers. This is
    what corrects OCR digit corruption (e.g. "3." misread as "8.") and
    rejects spurious boundaries the model may have hallucinated at a
    numbered sub-point inside an answer.

    A count mismatch means the LLM's structural read of the document
    disagrees with the reference, so we don't trust it - caller falls
    back to regex segmentation in that case.
    """
    if len(boundaries) != len(expected_questions):
        logger.warning(
            "LLM segmentation returned %d boundaries but the model-answer "
            "reference expects %d (%s); falling back to regex segmentation.",
            len(boundaries),
            len(expected_questions),
            expected_questions,
        )
        return None

    return [
        {"question_id": f"Q{expected_num}", "line_index": boundary["line_index"]}
        for boundary, expected_num in zip(boundaries, expected_questions)
    ]


def _slice_segments(boundaries: list[dict], lines: list[str]) -> list[dict]:
    segments = []
    for i, boundary in enumerate(boundaries):
        start = boundary["line_index"]
        end = boundaries[i + 1]["line_index"] if i + 1 < len(boundaries) else len(lines)
        segment_text = "\n".join(lines[start:end]).strip()
        if not segment_text:
            continue
        segments.append({"question_id": boundary["question_id"], "text": segment_text})
    return segments


async def segment_document_llm(
    text: str,
    engine,
    expected_questions: Optional[list[int]] = None,
    max_new_tokens: int = 1024,
) -> list[dict]:
    """
    Segment a document into question-wise sections using the pre-loaded
    Qwen2.5-VL / olmOCR model in text-only mode.

    `engine` is expected to be an olmocr_grading.ocr_engine.OlmOCREngine
    instance that has already had .load() called (i.e. main.OCR_ENGINE).

    `expected_questions`, when provided, is the sorted list of question
    numbers from the corresponding model-answer document in MongoDB
    (e.g. [1, 2, 3, 4]). It's used as a trusted reference: the model is
    told exactly how many questions to expect and in what order, and the
    returned boundaries are relabeled positionally against it. This is
    what makes segmentation robust to OCR digit corruption (a "3."
    misread as "8.") and to numbered sub-points inside an answer being
    mistaken for new question boundaries - the LLM's boundary COUNT must
    match the reference or the result is rejected outright.

    Design note: the model is only asked to identify each question's
    starting LINE NUMBER (not to reproduce any answer text). The actual
    segment text is then sliced out of the original document in Python.
    This keeps the model's output tiny regardless of document length
    (avoids max_new_tokens truncation on long answer sheets) and makes
    the segment text byte-for-byte identical to the source - the model
    can never corrupt, paraphrase, or truncate answer content.

    Always falls back to the regex-based segmenter (segmentation.py) on
    any failure: missing engine, generation error, unparsable output,
    boundaries that fail validation, or a boundary count that disagrees
    with expected_questions.
    """
    if not text or not text.strip():
        return []

    if engine is None or not engine.is_ready():
        logger.info("LLM engine not available; using regex segmentation.")
        return _regex_segment_document(text)

    numbered_text, lines = _number_lines(text)

    try:
        raw_output = await engine.generate_text(
            _build_prompt(numbered_text, expected_questions),
            max_new_tokens=max_new_tokens,
            is_json=True,
        )
    except Exception:
        logger.exception("LLM segmentation generation failed; falling back to regex segmentation.")
        return _regex_segment_document(text)

    parsed = _extract_json_array(raw_output)
    if parsed is None:
        logger.warning(
            "LLM segmentation output was not valid JSON (first 200 chars: %r); "
            "falling back to regex segmentation.",
            raw_output[:200],
        )
        return _regex_segment_document(text)

    boundaries = _normalize_boundaries(parsed, len(lines))
    if boundaries is None:
        logger.warning("LLM segmentation output failed validation; falling back to regex segmentation.")
        return _regex_segment_document(text)

    if expected_questions:
        boundaries = _reconcile_with_expected(boundaries, expected_questions)
        if boundaries is None:
            return _regex_segment_document(text)

    segments = _slice_segments(boundaries, lines)
    if not segments:
        logger.warning("LLM segmentation produced no usable segments; falling back to regex segmentation.")
        return _regex_segment_document(text)

    logger.info("LLM segmentation succeeded: %d segments.", len(segments))
    return segments