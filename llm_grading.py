"""
LLM-based answer grading.

Uses the already-loaded olmOCR / Qwen2.5-VL model (see olmocr_grading.ocr_engine),
in the same pure text-in/text-out mode as llm_segmentation.py, to grade each
question-wise student segment (produced by segment_document_llm /
segment_document) against the corresponding question's value points from the
model-answer / marking-scheme document already stored in MongoDB.

Input:
    segments: [{"question_id": "Q1", "text": "..."}, ...]
        (the output of segmentation.segment_document / llm_segmentation.segment_document_llm)

    reference_questions: the "questions" list stored on a model-answer MongoDB
        document (main.process_marking_scheme's output), i.e.
        [{"number": 1, "points": [{"question_text": "...", "text": "...", "marks": "5"}, ...]}, ...]
        May be None if no model-answer reference could be resolved.

Output - same segments, each enriched with three fields:

    [
        {
            "question_id": "Q1",
            "text": "...",
            "max_marks": 5.0,
            "marks_assigned": 3.5,
            "evaluation_feedback": "...",
        },
        ...
    ]

Safety net: exactly like llm_segmentation, every LLM call is validated before
being trusted. If the model/engine is unavailable, a question has no matching
reference, the call errors out, or the output isn't valid/parseable JSON, that
segment is returned with max_marks/marks_assigned left as None (or max_marks
only, if that much could still be computed) and evaluation_feedback explaining
why grading could not be completed - grading failures never drop or corrupt
the segment's original question_id/text.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import re
from typing import Any, Optional

logger = logging.getLogger("autoassess.llm_grading")

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_QID_RE = re.compile(r"^Q?\s*(\d+)$", re.IGNORECASE)

_NO_REFERENCE_FEEDBACK = (
    "No model-answer reference was found for this question, so it "
    "could not be graded automatically."
)
_ENGINE_UNAVAILABLE_FEEDBACK = (
    "The grading model is not currently loaded, so this answer could "
    "not be graded automatically."
)
_GENERATION_FAILED_FEEDBACK = (
    "Automated grading failed unexpectedly for this question. It has "
    "not been scored and should be reviewed manually."
)
_UNPARSEABLE_FEEDBACK = (
    "The grading model's response could not be parsed. This answer "
    "has not been scored and should be reviewed manually."
)

_INSTRUCTIONS = """You are a strict but fair examiner grading a student's answer against an official model answer / marking scheme.

You will be given:
1. The question.
2. The official marking-scheme value points for this question, each with the marks it is worth.
3. The student's answer, exactly as OCR'd/transcribed from their scanned answer sheet (it may contain OCR noise, spelling mistakes, or handwriting-transcription errors - judge the underlying content, not superficial transcription artifacts).

Grade ONLY against the value points given below. Award full marks for a value point only if the student's answer clearly and correctly covers it. Award partial marks if the student's answer partially/vaguely covers a value point. Award zero for a value point the student's answer omits or gets wrong. Do not award marks for content the marking scheme does not credit, however correct or impressive it may be.

Output ONLY a single JSON object and NOTHING else (no markdown fences, no commentary, no explanation outside the JSON):
{"analysis": "<step-by-step reasoning>", "marks_assigned": <number>, "evaluation_feedback": "<short justification>"}

Strict rules:
1. "analysis" must briefly map the student's answer to the value points to reason about the final score.
2. "marks_assigned" must be a number (integer or decimal, e.g. 3 or 2.5) between 0 and {max_marks} inclusive - never negative, never above {max_marks}.
3. "evaluation_feedback" must be a short (1-3 sentence) justification summarizing the analysis.
4. Never invent value points that aren't in the marking scheme below.

=== QUESTION ===
{question_text}

=== MARKING SCHEME (max {max_marks} marks) ===
{reference_points}

=== STUDENT'S ANSWER ===
{student_answer}

JSON object:"""


def _sum_marks(points: list[dict]) -> Optional[float]:
    """Sum a marking-scheme question's per-point marks into a max-marks
    total, mirroring main.generate_final_markdown's own total-marks
    logic (handles '½' as well as plain/decimal digit strings)."""

    total = 0.0
    saw_any = False

    for point in points:
        if not isinstance(point, dict):
            continue

        marks = point.get("marks")

        if not marks:
            continue

        try:
            value = float(str(marks).replace("½", ".5"))
        except (ValueError, TypeError):
            continue

        total += value
        saw_any = True

    if not saw_any:
        return None

    return int(total) if total.is_integer() else total


def _format_reference_points(points: list[dict]) -> str:
    lines = []

    for index, point in enumerate(points, start=1):
        if not isinstance(point, dict):
            continue

        text = (point.get("text") or "").strip()
        marks = point.get("marks") or "?"

        if not text:
            continue

        lines.append(f"{index}. [{marks} marks] {text}")

    return "\n".join(lines) if lines else "(no value points available)"


def _question_number_from_id(question_id: str) -> Optional[int]:
    match = _QID_RE.match((question_id or "").strip())
    return int(match.group(1)) if match else None


def _index_reference_questions(
    reference_questions: Optional[list[dict]],
) -> dict[int, dict]:
    if not reference_questions:
        return {}

    indexed: dict[int, dict] = {}

    for question in reference_questions:
        if not isinstance(question, dict):
            continue

        number = question.get("number")

        if isinstance(number, int):
            indexed[number] = question

    return indexed


def _build_prompt(
    question_text: str,
    reference_points: str,
    max_marks: float,
    student_answer: str,
) -> str:
    prompt = _INSTRUCTIONS

    # max_marks appears twice in the template (rule 1 and the section
    # header) - format() would choke on the JSON example's braces, so
    # plain .replace() is used instead (same approach as llm_segmentation).
    prompt = prompt.replace("{max_marks}", str(max_marks))
    prompt = prompt.replace("{question_text}", question_text or "(question text unavailable)")
    prompt = prompt.replace("{reference_points}", reference_points)
    prompt = prompt.replace("{student_answer}", student_answer or "(no answer text extracted)")

    return prompt


def _extract_json_object(raw: str) -> Optional[dict]:
    cleaned = _FENCE_RE.sub("", raw.strip()).strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = _JSON_OBJECT_RE.search(cleaned)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return None

    return None


def _normalize_grading_result(
    parsed: dict,
    max_marks: float,
) -> Optional[tuple[float, str]]:
    marks_assigned = parsed.get("marks_assigned")
    feedback = parsed.get("evaluation_feedback")

    if isinstance(marks_assigned, bool) or not isinstance(marks_assigned, (int, float)):
        return None

    if not isinstance(feedback, str) or not feedback.strip():
        return None

    # Clamp rather than reject an out-of-range score - handwritten-digit
    # style slips (e.g. the model saying 6 out of 5) shouldn't discard an
    # otherwise-usable justification.
    marks_assigned = max(0.0, min(float(marks_assigned), float(max_marks)))

    if float(marks_assigned).is_integer():
        marks_assigned = int(marks_assigned)

    return marks_assigned, feedback.strip()


def _grade_one_segment(
    segment: dict,
    reference_question: Optional[dict],
    engine,
    max_new_tokens: int,
) -> dict:
    graded = dict(segment)

    if reference_question is None:
        graded["max_marks"] = None
        graded["marks_assigned"] = None
        graded["evaluation_feedback"] = _NO_REFERENCE_FEEDBACK
        return graded

    points = reference_question.get("points") or []
    max_marks = _sum_marks(points)

    graded["max_marks"] = max_marks

    if max_marks is None:
        graded["marks_assigned"] = None
        graded["evaluation_feedback"] = _NO_REFERENCE_FEEDBACK
        return graded

    if engine is None or not engine.is_ready():
        graded["marks_assigned"] = None
        graded["evaluation_feedback"] = _ENGINE_UNAVAILABLE_FEEDBACK
        return graded

    question_text = ""
    for point in points:
        if isinstance(point, dict) and point.get("question_text"):
            question_text = point["question_text"]
            break

    prompt = _build_prompt(
        question_text=question_text,
        reference_points=_format_reference_points(points),
        max_marks=max_marks,
        student_answer=segment.get("text", ""),
    )

    try:
        raw_output = engine.generate_text(prompt, max_new_tokens=max_new_tokens, is_json=True)
    except Exception:
        logger.exception(
            "LLM grading generation failed for %s; leaving it ungraded.",
            segment.get("question_id"),
        )
        graded["marks_assigned"] = None
        graded["evaluation_feedback"] = _GENERATION_FAILED_FEEDBACK
        return graded

    parsed = _extract_json_object(raw_output)

    if parsed is None:
        logger.warning(
            "LLM grading output for %s was not valid JSON (first 200 chars: %r).",
            segment.get("question_id"),
            raw_output[:200],
        )
        graded["marks_assigned"] = None
        graded["evaluation_feedback"] = _UNPARSEABLE_FEEDBACK
        return graded

    normalized = _normalize_grading_result(parsed, max_marks)

    if normalized is None:
        logger.warning(
            "LLM grading output for %s failed validation: %r.",
            segment.get("question_id"),
            parsed,
        )
        graded["marks_assigned"] = None
        graded["evaluation_feedback"] = _UNPARSEABLE_FEEDBACK
        return graded

    marks_assigned, feedback = normalized
    graded["marks_assigned"] = marks_assigned
    graded["evaluation_feedback"] = feedback

    return graded


def compute_totals(
    segments: list[dict],
) -> tuple[Optional[float], Optional[float]]:
    """
    Sum the per-segment max_marks/marks_assigned (as set by
    grade_segments_llm) into paper-level totals.

    total_max_marks is the sum of every segment's max_marks that was
    actually resolved from the model-answer reference (i.e. the true
    total for the paper, regardless of whether every question could
    be graded). total_marks is the sum of marks_assigned for those
    SAME questions, counting an ungraded question (marks_assigned is
    None - engine unavailable, generation failed, etc.) as 0 rather
    than excluding it, so a partially-gradeable paper still reports an
    honest (if conservative) total instead of silently shrinking its
    own denominator.

    Returns (None, None) if no segment had a resolvable max_marks at
    all (e.g. no model-answer reference was found), so the caller can
    tell "nothing to grade" apart from "graded, scored zero".
    """

    gradable = [
        segment
        for segment in segments
        if isinstance(segment, dict)
        and isinstance(segment.get("max_marks"), (int, float))
        and not isinstance(segment.get("max_marks"), bool)
    ]

    if not gradable:
        return None, None

    total_max_marks = sum(segment["max_marks"] for segment in gradable)
    total_marks = sum(segment.get("marks_assigned") or 0 for segment in gradable)

    if float(total_max_marks).is_integer():
        total_max_marks = int(total_max_marks)

    if float(total_marks).is_integer():
        total_marks = int(total_marks)

    return total_marks, total_max_marks


def grade_segments_llm(
    segments: list[dict],
    reference_questions: Optional[list[dict]],
    engine,
    max_new_tokens: int = 512,
) -> list[dict]:
    """
    Grade each question-wise segment against the matching model-answer
    question's value points, using the pre-loaded Qwen2.5-VL / olmOCR
    model in text-only mode (now runs concurrently per segment).
    """

    if not segments:
        return []

    indexed_reference = _index_reference_questions(reference_questions)

    graded_segments = [None] * len(segments)  # type: ignore

    def _process(idx, segment):
        if not isinstance(segment, dict):
            graded_segments[idx] = segment
            return

        question_number = _question_number_from_id(segment.get("question_id", ""))
        reference_question = (
            indexed_reference.get(question_number)
            if question_number is not None
            else None
        )

        graded_segments[idx] = _grade_one_segment(
            segment,
            reference_question,
            engine,
            max_new_tokens,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, len(segments))) as executor:
        futures = [
            executor.submit(_process, idx, segment)
            for idx, segment in enumerate(segments)
        ]
        concurrent.futures.wait(futures)

    graded_count = sum(
        1 for s in graded_segments if isinstance(s, dict) and s.get("marks_assigned") is not None
    )
    logger.info(
        "LLM grading complete: %d/%d segments graded.",
        graded_count,
        len(graded_segments),
    )

    return graded_segments