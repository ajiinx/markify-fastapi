from __future__ import annotations

import io
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import List, Literal, Optional, Union

import pymupdf
import pymupdf4llm

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pydantic import BaseModel, Field

from PIL import Image

from olmocr_grading.config import OCRConfig, load_config
from olmocr_grading.image_utils import (
    SUPPORTED_IMAGE_EXTS,
    preprocess_image,
    resize_to_longest_dim,
)
from olmocr_grading.pdf_utils import pdf_to_images

from model_client import ModelServerError, OCREngineClient, PageResult

from segmentation import segment_document
from llm_segmentation import segment_document_llm
from llm_grading import grade_segments_llm, compute_totals
from llm_marking_scheme import (
    detect_stated_total_marks,
    extract_marking_scheme_llm,
)

import db

from db import (
    MODEL_ANSWERS_COLLECTION,
    LATEST_MODELS_LIMIT,
    get_documents,
)


# ============================================================
# Logging
# ============================================================

logger = logging.getLogger("autoassess")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="AutoAssess PDF Processing API",
    description=(
        "Uploads a PDF/image, extracts its text (PyMuPDF, falling back to "
        "olmOCR when the PDF has little/no embedded text), then either "
        "parses it as a marking scheme / model answer, or segments it "
        "into per-question sections for a scanned student answer sheet."
    ),
    version="0.1.0",
)


# ============================================================
# CORS
# ============================================================

origins = [
    "*",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# OCR CONFIGURATION
# ============================================================
#
# NOTE: model loading/device/dtype resolution used to live here, but
# that meant every `uvicorn --reload` restart reloaded the whole
# OlmOCREngine (and its CUDA VRAM) from scratch. That logic has moved
# to model_server.py, a separate long-lived process that owns the one
# OlmOCREngine instance. This process only keeps a plain OCRConfig for
# CPU-side preprocessing options (denoise/deskew/enhance_contrast,
# target_longest_dim, etc.) and talks to the model server through
# OCR_ENGINE, a lightweight HTTP client with the same
# transcribe()/generate_text() interface OlmOCREngine has.

OCR_CFG: OCRConfig = load_config(None)
OCR_ENGINE: OCREngineClient = OCREngineClient()


# ============================================================
# MODEL SERVER CONNECTIVITY CHECK
# ============================================================

@app.on_event("startup")
def check_model_server():
    """Doesn't load a model (that already happened in model_server.py) -
    just confirms the model server is up and reachable so failures are
    caught at FastAPI startup instead of on the first request."""

    logger.info(
        "Checking model server at %s ...",
        OCR_ENGINE.base_url,
    )

    try:
        OCR_ENGINE.load()
    except ModelServerError as e:
        logger.warning(
            "Model server not reachable/ready yet (%s). "
            "OCR/LLM-dependent endpoints will fail until it is up.",
            e,
        )
    else:
        info = OCR_ENGINE.health()
        logger.info(
            "Model server ready. Loaded models: %s",
            info.get("models", []),
        )


# ============================================================
# MONGODB LIFECYCLE
# ============================================================

@app.on_event("startup")
def connect_to_mongo():
    db.connect()


@app.on_event("shutdown")
def disconnect_from_mongo():
    db.close()


# ============================================================
# PDF → MARKDOWN
# ============================================================

def pdf_to_markdown(contents: bytes) -> str:
    """
    Convert PDF bytes directly to Markdown.

    No temporary file is used here.
    """

    doc = pymupdf.open(
        stream=contents,
        filetype="pdf",
    )

    try:
        return pymupdf4llm.to_markdown(doc)
    finally:
        doc.close()


# ============================================================
# MARKDOWN CLEANING
# ============================================================

def clean_markdown(markdown: str) -> str:
    """
    Remove common PDF extraction artifacts.
    """

    markdown = re.sub(
        r"\*{3}\s*\n\s*Page\s+\d+\s*\n\s*\*{3}",
        "",
        markdown,
        flags=re.IGNORECASE,
    )

    markdown = re.sub(
        r"\*+\s*-\s*o\s*O\s*o\s*-\s*\*+",
        "",
        markdown,
        flags=re.IGNORECASE,
    )

    markdown = re.sub(
        r"\n{3,}",
        "\n\n",
        markdown,
    )

    return markdown.strip()


# ============================================================
# OCR HELPERS
# ============================================================

def load_pages_from_bytes(
    data: bytes,
    suffix: str,
    cfg: OCRConfig,
) -> List[Image.Image]:

    suffix = suffix.lower()

    if suffix == ".pdf":
        tmp_path = None

        try:
            with tempfile.NamedTemporaryFile(
                suffix=".pdf",
                delete=False,
            ) as tmp:
                tmp.write(data)
                tmp_path = tmp.name

            images = pdf_to_images(
                tmp_path,
                target_longest_dim=cfg.target_longest_dim,
                render_dpi=cfg.pdf_render_dpi,
            )

            return images

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    if suffix in SUPPORTED_IMAGE_EXTS:
        image = Image.open(
            io.BytesIO(data)
        ).convert("RGB")

        image = resize_to_longest_dim(
            image,
            cfg.target_longest_dim,
        )

        images = [image]

    else:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            f"Expected PDF or {sorted(SUPPORTED_IMAGE_EXTS)}."
        )

    if cfg.preprocess:
        images = [
            preprocess_image(
                image,
                deskew=cfg.deskew,
                denoise=cfg.denoise,
                enhance_contrast=cfg.enhance_contrast,
            )
            for image in images
        ]

    return images


def run_ocr(
    images: List[Image.Image],
    engine: OCREngineClient,
) -> List[PageResult]:

    results: List[PageResult] = []

    for page_number, image in enumerate(images, start=1):
        try:
            result = engine.transcribe(
                image,
                page_number=page_number,
            )

        except Exception:
            logger.exception(
                "OCR failed on page %d",
                page_number,
            )

            result = PageResult(
                page=page_number,
                text="[UNCLEAR]",
                confidence=None,
            )

        results.append(result)

    return results


def ocr_document(
    contents: bytes,
    filename: str,
    preprocess: bool = False,
):
    """
    Run the complete OCR pipeline and combine all pages
    into one document.
    """

    if not OCR_ENGINE.is_ready():
        raise RuntimeError(
            "Model server is not reachable/ready. Start it with "
            "`python model_server.py` and try again."
        )

    suffix = Path(filename).suffix.lower()

    req_cfg = (
        OCR_CFG.copy()
        if hasattr(OCR_CFG, "copy")
        else OCR_CFG
    )

    req_cfg.preprocess = preprocess

    pages = load_pages_from_bytes(
        contents,
        suffix,
        req_cfg,
    )

    start = time.time()

    results = run_ocr(
        pages,
        OCR_ENGINE,
    )

    elapsed = time.time() - start

    text = "\n\n".join(
        result.text
        for result in results
    )

    return {
        "text": text,
        "num_pages": len(pages),
        "elapsed_seconds": round(elapsed, 2),
        "seconds_per_page": round(
            elapsed / max(len(pages), 1),
            2,
        ),
    }


# ============================================================
# DECIDE WHETHER OCR IS NECESSARY
# ============================================================

def should_use_ocr(markdown: str) -> bool:
    """
    Use OCR when PyMuPDF produced little meaningful text.
    """

    if not markdown:
        return True

    meaningful_text = re.sub(
        r"[\s#*_|`>\-]+",
        " ",
        markdown,
    ).strip()

    return len(meaningful_text) < 100


# ============================================================
# MODEL ANSWER DETECTION (safety net — not routing logic)
# ============================================================
#
# /markdown and /scanned no longer pick their own behavior based on
# this; the endpoint you call decides that. It's used only to reject
# uploads that look like the wrong document type for the route, so a
# scanned sheet on /markdown (or a marking scheme on /scanned) fails
# loudly instead of silently producing an empty/garbled result.

def is_model_answer_format(markdown: str) -> bool:
    text = markdown.upper()

    strong_indicators = [
        "MARKING SCHEME",
        "EXPECTED OUTCOMES",
        "VALUE POINTS",
    ]

    weak_indicators = [
        "MARKS",
        "Q.N",
        "QUESTION",
        "ANSWER",
    ]

    strong_matches = sum(
        indicator in text
        for indicator in strong_indicators
    )

    weak_matches = sum(
        indicator in text
        for indicator in weak_indicators
    )

    return (
        strong_matches >= 1
        and weak_matches >= 1
    )


# ============================================================
# PROCESS MARKING SCHEME
# ============================================================

def _clean_cell_text(text: str) -> str:
    """Collapse a table cell's <br> line breaks and extra whitespace into
    normal prose."""

    text = text.replace("<br>", " ")
    text = text.replace("<br/>", " ")
    text = text.replace("<br />", " ")

    return re.sub(r"\s+", " ", text).strip()


def process_marking_scheme(markdown: str):
    """
    Parse the pymupdf4llm markdown table produced from a model-answer /
    marking-scheme PDF into per-question value points.

    pymupdf4llm renders each table row as a single "|col|col|col|" line,
    but a question's own statement, its explanation, and its value
    points frequently all land in ONE cell (joined by "<br>"), because
    the source PDF has them stacked in the same table cell. The FIRST
    "<br>" inside that cell is what separates the question's statement
    from the marking content below it; any further "<br>"s inside are
    just PDF line-wrap artifacts, not additional boundaries, so only the
    first split is meaningful.

    A question's statement can also land on its own row with an empty
    marks column - sometimes the row immediately preceding the row that
    carries the question number (a PDF table-reconstruction quirk near
    page/column breaks). Such markless rows are held in `pending_text`
    until we know who they belong to: the next question that starts, or
    (if a scored row for the still-open question follows instead) that
    row's own point text.

    The Marks column can likewise stack multiple values in one cell
    (e.g. "2<br>3") when several value points' rows got squashed
    together - every value found in the cell is SUMMED into that row's
    single `marks` total (never just the first one), so a question's
    marks always add up to its true total even when its value points
    aren't individually broken out row-by-row.
    """

    lines = markdown.splitlines()

    questions: list[dict] = []
    current_question = None
    pending_text = ""

    for line in lines:
        line = line.strip()

        if not line.startswith("|"):
            continue

        # Split on "|" but only drop the OUTER empty cells produced by
        # the line's leading/trailing "|" delimiters. A naive "drop all
        # empty columns" (the previous approach) silently destroys
        # column alignment whenever a real cell (e.g. an empty Marks
        # column) is blank, e.g. "||some text||" incorrectly collapsing
        # to a single column instead of 3.
        columns = [column.strip() for column in line.split("|")]

        if columns and columns[0] == "":
            columns = columns[1:]

        if columns and columns[-1] == "":
            columns = columns[:-1]

        if len(columns) < 3:
            continue

        question_number = columns[0]
        answer_text = columns[1]
        marks_text = columns[2]

        if "Q.N" in question_number.upper():
            continue

        if "EXPECTED OUTCOMES" in answer_text.upper():
            continue

        if "MARKS" in marks_text.upper():
            continue

        if all(
            re.fullmatch(r"[-:]+", column)
            for column in columns[:3]
            if column
        ):
            continue

        # A Marks-column cell can stack MULTIPLE marks values (joined by
        # "<br>", e.g. "2<br>3" meaning two value points worth 2 and 3
        # marks that PDF table reconstruction squashed into one row's
        # cell) - every value found must be summed into this row's
        # total, not just the first. Silently keeping only the first
        # match (the previous behavior) drops real marks and produces
        # a wrong, too-low max_marks total for the question (and, once
        # graded, a wrong "total_max_marks"/"total_marks" for the
        # whole paper).
        marks_values = re.findall(r"\d+(?:½|\.5)?", marks_text)

        if marks_values:
            total_value = sum(
                float(value.replace("½", ".5")) for value in marks_values
            )
            marks = (
                str(int(total_value))
                if total_value.is_integer()
                else str(total_value)
            )
        else:
            marks = ""

        match = re.search(
            r"\d+",
            question_number,
        )

        if match:
            number = int(match.group())

            current_question = {
                "number": number,
                "question_text": pending_text,
                "points": [],
            }
            pending_text = ""

            questions.append(current_question)

            # The question statement and its first value point can be
            # bundled into this same row's cell, e.g.
            # "What is X? <br> Definition: ... <br> more line-wrap ...".
            # Only peel off the FIRST "<br>" segment as the question
            # statement; leave the rest as this row's point text.
            if not current_question["question_text"] and "<br>" in answer_text:
                question_part, _, rest = answer_text.partition("<br>")
                current_question["question_text"] = _clean_cell_text(question_part)
                answer_text = rest

        elif current_question is None:
            continue

        text = _clean_cell_text(answer_text)

        if not text:
            continue

        if not marks:
            # Nothing to score on this row - hold it rather than guess
            # whether it's this question's trailing text or the next
            # question's statement; resolved once we see what follows.
            pending_text = f"{pending_text} {text}".strip() if pending_text else text
            continue

        if pending_text:
            text = f"{pending_text} {text}".strip()
            pending_text = ""

        current_question["points"].append(
            {
                "question_text": current_question["question_text"],
                "text": text,
                "marks": marks,
            }
        )

    return questions

# ============================================================
# MARKING-SCHEME RESOLUTION (regex parse, LLM-verified)
# ============================================================

def _total_marks_for_points(points: list[dict]) -> float:
    """Sum a list of {..., "marks": "<str>"} value points into a plain
    float total. Non-numeric/empty marks are skipped rather than
    raising, so a partially-unparsed row doesn't blow up the total."""

    total = 0.0

    for point in points:
        marks = point.get("marks")

        if not marks:
            continue

        try:
            total += float(str(marks).replace("½", ".5"))
        except (ValueError, TypeError):
            continue

    return total


def resolve_marking_scheme(markdown: str, engine) -> list[dict]:
    """
    Produce the {number, question_text, points: [{text, marks}]} list
    for a model-answer/marking-scheme document, preferring whichever of
    the two extraction methods can actually be trusted:

    1. The regex table parser (process_marking_scheme) - fast, no LLM
       call needed. Trusted outright if its total matches the marking
       scheme's OWN stated grand total (see detect_stated_total_marks).
    2. LLM extraction (extract_marking_scheme_llm) - reads the whole
       document rather than row-by-row, so it isn't tripped up by the
       inconsistent table layouts that make the regex parser drop or
       misattribute marks. Used whenever the regex result can't be
       verified against the document's own stated total (either
       because it doesn't match, or because no such header total could
       be found at all - reading the whole document is safer than
       trusting single-row parsing blind).

    Falls back to the regex result if the LLM path is unavailable or
    its own output can't be validated either - always returns SOME
    result rather than raising, even if neither could be fully
    verified (logged either way).
    """

    regex_questions = process_marking_scheme(markdown)
    stated_total = detect_stated_total_marks(markdown)

    if stated_total is not None:
        regex_total = sum(
            _total_marks_for_points(q["points"]) for q in regex_questions
        )

        if abs(regex_total - stated_total) < 1e-6:
            logger.info(
                "Regex marking-scheme parse verified against document's "
                "stated total (%s marks); skipping LLM extraction.",
                stated_total,
            )
            return regex_questions

        logger.warning(
            "Regex marking-scheme parse totals %s but the document "
            "states %s; attempting LLM extraction instead.",
            regex_total,
            stated_total,
        )
    else:
        logger.info(
            "No stated total found in marking-scheme document; "
            "attempting LLM extraction to verify the regex parse."
        )

    llm_questions = extract_marking_scheme_llm(
        markdown,
        engine,
        stated_total=stated_total,
    )

    if llm_questions is not None:
        return llm_questions

    logger.warning(
        "LLM marking-scheme extraction unavailable/unverifiable; "
        "falling back to the regex parse (totals may be incorrect)."
    )
    return regex_questions


# ============================================================
# GENERATE MODEL ANSWER MARKDOWN
# ============================================================

def generate_final_markdown(questions):
    output = "# Model Answer Paper\n\n"

    for question in questions:
        number = question["number"]

        total_marks = _total_marks_for_points(question["points"])

        if total_marks.is_integer():
            total_marks = int(total_marks)

        output += (
            f"## Q{number}. Model Answer "
            f"[{total_marks} Marks]\n\n"
        )

        for index, point in enumerate(
            question["points"],
            start=1,
        ):
            text = point["text"]
            marks = point["marks"]

            if marks:
                output += (
                    f"### {index}. "
                    f"[{marks} Marks]\n\n"
                )
            else:
                output += f"### {index}.\n\n"

            output += f"{text}\n\n"

        output += "---\n\n"

    return output.strip()


# ============================================================
# SEGMENTATION REFERENCE (MODEL ANSWER)
# ============================================================

def get_reference_question_numbers(
    model_answer_id: Optional[str],
) -> Optional[List[int]]:
    """
    Look up the question-number sequence from a model-answer document in
    MongoDB, for use as a trusted reference during LLM segmentation.

    If `model_answer_id` is given, that specific document is used. If
    not, we best-effort fall back to the most recently saved model
    answer - useful for a single-question-paper workflow, but callers
    grading multiple different papers concurrently should pass an
    explicit model_answer_id to avoid picking up the wrong reference.

    Returns None (never raises) if no usable reference is found, so the
    caller can proceed without one.
    """

    doc = None

    if model_answer_id:
        doc = db.get_document(model_answer_id, db.MODEL_ANSWERS_COLLECTION)
        if doc is None:
            logger.warning(
                "model_answer_id=%s not found in '%s'; "
                "continuing without a segmentation reference.",
                model_answer_id,
                db.MODEL_ANSWERS_COLLECTION,
            )
    else:
        doc = db.get_latest_document(
            db.MODEL_ANSWERS_COLLECTION,
            {"type": "model_answer"},
        )
        if doc is not None:
            logger.info(
                "No model_answer_id provided; using most recent model "
                "answer (document_id=%s) as segmentation reference.",
                doc.get("_id"),
            )

    if not doc:
        return None

    questions = doc.get("questions")
    if not isinstance(questions, list) or not questions:
        return None

    numbers = sorted(
        {
            q.get("number")
            for q in questions
            if isinstance(q, dict) and isinstance(q.get("number"), int)
        }
    )

    return numbers or None


# ============================================================
# GRADING REFERENCE (MODEL ANSWER)
# ============================================================

def get_model_answer_reference_questions(
    model_answer_id: Optional[str],
) -> Optional[List[dict]]:
    """
    Look up the FULL per-question marking-scheme data (question_text,
    value points, and each point's marks) from a model-answer document
    in MongoDB, for use as the grading reference by llm_grading.

    Same resolution rules as get_reference_question_numbers: an
    explicit `model_answer_id` is used if given, otherwise the most
    recently saved model-answer document is used as a best-effort
    fallback.

    Returns None (never raises) if no usable reference is found, so
    the caller can proceed with every segment left ungraded.
    """

    doc = None

    if model_answer_id:
        doc = db.get_document(model_answer_id, db.MODEL_ANSWERS_COLLECTION)
        if doc is None:
            logger.warning(
                "model_answer_id=%s not found in '%s'; "
                "continuing without a grading reference.",
                model_answer_id,
                db.MODEL_ANSWERS_COLLECTION,
            )
    else:
        doc = db.get_latest_document(
            db.MODEL_ANSWERS_COLLECTION,
            {"type": "model_answer"},
        )
        if doc is not None:
            logger.info(
                "No model_answer_id provided; using most recent model "
                "answer (document_id=%s) as grading reference.",
                doc.get("_id"),
            )

    if not doc:
        return None

    questions = doc.get("questions")
    if not isinstance(questions, list) or not questions:
        return None

    return questions


# ============================================================
# API SCHEMA MODELS
# ============================================================
#
# These models exist purely to document the actual response shapes the
# handlers below already build as plain dicts - none of the business
# logic, field names, or values were changed to produce them. See the
# accompanying write-up for exactly which fields are optional/nullable
# and why (e.g. `ocr` is null when the fast PyMuPDF path was used and
# OCR never ran; `document_id` is null if the MongoDB write failed).

class MarkingSchemePoint(BaseModel):
    """One row/value-point of a marking scheme question."""

    question_text: str = Field(
        "",
        description="The question's own statement, as extracted from "
        "the model-answer table above its value points (whitespace-"
        "normalized). Repeated on every point of the same question for "
        "convenience. Empty string '' if the table didn't contain a "
        "separate question-statement row for this question.",
    )
    text: str = Field(
        ...,
        description="Marking-scheme row text (whitespace-normalized, "
        "<br> tags converted to spaces).",
    )
    marks: str = Field(
        ...,
        description="Marks for this point, as extracted text - e.g. "
        "'5', '2.5'. Sum of every marks value found in the row's "
        "Marks-column cell (a cell can stack more than one, e.g. "
        "'2<br>3' -> '5'). Empty string '' if no marks value could be "
        "parsed from the row. NOTE: kept as a string (not a number) "
        "for consistency with how it's parsed/summed, not because it "
        "needs non-numeric characters.",
    )


class MarkingSchemeQuestion(BaseModel):
    """One question parsed out of a marking-scheme/model-answer table."""

    number: int = Field(
        ...,
        description="Question number, parsed from the first table "
        "column (e.g. 'Q1' -> 1).",
    )
    points: List[MarkingSchemePoint] = Field(
        default_factory=list,
        description="Ordered marking-scheme rows belonging to this "
        "question.",
    )


class OcrMetadata(BaseModel):
    """Present only when the olmOCR fallback path actually ran."""

    device: Optional[str] = Field(
        None, description="Device OCR inference ran on (if available)."
    )
    dtype: Optional[str] = Field(
        None, description="Torch dtype used for OCR inference (if available)."
    )
    num_pages: int = Field(..., description="Number of pages/images OCR'd.")
    elapsed_seconds: float = Field(
        ..., description="Total OCR wall-clock time, rounded to 2dp."
    )
    seconds_per_page: float = Field(
        ..., description="Average OCR time per page, rounded to 2dp."
    )


class NoFileSelectedResponse(BaseModel):
    """
    Returned when the `myFile` multipart part was sent with an empty
    filename. NOTE: this is currently returned with HTTP 200, not 400 -
    preserved as-is from the existing behavior. See write-up for why
    this is worth reconsidering.
    """

    success: Literal[False]
    message: Literal["No file selected."]


class ModelAnswerFileResponse(BaseModel):
    """
    Returned from POST /markdown for a parsed marking scheme / model
    answer.
    """

    success: Literal[True]
    filename: str = Field(..., description="Original uploaded filename.")
    type: Literal["model_answer"]
    extraction_method: Literal["pymupdf", "olmocr"] = Field(
        ..., description="Which extraction path produced the markdown."
    )
    used_ocr: bool = Field(..., description="Whether the olmOCR fallback ran.")
    markdown: str = Field(
        ...,
        description="Regenerated '# Model Answer Paper' markdown built "
        "from the parsed table (NOT the raw extracted markdown).",
    )
    questions: List[MarkingSchemeQuestion]
    ocr: Optional[OcrMetadata] = Field(
        None, description="Null unless used_ocr is true."
    )
    document_id: Optional[str] = Field(
        None,
        description="MongoDB _id of the saved document, as a string. "
        "Null if the MongoDB write failed (checked server-side; the "
        "document is NOT persisted in that case). Pass this value back "
        "as `model_answer_id` on a later POST /scanned call for a "
        "student's scanned answer sheet against the same question "
        "paper.",
    )


class SegmentItem(BaseModel):
    """One question-wise slice of a segmented student answer sheet,
    graded against the model-answer reference (see `model_answer_id`
    on the request)."""

    question_id: str = Field(..., description="e.g. 'Q1', 'Q2'.")
    text: str = Field(
        ...,
        description="Exact (verbatim) slice of the source markdown "
        "belonging to this question.",
    )
    max_marks: Optional[float] = Field(
        None,
        description="Maximum marks obtainable for this question, "
        "summed from the model-answer reference's value-point marks. "
        "Null if no model-answer reference question could be matched "
        "to this segment.",
    )
    marks_assigned: Optional[float] = Field(
        None,
        description="Marks assigned by the LLM grader based on the "
        "student's answer vs. the model-answer reference. Null if "
        "this question could not be graded automatically (no "
        "reference matched, grading model unavailable, or the "
        "grading call failed/was unparsable) - see "
        "`evaluation_feedback` for why.",
    )
    evaluation_feedback: Optional[str] = Field(
        None,
        description="LLM-generated justification for the assigned "
        "marks (which value points were covered/missed), or - when "
        "`marks_assigned` is null - an explanation of why automatic "
        "grading could not be completed for this question.",
    )


class RawDocumentFileResponse(BaseModel):
    """
    Returned from POST /scanned for a segmented student answer sheet.
    """

    success: Literal[True]
    filename: str = Field(..., description="Original uploaded filename.")
    type: Literal["raw"]
    extraction_method: Literal["pymupdf", "olmocr"] = Field(
        ..., description="Which extraction path produced the markdown."
    )
    used_ocr: bool = Field(..., description="Whether the olmOCR fallback ran.")
    markdown: str = Field(
        ..., description="Full extracted/cleaned document markdown."
    )
    total_max_marks: Optional[float] = Field(
        None,
        description="Sum of `max_marks` across all gradable segments - "
        "the paper's true total, independent of how many questions "
        "actually got graded. Null if nothing could be graded.",
    )
    total_marks_assigned: Optional[float] = Field(
        None,
        description="Sum of `marks_assigned` across all gradable "
        "segments (an ungraded gradable segment counts as 0). Null "
        "under the same condition as `total_max_marks`.",
    )
    segments: List[SegmentItem] = Field(
        ..., description="Question-wise segmentation of `markdown`."
    )
    ocr: Optional[OcrMetadata] = Field(
        None, description="Null unless used_ocr is true."
    )
    segmentation_reference_questions: Optional[List[int]] = Field(
        None,
        description="Question numbers sourced from the model-answer "
        "reference (see `model_answer_id` on the request) and used to "
        "guide/validate LLM segmentation. Null if no reference document "
        "was found/used, in which case segmentation relied on the LLM's "
        "own read of the document (or the regex fallback).",
    )
    document_id: Optional[str] = Field(
        None,
        description="MongoDB _id of the saved document, as a string. "
        "Null if the MongoDB write failed.",
    )


# Unions used only for documentation/response-shape validation. FastAPI
# (Pydantic 'smart' union matching) picks the correct variant because
# each has a distinguishing required field/literal (`type` for the
# success shape, and the small required field set for the "no file"
# shape) - it does not merge or invent fields across variants.
MarkdownEndpointResponse = Union[
    ModelAnswerFileResponse,
    NoFileSelectedResponse,
]

ScannedEndpointResponse = Union[
    RawDocumentFileResponse,
    NoFileSelectedResponse,
]


class ErrorResponse(BaseModel):
    """Shape of FastAPI's default HTTPException JSON body."""

    detail: str = Field(..., description="Human-readable error message.")


class HealthResponse(BaseModel):
    """Response of GET /health."""

    status: Literal["ok", "loading"] = Field(
        ..., description="'ok' once the OCR model has finished loading."
    )
    device: Optional[str] = Field(
        None,
        description="Null if the model server is unreachable "
        "or if using vLLM which does not report this.",
    )
    dtype: Optional[str] = Field(
        None,
        description="Null if the model server is unreachable "
        "or if using vLLM which does not report this.",
    )
    gpu: Optional[str] = Field(
        None, description="GPU name. Present only when device == 'cuda'."
    )
    cuda_version: Optional[str] = Field(
        None, description="Present only when device == 'cuda'."
    )
    compute_capability: Optional[str] = Field(
        None,
        description="e.g. '8.0'. Present only when device == 'cuda'.",
    )
    gpu_memory_allocated_gb: Optional[float] = Field(
        None,
        description="Rounded to 2dp. Present only when device == 'cuda'.",
    )
    gpu_memory_reserved_gb: Optional[float] = Field(
        None,
        description="Rounded to 2dp. Present only when device == 'cuda'.",
    )


# ============================================================
# SHARED UPLOAD VALIDATION + EXTRACTION PIPELINE
# ============================================================
#
# Common to both /markdown and /scanned: validate the upload, read it,
# run PyMuPDF -> Markdown, fall back to olmOCR when needed. Returns
# None when no file was selected (caller returns the "No file
# selected." shape); otherwise a dict with the extracted/cleaned
# markdown plus extraction metadata.

async def extract_markdown_from_upload(
    myFile: UploadFile,
    preprocess: bool = False,
) -> Optional[dict]:
    # --------------------------------------------------------
    # Validate filename
    # --------------------------------------------------------

    if not myFile.filename:
        return None

    filename = myFile.filename
    suffix = Path(filename).suffix.lower()

    # --------------------------------------------------------
    # Validate file type
    # --------------------------------------------------------

    supported_files = {
        ".pdf"
    } | set(SUPPORTED_IMAGE_EXTS)

    if suffix not in supported_files:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{suffix}'. "
                f"Supported types: "
                f"{sorted(supported_files)}"
            ),
        )

    # --------------------------------------------------------
    # Read uploaded file
    # --------------------------------------------------------

    contents = await myFile.read()

    logger.info(
        "Received file=%s type=%s size=%d bytes",
        filename,
        myFile.content_type,
        len(contents),
    )

    # ========================================================
    # STEP 1: Normal PDF → Markdown
    # ========================================================

    markdown = ""
    extraction_method = "pymupdf"
    used_ocr = False
    ocr_metadata = None

    if suffix == ".pdf":
        try:
            markdown = pdf_to_markdown(contents)
            markdown = clean_markdown(markdown)

        except Exception:
            logger.exception(
                "PyMuPDF extraction failed."
            )
            markdown = ""

    # ========================================================
    # STEP 2: OCR fallback
    # ========================================================

    if should_use_ocr(markdown):
        logger.info(
            "Insufficient PDF text detected. "
            "Falling back to olmOCR."
        )

        try:
            ocr_result = ocr_document(
                contents,
                filename,
                preprocess=preprocess,
            )

            markdown = clean_markdown(
                ocr_result["text"]
            )

            extraction_method = "olmocr"
            used_ocr = True

            model_info = OCR_ENGINE.health()

            ocr_metadata = {
                "device": model_info.get("device"),
                "dtype": model_info.get("dtype"),
                "num_pages": ocr_result["num_pages"],
                "elapsed_seconds": (
                    ocr_result["elapsed_seconds"]
                ),
                "seconds_per_page": (
                    ocr_result["seconds_per_page"]
                ),
            }

        except Exception as e:
            logger.exception("OCR failed.")

            raise HTTPException(
                status_code=500,
                detail=f"OCR failed: {str(e)}",
            )

    return {
        "filename": filename,
        "markdown": markdown,
        "extraction_method": extraction_method,
        "used_ocr": used_ocr,
        "ocr_metadata": ocr_metadata,
    }


# ============================================================
# /markdown — MODEL ANSWER PDF → MARKDOWN
# ============================================================

@app.post(
    "/markdown",
    response_model=MarkdownEndpointResponse,
    summary="Upload a model-answer / marking-scheme PDF for parsing",
    response_description="Model-answer parse result, or a 'no file "
    "selected' notice - see the two response schemas.",
    tags=["Files"],
    responses={
        400: {
            "model": ErrorResponse,
            "description": "Unsupported file extension, or the "
            "uploaded document doesn't look like a model answer / "
            "marking scheme.",
        },
        500: {
            "model": ErrorResponse,
            "description": "OCR failed unexpectedly.",
        },
    },
)
async def accept_model_answer(
    myFile: UploadFile = File(
        ...,
        description="The model-answer/marking-scheme PDF (or image) "
        "to parse. Supported extensions: .pdf plus whatever "
        "SUPPORTED_IMAGE_EXTS declares (see olmocr_grading.image_utils).",
    ),
    preprocess: bool = Form(
        False,
        description="If true, apply image preprocessing (deskew/"
        "denoise/contrast enhancement) before OCR. Only affects the "
        "olmOCR fallback path; ignored when PyMuPDF text extraction "
        "already succeeds.",
    ),
):
    extracted = await extract_markdown_from_upload(
        myFile,
        preprocess=preprocess,
    )

    if extracted is None:
        return {
            "success": False,
            "message": "No file selected.",
        }

    if not is_model_answer_format(extracted["markdown"]):
        raise HTTPException(
            status_code=400,
            detail=(
                "Uploaded file doesn't look like a model answer / "
                "marking scheme (no 'MARKING SCHEME' / 'EXPECTED "
                "OUTCOMES' / 'VALUE POINTS' indicators found). If "
                "this is a scanned student answer sheet, use "
                "/scanned instead."
            ),
        )

    logger.info("Parsing as Model Answer / Marking Scheme")

    questions = resolve_marking_scheme(
        extracted["markdown"],
        OCR_ENGINE,
    )

    final_markdown = generate_final_markdown(
        questions
    )

    response = {
        "success": True,
        "filename": extracted["filename"],
        "type": "model_answer",
        "extraction_method": extracted["extraction_method"],
        "used_ocr": extracted["used_ocr"],
        "markdown": final_markdown,
        "questions": questions,
        "ocr": extracted["ocr_metadata"],
    }

    document_id = db.save_document(
        response,
        db.SCANNED_DOCUMENTS_COLLECTION if extracted["used_ocr"] else db.MODEL_ANSWERS_COLLECTION,
    )
    response["document_id"] = document_id

    return response


# ============================================================
# /scanned — SCANNED ANSWER SHEET → QUESTION SEGMENTATION
# ============================================================

@app.post(
    "/scanned",
    response_model=ScannedEndpointResponse,
    summary="Upload a scanned student answer sheet for OCR + segmentation",
    response_description="Segmented-document result, or a 'no file "
    "selected' notice - see the two response schemas.",
    tags=["Files"],
    responses={
        400: {
            "model": ErrorResponse,
            "description": "Unsupported file extension, or the "
            "uploaded document looks like a model answer / marking "
            "scheme rather than a scanned sheet.",
        },
        500: {
            "model": ErrorResponse,
            "description": "OCR or segmentation failed unexpectedly.",
        },
    },
)
async def accept_scanned_sheet(
    myFile: UploadFile = File(
        ...,
        description="The student's scanned answer sheet to process. "
        "Supported extensions: .pdf plus whatever SUPPORTED_IMAGE_EXTS "
        "declares (see olmocr_grading.image_utils).",
    ),
    preprocess: bool = Form(
        False,
        description="If true, apply image preprocessing (deskew/"
        "denoise/contrast enhancement) before OCR. Only affects the "
        "olmOCR fallback path; ignored when PyMuPDF text extraction "
        "already succeeds.",
    ),
    model_answer_id: Optional[str] = Form(
        None,
        description="MongoDB _id (the `document_id` returned from a "
        "prior /markdown upload) of the model-answer document to use "
        "as a segmentation reference. If omitted, the most recently "
        "saved model-answer document is used as a best-effort "
        "fallback - pass this explicitly when multiple question papers "
        "may be in flight concurrently.",
    ),
):
    extracted = await extract_markdown_from_upload(
        myFile,
        preprocess=preprocess,
    )

    if extracted is None:
        return {
            "success": False,
            "message": "No file selected.",
        }

    markdown = extracted["markdown"]

    if is_model_answer_format(markdown):
        raise HTTPException(
            status_code=400,
            detail=(
                "Uploaded file looks like a model answer / marking "
                "scheme, not a scanned student answer sheet. Use "
                "/markdown instead."
            ),
        )

    logger.info("Segmenting as scanned answer sheet")

    try:
        expected_question_numbers = get_reference_question_numbers(model_answer_id)
    except Exception:
        logger.exception(
            "Failed to load model-answer segmentation reference; "
            "continuing without it."
        )
        expected_question_numbers = None

    if expected_question_numbers:
        logger.info(
            "Using model-answer reference question numbers for "
            "segmentation: %s",
            expected_question_numbers,
        )

    try:
        segments = segment_document_llm(
            markdown,
            OCR_ENGINE,
            expected_questions=expected_question_numbers,
        )

    except Exception as e:
        logger.exception(
            "LLM segmentation raised unexpectedly; "
            "falling back to regex segmentation."
        )

        try:
            segments = segment_document(markdown)
        except Exception as fallback_e:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Segmentation failed: {str(fallback_e)}"
                ),
            )

    # ========================================================
    # LLM GRADING (against the model-answer reference)
    # ========================================================

    try:
        reference_questions = get_model_answer_reference_questions(model_answer_id)
    except Exception:
        logger.exception(
            "Failed to load model-answer grading reference; "
            "continuing with segments ungraded."
        )
        reference_questions = None

    try:
        segments = grade_segments_llm(
            segments,
            reference_questions,
            OCR_ENGINE,
        )
    except Exception:
        logger.exception(
            "LLM grading raised unexpectedly; returning segments ungraded."
        )

    total_marks, total_max_marks = compute_totals(segments)

    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    response = {
        "success": True,
        "filename": extracted["filename"],
        "type": "raw",
        "extraction_method": extracted["extraction_method"],
        "used_ocr": extracted["used_ocr"],
        "markdown": markdown,
        "total_max_marks": total_max_marks,
        "total_marks_assigned": total_marks,
        "segments": segments,
        "ocr": extracted["ocr_metadata"],
        "segmentation_reference_questions": expected_question_numbers,
    }

    document_id = db.save_document(
        response,
        db.SCANNED_DOCUMENTS_COLLECTION if extracted["used_ocr"] else db.MODEL_ANSWERS_COLLECTION,
    )
    response["document_id"] = document_id

    return response


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    response_model_exclude_none=True,
    summary="Health check",
    description="Reports the MODEL SERVER's engine load status plus "
    "device/dtype (proxied from model_server.py, which is the process "
    "that actually holds the model in CUDA VRAM). The gpu/cuda_version/"
    "compute_capability/gpu_memory_* fields are included only when "
    "running on CUDA (they are OMITTED from the JSON entirely on CPU, "
    "not sent as null - response_model_exclude_none preserves that). "
    "If the model server itself is unreachable, status is 'loading'.",
    response_description="Current OCR engine health/device status.",
    tags=["Health"],
)
def health():
    info = OCR_ENGINE.health()

    status = "ok" if info.get("status") == "ok" else "loading"

    response = {
        "status": status,
        "device": info.get("device"),
        "dtype": info.get("dtype"),
    }

    for key in (
        "gpu",
        "cuda_version",
        "compute_capability",
        "gpu_memory_allocated_gb",
        "gpu_memory_reserved_gb",
    ):
        if key in info:
            response[key] = info[key]

    return response


@app.get("/get_latest_model")
def get_latest_model():
    documents = get_documents(
        MODEL_ANSWERS_COLLECTION,
        {"type": "model_answer"},
        limit=5,
    )

    if not documents:
        raise HTTPException(
            status_code=404,
            detail="No model answers found",
        )

    return [
        {
            "model_id": document["_id"],
            "name": document.get("filename"),
        }
        for document in documents
    ]