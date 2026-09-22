from __future__ import annotations

import re

# ============================================================
# Primary marker pattern
# ============================================================
#
# Matches a question/answer marker sitting at the start of a line, e.g.:
#
#   Q1, Q.1, Q 1, Q1)
#   Question No 1, Question 2
#   A2, A2)                    <- OCR frequently misreads "Q" as "A"
#   Ans(1), ANS(2), Answer 3   <- students label answers, not questions
#   Ans - Q1                   <- combined answer + question marker
#
# It deliberately REQUIRES a Q/A/Ans/Answer/Question keyword before the
# number. This is what makes it "strong": a bare numbered line inside an
# answer, like "(1) Input:", "1. Array", or "① Linear", is never mistaken
# for a new question boundary. That mixing of sub-point numbering with
# question numbering was the single biggest source of bad segmentation
# in earlier attempts.

_MARKER = re.compile(
    r"(?im)"
    r"^[ \t]*"
    r"(?:Q(?:UESTION)?|A(?:NS(?:WER)?)?)"   # Q / QUESTION / A / ANS / ANSWER
    r"\s*(?:NO\.?)?\s*[-:.]?\s*"            # optional "No", optional separator
    r"Q?\s*\(?\s*(?P<num>\d+)\s*\)?"        # optional embedded "Q", the number
    r"[ \t]*(?:[.):\-])?"                   # optional trailing punctuation
    r"(?:[ \t]+|$)"                         # must be followed by space/EOL
)

# ============================================================
# Fallback marker pattern
# ============================================================
#
# Used ONLY when the whole document contains zero primary markers - i.e.
# there is no "Q"/"A"/"Ans" style labelling anywhere. In that situation a
# plain numbered list ("1) ...", "1. ...", "(1) ...", "① ...") is the best
# available signal, so we fall back to treating those as question
# boundaries instead of returning the entire document as a single blob.

_FALLBACK_MARKER = re.compile(
    r"(?m)"
    r"^[ \t]*"
    r"(?:\(?\s*(?P<num>\d+)\s*[.)\]]|(?P<circle>[①②③④⑤⑥⑦⑧⑨⑩]))"
    r"[ \t]*"
)

_CIRCLED_VALUES = {ch: index + 1 for index, ch in enumerate("①②③④⑤⑥⑦⑧⑨⑩")}


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse excessive blank lines produced by noisy OCR output so
    # marker detection stays reliable near page breaks / scan artifacts.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _segments_from_matches(text: str, matches: list[re.Match]) -> list[dict]:
    segments: list[dict] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        segment_text = text[start:end].strip()
        if not segment_text:
            continue
        number = match.group("num") if "num" in match.groupdict() and match.group("num") else None
        if number is None:
            number = _CIRCLED_VALUES.get(match.group("circle"), index + 1)
        segments.append({"question_id": f"Q{int(number)}", "text": segment_text})
    return segments


def segment_document(text: str) -> list[dict]:
    """
    Segment a complete OCR/markdown document into question-wise sections.

    The input should be the complete document text, not individual pages.

    Strategy (in order):
      1. Split on strong Q/A/Ans/Answer/Question markers. This is robust
         to OCR misreading "Q" as "A", to students labelling their own
         answers ("Ans(1)", "ANS(2)") instead of question numbers, and to
         inconsistent punctuation/spacing around the marker.
      2. If no strong markers exist anywhere in the document, fall back
         to plain numbered lines ("1)", "1.", "(1)", "①") as boundaries.
      3. If nothing at all is detected, return the whole document as a
         single "Q1" segment rather than dropping content.

    Returns:
        [
            {"question_id": "Q1", "text": "..."},
            ...
        ]
    """

    if not text or not text.strip():
        return []

    text = _normalize(text)

    matches = list(_MARKER.finditer(text))
    if matches:
        segments = _segments_from_matches(text, matches)
        if segments:
            return segments

    fallback_matches = list(_FALLBACK_MARKER.finditer(text))
    if fallback_matches:
        segments = _segments_from_matches(text, fallback_matches)
        if segments:
            return segments

    return [
        {
            "question_id": "Q1",
            "text": text.strip(),
        }
    ]