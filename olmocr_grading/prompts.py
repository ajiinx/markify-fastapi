"""
Prompt templates.

olmOCR-2-7B-1025 is a Qwen2.5-VL-7B-Instruct fine-tune. The official
"olmocr" toolkit ships a fixed prompt (build_no_anchoring_v4_yaml_prompt)
that wraps output in a YAML front-matter block (primary_language,
is_rotation_valid, is_table, is_diagram, ...) followed by the transcribed
text. That prompt is tuned for general document digitization and does not
know anything about grading-specific constraints.

For handwritten answer-sheet grading we need stricter behaviour:
    - never "correct" or complete the student's answer
    - mark illegible spans with a literal [UNCLEAR] token instead of
      guessing or skipping them
    - keep question numbers and answer boundaries intact for downstream
      parsing

So by default this pipeline uses a custom instruction prompt (below)
instead of the stock one. Because the model is still a standard
instruction-tuned VLM under the hood, it follows a well-specified custom
prompt reliably. Set use_stock_prompt=True in the config (requires
`pip install olmocr`) if you want the exact benchmark-reported prompt
instead -- see get_prompt().
"""

from __future__ import annotations

GRADING_OCR_PROMPT = """Below is an image of a single page from a handwritten student exam / answer sheet.

Transcribe ALL text on the page exactly as written, preserving:
- The original reading order (top to bottom; left column before right column if the page is split into columns).
- Question numbers and sub-parts exactly as written (e.g. "1.", "1)", "Q2", "(a)", "iii)"), each starting its own line.
- Paragraph breaks and line breaks as they appear in the handwriting.
- Bullet points and numbered lists, keeping the original marker style.
- Indentation of sub-answers relative to their question number (use leading spaces).
- Mathematical expressions and equations, written out in plain text or simple notation (e.g. "x^2 + 3x - 4 = 0", "H2O", "integral of x dx").
- Tables: if a real table structure can't be reproduced in plain text, describe it as "[TABLE: <row/column contents summarized>]".
- Diagrams, sketches, or graphs: do not try to draw them; instead insert a short bracketed description, e.g. "[DIAGRAM: labelled circuit with resistor and battery]".

Strict rules:
1. Do NOT correct spelling, grammar, or factual errors in the student's answer. Transcribe exactly what is written, mistakes and all.
2. Do NOT invent, guess, auto-complete, or "fix" any word, number, symbol, or answer you cannot clearly read.
3. If a word, phrase, digit, or region is illegible or ambiguous, output the literal token [UNCLEAR] in its place. Never silently drop it.
4. Never merge two different questions' answers into one block of text. When a new question number begins, start a new line.
5. Preserve the exact order the answers appear in on the page.
6. Output plain transcribed text only. Do not add your own commentary, summaries, headers, or explanations outside the transcription itself.

Transcribe the page now:"""


def get_prompt(use_stock_prompt: bool = False) -> str:
    """
    Return the text prompt to send alongside the page image.

    use_stock_prompt=True attempts to use the official olmOCR toolkit
    prompt (requires `pip install olmocr`). Falls back to the grading
    prompt with a warning if the package isn't installed.
    """
    if use_stock_prompt:
        try:
            from olmocr.prompts import build_no_anchoring_v4_yaml_prompt

            return build_no_anchoring_v4_yaml_prompt()
        except ImportError:
            import logging

            logging.getLogger(__name__).warning(
                "use_stock_prompt=True but the 'olmocr' package is not installed "
                "(pip install olmocr>=0.4.0). Falling back to the built-in grading prompt."
            )
    return GRADING_OCR_PROMPT
