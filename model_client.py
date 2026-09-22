"""
Lightweight client for the persistent vLLM model server.

FastAPI (main.py) uses OCREngineClient here instead of instantiating
the model directly. That means `uvicorn main:app --reload` restarts the API 
process without ever touching CUDA VRAM - the actual model stays loaded in 
the separate vLLM server process.

OCREngineClient mirrors the subset of the original OlmOCREngine interface that
the rest of the project actually calls (transcribe, generate_text),
so downstream files need no changes beyond receiving this
object instead of an OlmOCREngine instance - it's a drop-in swap.

Plain `requests` (already a project dependency) is used for the HTTP
call to vLLM's standard OpenAI-compatible /v1/chat/completions endpoint.
"""

from __future__ import annotations

import base64
import io
import os
import re
from dataclasses import dataclass
from typing import Optional

import requests
import httpx
from PIL import Image

MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://127.0.0.1:8001")
DEFAULT_TIMEOUT = float(os.getenv("MODEL_SERVER_TIMEOUT", "300"))
# Defaults to Qwen2.5-VL-7B-Instruct or allenai/olmOCR-2-7B-1025 depending on vllm start
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct") 

_FRONT_MATTER_RE = re.compile(r"^\s*---.*?---\s*\n?", re.DOTALL)


class ModelServerError(RuntimeError):
    """Raised when the model server is unreachable, still loading, or
    returns an error for a transcribe/generate_text call."""


@dataclass
class PageResult:
    page: int
    text: str
    confidence: Optional[float] = None
    raw_metadata: Optional[str] = None


class OCREngineClient:
    """Drop-in stand-in for OlmOCREngine that forwards transcribe()/
    generate_text() calls to the vLLM OpenAI API server over HTTP."""

    def __init__(
        self,
        base_url: str = MODEL_SERVER_URL,
        timeout: float = DEFAULT_TIMEOUT,
        model_id: str = MODEL_ID,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.model_id = model_id
        
        # Determine if we should attempt to strip YAML front-matter 
        # (required for stock olmOCR prompt).
        from olmocr_grading.config import load_config
        self.ocr_config = load_config(None)

    # ---------- lifecycle ----------

    def load(self) -> None:
        """Called from main.py's startup event to fail fast with a
        clear error if the vLLM model server isn't up yet."""

        if not self.is_ready():
            raise ModelServerError(
                f"Model server at {self.base_url} is not ready. "
                f"Start it first with `vllm serve {self.model_id}` (or your actual model id)."
            )
            
        # Discover the actual model ID from vLLM if possible so we don't send mismatched requests
        try:
            resp = requests.get(f"{self.base_url}/v1/models", timeout=5)
            if resp.status_code == 200:
                models = resp.json().get("data", [])
                if models:
                    self.model_id = models[0]["id"]
        except Exception:
            pass

    # ---------- inference ----------

    async def transcribe(self, image: Image.Image, page_number: int = 1) -> PageResult:
        from olmocr_grading.prompts import get_prompt
        prompt_text = get_prompt(self.ocr_config.use_stock_prompt)

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")

        payload = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                    ]
                }
            ],
            "max_tokens": self.ocr_config.max_new_tokens,
            "temperature": self.ocr_config.temperature if self.ocr_config.do_sample else 0.0,
            "presence_penalty": self.ocr_config.repetition_penalty - 1.0 if self.ocr_config.repetition_penalty > 1.0 else 0.0,
        }

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
            except httpx.RequestError as e:
                raise ModelServerError(
                    f"vLLM /v1/chat/completions call failed for page {page_number}: {e}"
                ) from e
            except httpx.HTTPStatusError as e:
                raise ModelServerError(
                    f"vLLM /v1/chat/completions call failed with status: {e.response.status_code}"
                ) from e

        data = resp.json()
        raw_text = data["choices"][0]["message"]["content"].strip()
        
        metadata = None
        text = raw_text
        if self.ocr_config.use_stock_prompt:
            match = _FRONT_MATTER_RE.match(raw_text)
            if match:
                metadata = match.group(0).strip()
                text = raw_text[match.end():].strip()

        return PageResult(
            page=page_number,
            text=text,
            confidence=None, # Mean confidence requires logprobs natively, unsupported here
            raw_metadata=metadata,
        )

    async def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 3072,
        do_sample: bool = False,
        repetition_penalty: float = 1.05,
        is_json: bool = False,
    ) -> str:
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_new_tokens,
            "temperature": 0.7 if do_sample else 0.0,
            "presence_penalty": repetition_penalty - 1.0 if repetition_penalty > 1.0 else 0.0,
        }
        
        if is_json:
            payload["response_format"] = {"type": "json_object"}

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
            except httpx.RequestError as e:
                raise ModelServerError(
                    f"vLLM /v1/chat/completions call failed: {e}"
                ) from e
            except httpx.HTTPStatusError as e:
                raise ModelServerError(
                    f"vLLM /v1/chat/completions call failed with status: {e.response.status_code}"
                ) from e

        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    # ---------- status ----------

    def health(self) -> dict:
        """Never raises - returns {"status": "unreachable", ...} if
        the model server can't be reached at all, so callers can degrade gracefully."""
        try:
            resp = requests.get(f"{self.base_url}/v1/models", timeout=5)
            resp.raise_for_status()
            models = resp.json().get("data", [])
            loaded_models = [m["id"] for m in models]
            return {"status": "ok", "models": loaded_models}
        except requests.RequestException as e:
            return {"status": "unreachable", "error": str(e)}

    def is_ready(self) -> bool:
        return self.health().get("status") == "ok"
