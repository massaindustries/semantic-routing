'''Utility functions for multimodal (audio/image) processing.

This module provides helpers that were originally part of `core.py` but are
re‑implemented here to keep the new modular architecture clean.  The functions
are deliberately lightweight and only depend on standard library modules and
the `httpx` package (already a dependency of the providers).

The primary responsibilities are:
- Detect which modality (text, image, audio) is present in a list of
  OpenAI‑style messages.
- Extract text, image URLs, and audio URLs from the message content.
- Perform audio transcription via the Regolo API (faster‑whisper).
- Perform OCR on images via the Regolo API (deepseek‑ocr).

These helpers are used by the FastAPI gateway to pre‑process multimodal
requests before routing through the VSR client.
'''  # noqa: D400

from __future__ import annotations

import os
import base64
import io
import json
import logging
from typing import List, Dict, Any

import httpx

logger = logging.getLogger(__name__)


def detect_modality(messages: List[Dict[str, Any]]) -> Dict[str, bool]:
    """Return a dict indicating presence of ``text``, ``image`` and ``audio``.

    The function mirrors the behaviour of the original implementation in
    ``core.py``.  ``messages`` is a list of OpenAI‑style message dicts where the
    ``content`` field may be a string or a list of content blocks.
    """
    has_image = False
    has_text = False
    has_audio = False

    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block_type = block.get("type", "")
                    if block_type == "text":
                        has_text = True
                    elif block_type == "image_url":
                        has_image = True
                    elif block_type == "audio" or block_type == "input_audio":
                        # Both ``audio`` and ``input_audio`` are used in tests.
                        has_audio = True
        elif isinstance(content, str) and content.strip():
            has_text = True
    return {"image": has_image, "text": has_text, "audio": has_audio}


def extract_text(content: Any) -> str:
    """Extract plain‑text content from a message ``content`` field.

    ``content`` can be a string or a list of blocks.  Only blocks of type
    ``text`` are concatenated.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                txt = block.get("text", "")
                if txt:
                    texts.append(txt)
        return " ".join(texts)
    return ""


def extract_image_url(content: Any) -> str:
    """Return the first image URL found in ``content`` or an empty string.
    Supports both the ``image_url`` block format used by the tests and a plain
    string URL (unlikely but handled for robustness).
    """
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image_url":
                img_obj = block.get("image_url", {})
                if isinstance(img_obj, dict):
                    return img_obj.get("url", "")
                if isinstance(img_obj, str):
                    return img_obj
    return ""


def extract_audio_url(content: Any) -> str:
    """Return the first audio URL found in ``content``.

    The tests use two block types: ``audio`` (with an ``audio`` field) and
    ``input_audio`` (with an ``audio_url`` dict).  This helper normalises both.
    """
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type", "")
            if block_type == "audio":
                # Direct ``audio`` field may contain the data URI.
                return block.get("audio", "") or block.get("url", "")
            if block_type == "input_audio":
                audio_url = block.get("audio_url", {})
                if isinstance(audio_url, dict):
                    return audio_url.get("url", "")
                if isinstance(audio_url, str):
                    return audio_url
    return ""


async def transcribe_audio(audio_content: str) -> Dict[str, Any]:
    """Transcribe base64‑encoded audio via Regolo's ``audio/transcriptions``.

    ``audio_content`` is expected to be either a plain base64 string or a data
    URI of the form ``data:audio/...;base64,<payload>``.
    """
    # Strip optional ``data:`` prefix.
    if audio_content.startswith("data:"):
        parts = audio_content.split(",", 1)
        if len(parts) == 2:
            audio_content = parts[1]
    try:
        audio_bytes = base64.b64decode(audio_content)
    except Exception as exc:
        logger.error(f"Failed to decode audio base64: {exc}")
        return {"error": {"message": f"Invalid base64: {exc}"}}

    headers = {"Authorization": f"Bearer {os.getenv('REGOLO_API_KEY', '')}"}
    audio_file = io.BytesIO(audio_bytes)
    files = {"file": ("audio.webm", audio_file, "application/octet-stream")}
    data = {"model": "faster-whisper-large-v3"}
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.post(
                "https://api.regolo.ai/v1/audio/transcriptions",
                files=files,
                data=data,
                headers=headers,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.error(f"Audio transcription request failed: {exc}")
            return {"error": {"message": str(exc)}}


async def process_image(image_content: str) -> Dict[str, Any]:
    """Run OCR on a base64‑encoded image using Regolo's ``deepseek-ocr`` model.
    ``image_content`` should be a data URI (``data:image/...;base64,<payload>``).
    """
    headers = {
        "Authorization": f"Bearer {os.getenv('REGOLO_API_KEY', '')}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "deepseek-ocr",
        "messages": [{
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": image_content}}]
        }]
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.post("https://api.regolo.ai/v1/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.error(f"Image OCR request failed: {exc}")
            return {"error": {"message": str(exc)}}
