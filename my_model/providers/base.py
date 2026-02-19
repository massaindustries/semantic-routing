'''Base provider interface for OpenAI chat completions.

This module defines a common abstract base class that concrete provider
implementations must inherit from.  It also defines a small error class and
a helper to normalise OpenAI‑style messages to plain‑text only (MVP).
''' 

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional, Union


class ProviderError(Exception):
    """Standardised error for provider failures.

    The OpenAI API returns errors as a JSON body with an ``error`` field.
    This exception captures the HTTP status code and the error message so
    that callers can convert it to the expected OpenAI shape.
    """

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message

    def to_dict(self) -> dict:
        """Return a dictionary in the OpenAI error format.

        ``{"error": {"message": <msg>}, "status_code": <code>}``
        """
        return {"error": {"message": self.message}, "status_code": self.status_code}


def _normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return messages where ``content`` is a plain string.

    For the MVP we only support text content. If a message contains a list of
    blocks (as used by the OpenAI schema) the function extracts the ``text``
    blocks and concatenates them.
    """
    normalized: List[Dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            texts: List[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    if text:
                        texts.append(text)
            normalized.append({
                "role": msg.get("role", "user"),
                "content": " ".join(texts) if texts else "",
            })
        else:
            normalized.append({
                "role": msg.get("role", "user"),
                "content": str(content),
            })
    return normalized


class BaseProvider(ABC):
    """Abstract base class for a chat‑completion provider.

    Concrete providers must implement :meth:`chat_completions`.  The method can
    return either a fully‑realised JSON‑compatible ``dict`` (when ``stream`` is
    ``False``) or an ``AsyncIterator[bytes]`` that yields SSE ``data`` lines
    (when ``stream`` is ``True``).
    """

    provider_id: str

    @abstractmethod
    async def chat_completions(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        stream: bool = False,
        temperature: float = 0.0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Union[Dict[str, Any], AsyncIterator[bytes]]:
        """Send a chat‑completion request.

        Parameters
        ----------
        messages:
            List of messages following the OpenAI chat schema.
        model:
            Identifier of the backend model to use.
        stream:
            When ``True`` the provider must return an async iterator that
            yields raw SSE ``bytes``.  When ``False`` a complete JSON payload is
            returned.
        temperature:
            Sampling temperature – forwarded to the backend when applicable.
        extra:
            Optional provider‑specific parameters.
        """
        raise NotImplementedError

    @staticmethod
    def normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Utility to normalise incoming messages to plain‑text only.
        """
        return _normalize_messages(messages)
