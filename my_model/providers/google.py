'''Google Gemini provider – adapter for the Gemini API.

This provider implements the :class:`BaseProvider` contract for the
Google Generative Language (Gemini) API. It translates the OpenAI‑style
chat request into Gemini's ``generateContent`` request and returns a
response compatible with the OpenAI format expected by the gateway.

Both synchronous (non‑stream) and streaming responses are supported.
'''

import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from .base import BaseProvider, ProviderError


class GoogleGeminiProvider(BaseProvider):
    """Adapter for Google Gemini chat completions.

    Parameters
    ----------
    api_key: Optional[str]
        API key for the Gemini service. If omitted, the request is sent
        without the ``key`` query parameter (useful for mock servers).
    base_url: str, optional
        Base URL for the Gemini API. Defaults to the public endpoint
        ``https://generativelanguage.googleapis.com/v1``.
    timeout: int, default 30
        HTTP timeout (seconds) for both connect and read operations.
    max_retries: int, default 3
        Number of retry attempts for transient ``429`` or ``5xx`` errors.
    """

    provider_id = "google"

    def __init__(self, api_key: Optional[str] = None, *, base_url: str = "https://generativelanguage.googleapis.com/v1", timeout: int = 30, max_retries: int = 3):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        # Reuse a single AsyncClient for the lifetime of the instance.
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=timeout, write=timeout, pool=timeout))

    def _convert_message(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Convert an OpenAI‑style message to Gemini format.

        Gemini expects a ``role`` field where ``assistant`` becomes ``model``.
        The content is expressed as a list of ``parts`` containing text.
        """
        role = msg.get("role", "user")
        if role == "assistant":
            role = "model"
        content = msg.get("content", "")
        return {"role": role, "parts": [{"text": str(content)}]}

    def _prepare_payload(self, messages: List[Dict[str, Any]], model: str, stream: bool, temperature: float, extra: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Build the JSON payload for a Gemini ``generateContent`` request."""
        gemini_messages = [self._convert_message(m) for m in messages]
        payload: Dict[str, Any] = {
            "contents": gemini_messages,
            "generationConfig": {"temperature": temperature},
            "stream": stream,
        }
        if extra:
            payload.update(extra)
        return payload

    async def _request(self, model: str, payload: Dict[str, Any], stream: bool) -> Any:
        """Internal helper to perform the HTTP request with retries.

        Returns a JSON ``dict`` for non‑streaming calls or an ``AsyncIterator[bytes]``
        yielding SSE ``data`` lines for streaming calls.
        """
        url = f"{self.base_url}/models/{model}:generateContent"
        params: Dict[str, str] = {}
        if self.api_key:
            params["key"] = self.api_key
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        for attempt in range(self.max_retries):
            try:
                if stream:
                    async with self._client.stream("POST", url, json=payload, params=params, headers=headers) as response:
                        response.raise_for_status()
                        return self._stream_iterator(response)
                else:
                    response = await self._client.post(url, json=payload, params=params, headers=headers)
                    response.raise_for_status()
                    return response.json()
            except httpx.HTTPStatusError as exc:
                try:
                    err_json = exc.response.json()
                    message = err_json.get("error", {}).get("message", exc.response.text)
                except Exception:
                    message = str(exc)
                if exc.response.status_code in {429, 500, 502, 503, 504} and attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise ProviderError(exc.response.status_code, message) from exc
            except Exception as exc:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise ProviderError(0, str(exc)) from exc
        raise ProviderError(0, "Maximum retry attempts exceeded")

    async def _stream_iterator(self, response: httpx.Response) -> AsyncIterator[bytes]:
        """Yield SSE ``bytes`` lines from a Gemini streaming response.

        Gemini streams raw JSON objects separated by newlines. We normalise each
        line to the ``data: <json>\n\n`` format expected by the FastAPI gateway.
        """
        async for raw in response.aiter_bytes():
            if not raw:
                continue
            text = raw.decode("utf-8", errors="replace")
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                # Remove any existing ``data: `` prefix.
                if line.startswith("data:"):
                    line = line[5:].lstrip()
                yield f"data: {line}\n\n".encode("utf-8")

    async def chat_completions(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        stream: bool = False,
        temperature: float = 0.0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Send a chat‑completion request to Gemini.

        The method normalises messages, builds the Gemini payload, and forwards
        the request using ``_request``.
        """
        # Normalise messages (kept for parity with other providers).
        _ = self.normalize_messages(messages)
        payload = self._prepare_payload(messages, model, stream, temperature, extra)
        result = await self._request(model, payload, stream=stream)
        return result
