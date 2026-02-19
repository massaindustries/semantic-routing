# '''Anthropic provider – OpenAI‑compatible wrapper for Anthropic Messages API.

# This provider implements the BaseProvider contract for the Anthropic API.
# It forwards a request to the Anthropic Messages endpoint and returns a response
# in the OpenAI‑compatible shape. For streaming, the raw SSE from Anthropic is
# normalised to the `data: ...` format expected by the gateway.
# '''

import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from .base import BaseProvider, ProviderError


class AnthropicProvider(BaseProvider):
    """Adapter for the Anthropic Messages API.

    Parameters
    ----------
    api_key: Optional[str]
        The Anthropic `x-api-key`. If omitted, no authentication header is sent.
    base_url: str, optional
        Base URL for the service. Defaults to "https://api.anthropic.com/v1".
    timeout: int, default 30
        HTTP timeout (seconds) for both connect and read operations.
    max_retries: int, default 3
        Number of retry attempts for transient `429` or `5xx` errors.
    """

    provider_id = "anthropic"

    def __init__(self, api_key: Optional[str] = None, *, base_url: str = "https://api.anthropic.com/v1", timeout: int = 30, max_retries: int = 3):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=timeout, write=timeout, pool=timeout))

    async def _request(self, payload: Dict[str, Any], stream: bool) -> Any:
        """Internal helper to perform the HTTP request with retries.

        Returns either a JSON dict (non‑stream) or an AsyncIterator[bytes]
        for streaming mode.
        """
        url = f"{self.base_url}/messages"
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key

        for attempt in range(self.max_retries):
            try:
                if stream:
                    async with self._client.stream("POST", url, json=payload, headers=headers) as response:
                        response.raise_for_status()
                        return self._stream_iterator(response)
                else:
                    response = await self._client.post(url, json=payload, headers=headers)
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
        """Yield SSE bytes lines from a streaming Anthropic response.

        Anthropic already returns `data: {...}` lines. We normalise each line to
        `data: <payload>\n\n` – the same format used by other providers.
        """
        async for raw in response.aiter_bytes():
            if not raw:
                continue
            text = raw.decode("utf-8", errors="replace")
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("data: "):
                    line = line[6:]
                yield f"data: {line}\n\n".encode("utf-8")

    async def chat_completions(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        stream: bool = False,
        temperature: float = 0.0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Send a chat‑completion request to Anthropic.

        The request payload follows the Anthropic Messages API. `extra` may be used
        to provide additional fields such as `max_tokens`. If `max_tokens` is not
        supplied we default to 1024.
        """
        _ = self.normalize_messages(messages)
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }
        if extra:
            payload.update(extra)
        if "max_tokens" not in payload:
            payload["max_tokens"] = 1024
        result = await self._request(payload, stream=stream)
        return result
