'''OpenAI-compatible generic provider.

This provider implements the :class:`BaseProvider` contract for any service
that follows the OpenAI Chat Completions API (e.g. OpenAI, Azure OpenAI,
local OAI‑compatible servers).  It supports both synchronous (non‑streaming)
responses and streaming SSE responses.

Typical usage::

    from my_model.providers.openai import OpenAIProvider
    provider = OpenAIProvider(base_url="https://api.openai.com", api_key="sk-…")
    response = await provider.chat_completions(messages, model="gpt-4o", stream=False)

For streaming::

    async for chunk in await provider.chat_completions(messages, model="gpt-4o", stream=True):
        # ``chunk`` is a ``bytes`` object containing an SSE ``data: ...`` line.
        print(chunk.decode())
'''  

import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from .base import BaseProvider, ProviderError


class OpenAIProvider(BaseProvider):
    """Generic OpenAI‑compatible provider.

    Parameters
    ----------
    base_url: str
        Base URL of the service (e.g. ``"https://api.openai.com"``).  The
        provider will POST to ``"{base_url}/v1/chat/completions"``.
    api_key: Optional[str]
        Bearer token for ``Authorization`` header.  If omitted no auth header
        is sent – useful for local servers that do not require a key.
    timeout: int, default 30
        HTTP timeout (seconds) for both connect and read operations.
    max_retries: int, default 3
        Number of retry attempts for transient ``429`` or ``5xx`` errors.
    """

    provider_id = "openai"

    def __init__(self, base_url: str, api_key: Optional[str] = None, *, timeout: int = 30, max_retries: int = 3):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        # Create a single AsyncClient for the lifetime of the instance.
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=timeout, write=timeout, pool=timeout))

    async def _request(self, payload: Dict[str, Any], stream: bool) -> Any:
        """Internal helper to perform the HTTP request with retries.

        Returns either a JSON ``dict`` (non‑stream) or an ``AsyncIterator[bytes]``
        for streaming mode.
        """
        url = f"{self.base_url}/v1/chat/completions"
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Simple exponential back‑off for transient errors.
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
                # Extract a user‑friendly message if possible.
                try:
                    err_json = exc.response.json()
                    message = err_json.get("error", {}).get("message", exc.response.text)
                except Exception:
                    message = str(exc)
                # Retry on rate‑limit or server errors.
                if exc.response.status_code in {429, 500, 502, 503, 504} and attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise ProviderError(exc.response.status_code, message) from exc
            except Exception as exc:
                # Non‑HTTP errors (network, timeout, etc.)
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise ProviderError(0, str(exc)) from exc
        # Should never reach here because the loop either returns or raises.
        raise ProviderError(0, "Maximum retry attempts exceeded")

    async def _stream_iterator(self, response: httpx.Response) -> AsyncIterator[bytes]:
        """Yield SSE ``bytes`` lines from a streaming OpenAI response.

        The upstream API already returns ``data: {...}`` lines.  For consistency
        with the ``MockProvider`` we strip any existing ``data: `` prefix and
        re‑emit each line as ``data: <payload>\n\n``.
        """
        async for raw in response.aiter_bytes():
            if not raw:
                continue
            text = raw.decode("utf-8", errors="replace")
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                # Normalise the line to the expected SSE format.
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
        """Send a chat‑completion request.

        Returns a JSON ``dict`` when ``stream=False`` or an ``AsyncIterator``
        yielding SSE ``bytes`` when ``stream=True``.
        """
        # Normalise message content to plain‑text (MVP).
        _ = self.normalize_messages(messages)

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }
        if extra:
            payload.update(extra)

        result = await self._request(payload, stream=stream)
        return result
