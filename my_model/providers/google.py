'''Google Gemini provider – OpenAI‑compatible adapter.

This provider implements :class:`BaseProvider` for Google's Gemini API
(Generative Language). It forwards a chat‑completion request to the Gemini
`generateContent` endpoint. The provider supports both normal (JSON) and
streaming (SSE) responses.
'''

import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from .base import BaseProvider, ProviderError


class GoogleGeminiProvider(BaseProvider):
    """Adapter for Google Gemini (Generative Language) API.

    Parameters
    ----------
    api_key: Optional[str]
        Google API key used as a query‑parameter `key`. If omitted the request
        is sent without authentication (useful for mock endpoints).
    base_url: str, optional
        Base URL for the service. Defaults to the public Gemini endpoint
        `https://generativelanguage.googleapis.com/v1`.
    timeout: int, default 30
        HTTP timeout (seconds) for both connect and read operations.
    max_retries: int, default 3
        Number of retry attempts for transient `429` or `5xx` errors.
    """
    provider_id = 'google'

    def __init__(self, api_key: Optional[str] = None, *, base_url: str = 'https://generativelanguage.googleapis.com/v1', timeout: int = 30, max_retries: int = 3):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=timeout, write=timeout, pool=timeout))

    def _map_role(self, role: str) -> str:
        """Map OpenAI role to Gemini role.

        Gemini uses `user` for user messages and `model` for assistant
        messages. Any other role is passed through unchanged.
        """
        if role.lower() == 'assistant':
            return 'model'
        return role.lower()

    def _messages_to_contents(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI chat messages to Gemini ``contents`` format.

        Each OpenAI message is turned into a dict with ``role`` and a single
        ``parts`` element containing the plain text content.
        """
        contents: List[Dict[str, Any]] = []
        for msg in messages:
            role = self._map_role(msg.get('role', 'user'))
            content = msg.get('content', '')
            contents.append({
                'role': role,
                'parts': [{'text': str(content)}],
            })
        return contents

    async def _request(self, model: str, payload: Dict[str, Any], stream: bool) -> Any:
        """Internal helper to perform the HTTP request with retries.

        Returns either a JSON ``dict`` (non‑stream) or an ``AsyncIterator[bytes]``
        for streaming mode.
        """
        url = f'{self.base_url}/models/{model}:generateContent'
        params: Dict[str, str] = {}
        if self.api_key:
            params['key'] = self.api_key
        if stream:
            params['alt'] = 'sse'
        headers: Dict[str, str] = {'Content-Type': 'application/json'}

        for attempt in range(self.max_retries):
            try:
                if stream:
                    async with self._client.stream('POST', url, json=payload, headers=headers, params=params) as response:
                        response.raise_for_status()
                        return self._stream_iterator(response)
                else:
                    response = await self._client.post(url, json=payload, headers=headers, params=params)
                    response.raise_for_status()
                    return response.json()
            except httpx.HTTPStatusError as exc:
                try:
                    err_json = exc.response.json()
                    message = err_json.get('error', {}).get('message', exc.response.text)
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
        raise ProviderError(0, 'Maximum retry attempts exceeded')

    async def _stream_iterator(self, response: httpx.Response) -> AsyncIterator[bytes]:
        """Yield SSE ``bytes`` lines from a streaming Gemini response.

        Gemini already returns ``data: {...}`` lines when ``alt=sse`` is used.
        We normalise each line to the ``data: <payload>\n\n`` format expected by
        the FastAPI ``StreamingResponse``.
        """
        async for raw in response.aiter_bytes():
            if not raw:
                continue
            text = raw.decode('utf-8', errors='replace')
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith('data: '):
                    line = line[6:]
                yield f'data: {line}\n\n'.encode('utf-8')

    async def chat_completions(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        stream: bool = False,
        temperature: float = 0.0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Send a chat‑completion request to Gemini.

        Parameters
        ----------
        messages:
            List of OpenAI‑style message dicts.
        model:
            Gemini model name, e.g. ``gemini-pro`` or ``gemini-pro-vision``.
        stream:
            When ``True`` the request uses the SSE streaming endpoint.
        temperature:
            Sampling temperature, passed through to Gemini's ``generationConfig``.
        extra:
            Optional provider‑specific fields that are merged into the request
            payload.
        """
        _ = self.normalize_messages(messages)

        contents = self._messages_to_contents(messages)

        payload: Dict[str, Any] = {
            'contents': contents,
            'generationConfig': {
                'temperature': temperature,
            },
        }
        if extra:
            payload.update(extra)

        result = await self._request(model=model, payload=payload, stream=stream)
        return result
