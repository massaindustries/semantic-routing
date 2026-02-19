'''Regolo provider – OpenAI-compatible specialized adapter.

This provider implements the :class:`BaseProvider` contract for the Regolo API,
which follows the standard OpenAI chat completions schema.  The default
``base_url`` points to the public Regolo endpoint ``https://api.regolo.ai/v1``.

Typical usage::

    from my_model.providers.regolo import RegoloProvider
    provider = RegoloProvider(api_key='REG...')
    response = await provider.chat_completions(messages, model='gpt-4-mini', stream=False)

For streaming::

    async for chunk in await provider.chat_completions(messages, model='gpt-4-mini', stream=True):
        print(chunk.decode())
'''  

import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from .base import BaseProvider, ProviderError


class RegoloProvider(BaseProvider):
    '''Regolo-specific provider.
    
    Parameters
    ----------
    api_key: Optional[str]
        Bearer token for authentication.  If omitted, no ``Authorization`` header
        is sent – useful for public or mock endpoints.
    base_url: str, optional
        Base URL for the service.  Defaults to ``"https://api.regolo.ai/v1"``.
    timeout: int, default 30
        HTTP timeout (seconds) for both connect and read operations.
    max_retries: int, default 3
        Number of retry attempts for transient ``429`` or ``5xx`` errors.
    '''

    provider_id = 'regolo'

    def __init__(self, api_key: Optional[str] = None, *, base_url: str = 'https://api.regolo.ai/v1', timeout: int = 30, max_retries: int = 3):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        # Reuse a single AsyncClient for the lifetime of the instance.
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=timeout, write=timeout, pool=timeout))

    async def _request(self, payload: Dict[str, Any], stream: bool) -> Any:
        '''Internal helper to perform the HTTP request with retries.
        
        Returns either a JSON ``dict`` (non‑stream) or an ``AsyncIterator[bytes]``
        for streaming mode.
        '''
        # ``/v1`` is already part of ``base_url`` – we only need the ``chat/completions`` suffix.
        url = f"{self.base_url}/chat/completions"
        headers: Dict[str, str] = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f"Bearer {self.api_key}"

        for attempt in range(self.max_retries):
            try:
                if stream:
                    async with self._client.stream('POST', url, json=payload, headers=headers) as response:
                        response.raise_for_status()
                        return self._stream_iterator(response)
                else:
                    response = await self._client.post(url, json=payload, headers=headers)
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
        '''Yield SSE ``bytes`` lines from a streaming Regolo response.
        
        The upstream Regolo API already returns ``data: {...}`` lines.  For
        consistency with the :class:`MockProvider` we normalise each line to the
        ``data: <payload>\n\n`` format expected by FastAPI ``StreamingResponse``.
        '''
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

    async def chat_completions(self, messages: List[Dict[str, Any]], model: str, stream: bool = False, temperature: float = 0.0, extra: Optional[Dict[str, Any]] = None) -> Any:
        '''Send a chat‑completion request to Regolo.
        
        Parameters are forwarded to the Regolo API.  ``extra`` can be used to
        supply provider‑specific fields such as ``tools`` or ``tool_choice``.
        '''
        # Normalise messages to plain‑text (MVP).
        _ = self.normalize_messages(messages)
        payload: Dict[str, Any] = {
            'model': model,
            'messages': messages,
            'temperature': temperature,
            'stream': stream,
        }
        if extra:
            payload.update(extra)
        result = await self._request(payload, stream=stream)
        return result
