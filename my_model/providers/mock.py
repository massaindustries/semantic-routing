'''Mock provider for testing without external network calls.

The mock provider implements the same contract as :class:`BaseProvider` but
returns deterministic data.  It can be used in unit tests or examples where a
real LLM backend is not required.
''' 

import json
import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from .base import BaseProvider, ProviderError


class MockProvider(BaseProvider):
    """A simple in‑memory provider that returns static responses.

    The implementation is deliberately minimal – it normalises the incoming
    messages (to demonstrate that the helper works) but otherwise ignores the
    content and always returns the same mock response.
    """

    provider_id = "mock"

    async def chat_completions(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        stream: bool = False,
        temperature: float = 0.0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Union[Dict[str, Any], AsyncIterator[bytes]]:
        # Normalise messages to plain‑text only – the mock does not use them.
        _ = self.normalize_messages(messages)

        if stream:
            return self._stream_response(model)
        else:
            return self._full_response(model)

    def _full_response(self, model: str) -> Dict[str, Any]:
        """Return a static JSON payload that mimics an OpenAI chat response."""
        return {
            "id": "mock-response",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "This is a mock response."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    async def _stream_response(self, model: str) -> AsyncIterator[bytes]:
        """Yield a minimal SSE stream consisting of a single chunk.

        The format follows the OpenAI streaming specification where each line
        is prefixed by ``data: `` and terminated with a double newline.
        """
        # Simulate a tiny delay to make the async iterator realistic.
        await asyncio.sleep(0.01)

        chunk = {
            "id": "mock-stream",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "This is a mock streaming response."},
                    "finish_reason": None,
                }
            ],
        }
        line = f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
        yield line
