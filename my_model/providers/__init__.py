"""
Provider adapters for LLM backends.

This package defines a common interface (BaseProvider) that all provider
implementations must follow. It also includes a simple MockProvider that can
be used in tests without performing any network requests.
"""

from .base import BaseProvider, ProviderError
from .mock import MockProvider
from .openai import OpenAIProvider
from .regolo import RegoloProvider

__all__ = ["BaseProvider", "ProviderError", "MockProvider", "OpenAIProvider", "RegoloProvider"]
