"""
Celeste Embedding Client - Multi-provider embedding client for Python.
"""

from typing import Any

from .base import BaseEmbedder
from .core import Embedding, EmbeddingsProvider

__version__ = "0.1.0"

SUPPORTED_PROVIDERS = [
    "google",
    "openai",
    "mistral",
    "huggingface",
    "ollama",
]


def create_embedder(provider: str, **kwargs: Any) -> BaseEmbedder:
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider}")

    if provider == "google":
        from .providers.google import GoogleEmbedder

        return GoogleEmbedder(**kwargs)
    elif provider == "mistral":
        from .providers.mistral import MistralEmbedder

        return MistralEmbedder(**kwargs)
    # Other providers to be implemented

    raise ValueError(f"Provider {provider} not implemented")


__all__ = [
    "create_embedder",
    "BaseEmbedder",
    "EmbeddingsProvider",
    "Embedding",
]
