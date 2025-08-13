"""
Celeste Embedding Client - Multi-provider embedding client for Python.
"""

from typing import Any

from celeste_core import Provider
from celeste_core.base.embedder import BaseEmbedder
from celeste_core.config.settings import settings

from .core import Embedding

__version__ = "0.1.0"

SUPPORTED_PROVIDERS: set[Provider] = {
    Provider.GOOGLE,
    Provider.MISTRAL,
}


def create_embedder(provider: str | Provider, **kwargs: Any) -> BaseEmbedder:
    # normalize to enum
    provider_enum = Provider(provider) if isinstance(provider, str) else provider

    if provider_enum not in SUPPORTED_PROVIDERS:
        supported = [p.value for p in SUPPORTED_PROVIDERS]
        raise ValueError(
            f"Unsupported provider: {provider_enum.value}. Supported: {supported}"
        )

    # Validate environment for the chosen provider
    settings.validate_for_provider(provider_enum.value)

    mapping = {
        Provider.GOOGLE: (".providers.google", "GoogleEmbedder"),
        Provider.MISTRAL: (".providers.mistral", "MistralEmbedder"),
    }

    if provider_enum not in mapping:
        supported = [p.value for p in mapping.keys()]
        raise ValueError(
            f"Provider {provider_enum.value} not implemented. Supported: {supported}"
        )

    module_path, class_name = mapping[provider_enum]
    module = __import__(f"celeste_embeddings{module_path}", fromlist=[class_name])
    embedder_class = getattr(module, class_name)
    return embedder_class(**kwargs)


__all__ = [
    "create_embedder",
    "BaseEmbedder",
    "Embedding",
]
