"""
Celeste Embedding Client - Multi-provider embedding client for Python.
"""

from importlib import import_module
from typing import Any

from celeste_core import Provider
from celeste_core.base.embedder import BaseEmbedder
from celeste_core.config.settings import settings

from .core import Embedding
from .mapping import PROVIDER_MAPPING

__version__ = "0.1.0"


def create_embedder(provider: str | Provider, **kwargs: Any) -> BaseEmbedder:
    provider_enum = Provider(provider) if isinstance(provider, str) else provider
    if provider_enum not in PROVIDER_MAPPING:
        raise ValueError(
            f"Provider '{provider_enum.value}' is not wired for embeddings."
        )
    settings.validate_for_provider(provider_enum.value)
    module_path, class_name = PROVIDER_MAPPING[provider_enum]
    module = import_module(f"celeste_embeddings{module_path}")
    return getattr(module, class_name)(**kwargs)


__all__ = ["create_embedder", "BaseEmbedder", "Embedding"]
