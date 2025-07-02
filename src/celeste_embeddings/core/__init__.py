"""
Core data definitions for Celeste Embedding Client.
"""

from .enums import LogLevel, EmbeddingsProvider
from .types import Embedding

__all__ = [
    "EmbeddingsProvider",
    "LogLevel",
    "Embedding",
]
