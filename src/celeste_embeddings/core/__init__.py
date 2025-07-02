"""
Core data definitions for Celeste Embedding Client.
"""

from .enums import EmbeddingsProvider, LogLevel
from .types import Embedding

__all__ = [
    "EmbeddingsProvider",
    "LogLevel",
    "Embedding",
]
