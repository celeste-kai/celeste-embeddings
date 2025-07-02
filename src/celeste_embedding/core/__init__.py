"""
Core data definitions for Celeste Embedding Client.
"""

from .enums import LogLevel, Provider
from .types import Embedding

__all__ = [
    "Provider",
    "LogLevel",
    "Embedding",
]
