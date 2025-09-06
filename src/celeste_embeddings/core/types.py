"""
Core data types for agent communication.
"""

from typing import Any

from celeste_core import Provider
from pydantic import BaseModel, Field


class Embedding(BaseModel):
    """Response from AI providers."""

    values: list[float]

    provider: Provider | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
