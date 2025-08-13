"""
Core data types for agent communication.
"""

from typing import Any, Dict, Optional

from celeste_core import Provider
from pydantic import BaseModel, Field


class Embedding(BaseModel):
    """Response from AI providers."""

    values: list[float]

    provider: Optional[Provider] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
