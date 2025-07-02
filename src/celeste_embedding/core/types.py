"""
Core data types for agent communication.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict

from .enums import Provider


class AIUsage(BaseModel):
    """Token usage metrics for AI responses."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int
    output_tokens: int
    total_tokens: int


class Embedding(BaseModel):
    """Response from AI providers."""

    values: list[float]
    usage: Optional[AIUsage] = None
    provider: Optional[Provider] = None
    metadata: Dict[str, Any] = {}
