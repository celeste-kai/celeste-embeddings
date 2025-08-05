"""
Core data types for agent communication.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

from .enums import EmbeddingsProvider


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
    provider: Optional[EmbeddingsProvider] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
