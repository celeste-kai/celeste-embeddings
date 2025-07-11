"""
Core enumerations for Celeste AI Client.
"""

from enum import Enum


class EmbeddingsProvider(Enum):
    """AI provider enumeration for multi-provider agent support."""

    GOOGLE = "google"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    MISTRAL = "mistral"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"


class LogLevel(Enum):
    """Logging level enumeration for agent operations."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class GoogleEmbedding(Enum):
    """Gemini 2.5 model enumeration for provider-specific model selection."""

    GEMINIEMBEDDING = "gemini-embedding-exp-03-07"
    TEXTEMBEDDING004 = "text-embedding-004"
    EMBEDDING001 = "embedding-001"


class MistralEmbedding(Enum):
    """Mistral AI model enumeration for provider-specific model selection."""

    MISTRAL_EMBED = "mistral-embed"
    CODESTRAL_EMBED = "codestral-embed"
