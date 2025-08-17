from celeste_core.enums.capability import Capability
from celeste_core.enums.providers import Provider

CAPABILITY: Capability = Capability.EMBEDDINGS

PROVIDER_MAPPING: dict[Provider, tuple[str, str]] = {
    Provider.GOOGLE: (".providers.google", "GoogleEmbedder"),
    Provider.MISTRAL: (".providers.mistral", "MistralEmbedder"),
}

__all__ = ["CAPABILITY", "PROVIDER_MAPPING"]
