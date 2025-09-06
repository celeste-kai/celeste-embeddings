from typing import Any

from celeste_core import AIResponse, Provider
from celeste_core.base.embedder import BaseEmbedder
from celeste_core.config.settings import settings
from mistralai import Mistral


class MistralEmbedder(BaseEmbedder):
    def __init__(self, model: str = "mistral-embed", **kwargs: Any) -> None:
        super().__init__(model=model, provider=Provider.MISTRAL, **kwargs)
        self.client = Mistral(api_key=settings.mistral.api_key)

    async def generate_embeddings(self, texts: str | list[str], **kwargs: Any) -> AIResponse:  # noqa: ARG002
        """Generate embeddings for the given text(s).

        Returns AIResponse[List[List[float]]].
        """
        if isinstance(texts, str):
            texts = [texts]

        response = await self.client.embeddings.create_async(model=self.model, inputs=texts)

        vectors: list[list[float]] = [data.embedding for data in response.data]

        return AIResponse(
            content=vectors,
            provider=Provider.MISTRAL,
            metadata={"model": self.model},
        )
