from typing import Any

from celeste_core import AIResponse, Provider
from celeste_core.base.embedder import BaseEmbedder
from celeste_core.config.settings import settings
from google import genai


class GoogleEmbedder(BaseEmbedder):
    def __init__(self, model: str = "gemini-embedding-exp-03-07", **kwargs: Any) -> None:
        super().__init__(model=model, provider=Provider.GOOGLE, **kwargs)
        self.client = genai.Client(api_key=settings.google.api_key)

    async def generate_embeddings(self, texts: str | list[str], **kwargs: Any) -> AIResponse:  # noqa: ARG002
        """Generate embeddings for the given text(s).

        Returns AIResponse[List[List[float]]].
        """
        if isinstance(texts, str):
            texts = [texts]

        response = await self.client.aio.models.embed_content(model=self.model, contents=texts)

        vectors: list[list[float]] = [embedding.values for embedding in response.embeddings]

        return AIResponse(
            content=vectors,
            provider=Provider.GOOGLE,
            metadata={"model": self.model},
        )
