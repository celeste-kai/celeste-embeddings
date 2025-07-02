from typing import Any, List, Union

from google import genai

from ..base import BaseEmbedder
from ..core.config import GOOGLE_API_KEY
from ..core.enums import GoogleEmbedding, Provider
from ..core.types import Embedding


class GoogleEmbedder(BaseEmbedder):
    def __init__(
        self, model: str = GoogleEmbedding.GEMINIEMBEDDING.value, **kwargs: Any
    ) -> None:
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.embedding_model = model

    async def generate_embeddings(
        self, texts: Union[str, List[str]], **kwargs: Any
    ) -> List[Embedding]:
        """Generate embeddings for the given text(s)."""
        if isinstance(texts, str):
            texts = [texts]

        response = await self.client.aio.models.embed_content(
            model=self.embedding_model, contents=texts
        )

        return [
            Embedding(
                values=embedding.values,
                usage=None,
                provider=Provider.GOOGLE,
                metadata={"model": self.embedding_model},
            )
            for embedding in response.embeddings
        ]
