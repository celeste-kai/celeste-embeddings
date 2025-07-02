from typing import Any, List, Union

from mistralai import Mistral

from ..base import BaseEmbedder
from ..core.config import MISTRAL_API_KEY
from ..core.enums import EmbeddingsProvider, MistralEmbedding
from ..core.types import Embedding


class MistralEmbedder(BaseEmbedder):
    def __init__(
        self, model: str = MistralEmbedding.MISTRAL_EMBED.value, **kwargs: Any
    ) -> None:
        self.client = Mistral(api_key=MISTRAL_API_KEY)
        self.model = model

    async def generate_embeddings(
        self, texts: Union[str, List[str]], **kwargs: Any
    ) -> List[Embedding]:
        """Generate embeddings for the given text(s)."""
        if isinstance(texts, str):
            texts = [texts]

        response = await self.client.embeddings.create_async(
            model=self.model, inputs=texts
        )

        return [
            Embedding(
                values=data.embedding,
                usage=None,
                provider=EmbeddingsProvider.MISTRAL,
                metadata={"model": self.model},
            )
            for data in response.data
        ]
