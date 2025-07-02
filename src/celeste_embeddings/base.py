from abc import ABC, abstractmethod
from typing import Any, List, Union

from .core.types import Embedding


class BaseEmbedder(ABC):
    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        """
        Initializes the client, loading credentials from the environment.
        Provider-specific arguments can be passed via kwargs.
        """
        pass

    @abstractmethod
    async def generate_embeddings(
        self, texts: Union[str, List[str]], **kwargs: Any
    ) -> List[Embedding]:
        """
        Generates embeddings for the given text(s).
        """
        pass
