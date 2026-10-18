from typing import Protocol


class EmbeddingProvider(Protocol):

    @property
    # Identify the embedding model and configuration used by this provider.
    def model_id(self) -> str:
        ...

    # Generate one vector per document, preserving input order.
    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        ...

    # Generate a search-query vector compatible with document vectors.
    def embed_query(self, text: str) -> list[float]:
        ...