import math

from ollama import Client

DEFAULT_MODEL = "nomic-embed-text:v1.5"
DEFAULT_HOST = "http://localhost:11434"


class OllamaEmbeddingProvider:

    # Connect to Ollama using the supported Nomic embedding model.
    def __init__(
        self,
        host: str = DEFAULT_HOST,
        timeout: float = 120.0,
    ) -> None:
        self._model = DEFAULT_MODEL
        self._client = Client(host=host, timeout=timeout)

    # Include the provider, model and prefix strategy in the identity.
    @property
    def model_id(self) -> str:
        return f"ollama/{self._model}/search-prefix-v1"

    # Generate document embeddings using Nomic's document prefix.
    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return self._embed(
            texts,
            prefix="search_document: ",
        )

    # Generate a query embedding using Nomic's query prefix.
    def embed_query(self, text: str) -> list[float]:
        return self._embed(
            [text],
            prefix="search_query: ",
        )[0]

    # Call Ollama and validate the returned vectors.
    def _embed(
        self,
        texts: list[str],
        prefix: str,
    ) -> list[list[float]]:
        if not texts:
            return []

        if any(not text.strip() for text in texts):
            raise ValueError("Embedding inputs must not be blank.")

        response = self._client.embed(
            model=self._model,
            input=[f"{prefix}{text}" for text in texts],
            truncate=False,
        )

        vectors = [
            [float(value) for value in vector]
            for vector in response.embeddings
        ]

        if len(vectors) != len(texts):
            raise RuntimeError(
                "Embedding count does not match input count."
            )

        dimensions = {len(vector) for vector in vectors}

        if len(dimensions) != 1 or 0 in dimensions:
            raise RuntimeError(
                "Embedding vectors have invalid dimensions."
            )

        if any(
            not math.isfinite(value)
            for vector in vectors
            for value in vector
        ):
            raise RuntimeError(
                "Embedding vectors contain non-finite values."
            )

        return vectors