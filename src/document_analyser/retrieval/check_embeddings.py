from document_analyser.retrieval.embeddings import EmbeddingProvider
from document_analyser.retrieval.providers.ollama_embeddings import (
    OllamaEmbeddingProvider,
)


# Verify that a provider returns compatible document and query vectors.
def check_provider(provider: EmbeddingProvider) -> None:
    document_vector = provider.embed_documents([
        "The taxpayer must retain supporting receipts for five years."
    ])[0]

    query_vector = provider.embed_query(
        "How long should receipts be kept?"
    )

    if len(document_vector) != len(query_vector):
        raise RuntimeError(
            "Document and query dimensions differ."
        )

    print(f"Model: {provider.model_id}")
    print(f"Vector dimensions: {len(document_vector)}")
    print(f"First five values: {document_vector[:5]}")


# Select the concrete provider at the application entry point.
def main() -> None:
    provider = OllamaEmbeddingProvider()
    check_provider(provider)


if __name__ == "__main__":
    main()