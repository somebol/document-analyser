import argparse
from dataclasses import dataclass
from hashlib import sha256

from qdrant_client import QdrantClient, models

from document_analyser.config import get_settings
from document_analyser.retrieval.embeddings import EmbeddingProvider
from document_analyser.retrieval.providers.ollama_embeddings import (
    OllamaEmbeddingProvider,
)


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    document_version_id: str
    source_file: str
    item_ref: str
    page_numbers: tuple[int, ...]
    start: int
    end: int
    text: str
    score: float


# Retrieve source-linked passages using the configured embedding provider.
def search_passages(
    question: str,
    provider: EmbeddingProvider,
    client: QdrantClient,
    limit: int | None = None,
    document_version_id: str | None = None,
) -> list[SearchResult]:
    if not question.strip():
        raise ValueError("The search question must not be blank.")

    limit = get_settings().retrieval_limit if limit is None else limit
    if limit < 1:
        raise ValueError("The result limit must be positive.")

    model_key = sha256(provider.model_id.encode("utf-8")).hexdigest()[:12]

    collection_name = f"document_chunks_{model_key}"

    if not client.collection_exists(collection_name):
        raise ValueError(
            f"Collection not found: {collection_name}. "
            "Index documents with this embedding configuration first."
        )

    query_vector = provider.embed_query(question)

    conditions = [
        models.FieldCondition(
            key="embedding_model_id",
            match=models.MatchValue(value=provider.model_id),
        )
    ]

    if document_version_id is not None:
        conditions.append(
            models.FieldCondition(
                key="document_version_id",
                match=models.MatchValue(value=document_version_id),
            )
        )

    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=models.Filter(must=conditions),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )

    results: list[SearchResult] = []

    for point in response.points:
        payload = point.payload

        if payload is None:
            raise RuntimeError("A search result has no source metadata.")

        results.append(
            SearchResult(
                chunk_id=payload["chunk_id"],
                document_version_id=payload["document_version_id"],
                source_file=payload["source_file"],
                item_ref=payload["item_ref"],
                page_numbers=tuple(payload["page_numbers"]),
                start=payload["start"],
                end=payload["end"],
                text=payload["text"],
                score=float(point.score),
            )
        )

    return results


# Select infrastructure and print retrieved evidence.
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search indexed document passages by meaning.",
    )
    parser.add_argument(
        "question",
        help="Natural-language search question.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of results (default: configured retrieval limit).",
    )
    parser.add_argument(
        "--document-version",
        help="Restrict results to a document_version_id.",
    )
    args = parser.parse_args()

    provider = OllamaEmbeddingProvider()
    client = QdrantClient(
        url=get_settings().qdrant_url,
        timeout=get_settings().qdrant_timeout,
    )

    try:
        results = search_passages(
            question=args.question,
            provider=provider,
            client=client,
            limit=args.limit,
            document_version_id=args.document_version,
        )
    finally:
        client.close()

    if not results:
        print("No passages found.")
        return

    for rank, result in enumerate(results, start=1):
        pages = ", ".join(str(page) for page in result.page_numbers)

        print(f"\n[{rank}] Similarity: {result.score:.4f}")
        print(
            f"Source: {result.source_file} | "
            f"Pages: {pages or 'unknown'} | "
            f"Item: {result.item_ref}"
        )
        print(f"Characters: {result.start}:{result.end}")
        print(f"Chunk: {result.chunk_id}")
        print(result.text)


if __name__ == "__main__":
    main()
