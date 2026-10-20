from hashlib import sha256
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from document_analyser.config import get_settings
from document_analyser.retrieval.embeddings import EmbeddingProvider
from document_analyser.retrieval.providers.ollama_embeddings import (
    OllamaEmbeddingProvider,
)


# Store sample embeddings and retrieve passages for a question.
def check_vector_search(provider: EmbeddingProvider) -> None:
    passages = [
        "Supporting receipts must be retained for five years.",
        "The company's registered office is in Sydney.",
        "The director was appointed in March 2019.",
    ]
    question = "How long should I keep proof of expenses?"

    vectors = provider.embed_documents(passages)
    query_vector = provider.embed_query(question)

    dimension = len(query_vector)

    if any(len(vector) != dimension for vector in vectors):
        raise ValueError("Document and query dimensions differ.")

    model_key = sha256(provider.model_id.encode("utf-8")).hexdigest()[:12]

    collection = f"retrieval_demo_{model_key}"

    client = QdrantClient(
        url=get_settings().qdrant_url,
        timeout=get_settings().qdrant_timeout,
    )

    try:
        if not client.collection_exists(collection):
            client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE,
                ),
            )

        config = client.get_collection(collection).config.params.vectors

        if not isinstance(config, models.VectorParams):
            raise TypeError("Expected a single unnamed vector.")

        if config.size != dimension or config.distance != models.Distance.COSINE:
            raise ValueError("Collection vector configuration differs.")

        points = [
            models.PointStruct(
                id=str(
                    uuid5(
                        NAMESPACE_URL,
                        f"{provider.model_id}|demo-v1|{index}",
                    )
                ),
                vector=vector,
                payload={
                    "text": text,
                    "model_id": provider.model_id,
                    "dataset": "demo-v1",
                },
            )
            for index, (text, vector) in enumerate(zip(passages, vectors, strict=True))
        ]

        client.upsert(
            collection_name=collection,
            points=points,
            wait=True,
        )

        result = client.query_points(
            collection_name=collection,
            query=query_vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="model_id",
                        match=models.MatchValue(
                            value=provider.model_id,
                        ),
                    ),
                    models.FieldCondition(
                        key="dataset",
                        match=models.MatchValue(
                            value="demo-v1",
                        ),
                    ),
                ]
            ),
            limit=3,
            with_payload=True,
        )

        print(f"Collection: {collection}")
        print(f"Question: {question}\n")

        for rank, point in enumerate(result.points, start=1):
            payload = point.payload or {}
            print(f"{rank}. score={point.score:.4f} {payload.get('text', '')}")
    finally:
        client.close()


# Choose the embedding provider at the application entry point.
def main() -> None:
    provider = OllamaEmbeddingProvider()
    check_vector_search(provider)


if __name__ == "__main__":
    main()
