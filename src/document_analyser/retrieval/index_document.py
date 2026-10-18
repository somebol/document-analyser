import argparse
import json
import logging
from hashlib import sha256
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from document_analyser.retrieval.embeddings import EmbeddingProvider
from document_analyser.retrieval.providers.ollama_embeddings import (
    OllamaEmbeddingProvider,
)

logger = logging.getLogger(__name__)

BATCH_SIZE = 16


# Create or validate a collection for this embedding configuration.
def prepare_collection(
    client: QdrantClient,
    collection_name: str,
    dimensions: int,
) -> None:
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=dimensions,
                distance=models.Distance.COSINE,
            ),
        )

    config = client.get_collection(
        collection_name
    ).config.params.vectors

    if not isinstance(config, models.VectorParams):
        raise ValueError("Expected a single unnamed vector.")

    if (
        config.size != dimensions
        or config.distance != models.Distance.COSINE
    ):
        raise ValueError("Collection vector configuration differs.")


# Embed chunks in batches and upsert source-linked Qdrant points.
def index_chunks(
    source: Path,
    provider: EmbeddingProvider,
    client: QdrantClient,
) -> str:
    chunks = json.loads(source.read_text(encoding="utf-8"))

    if not isinstance(chunks, list) or not chunks:
        raise ValueError("Expected a nonempty JSON list of chunks.")

    required_fields = {
        "chunk_id",
        "document_version_id",
        "source_file",
        "item_ref",
        "page_numbers",
        "start",
        "end",
        "text",
    }

    for chunk in chunks:
        if not isinstance(chunk, dict) or not required_fields <= chunk.keys():
            raise ValueError("A chunk is missing required fields.")

        if not isinstance(chunk["text"], str) or not chunk["text"].strip():
            raise ValueError("Chunk text must be a nonempty string.")

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("The input contains duplicate chunk IDs.")

    model_key = sha256(
        provider.model_id.encode("utf-8")
    ).hexdigest()[:12]

    collection_name = f"document_chunks_{model_key}"
    dimensions: int | None = None

    for offset in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[offset:offset + BATCH_SIZE]

        vectors = provider.embed_documents(
            [chunk["text"] for chunk in batch]
        )

        if len(vectors) != len(batch):
            raise ValueError("Embedding count does not match chunk count.")

        if dimensions is None:
            dimensions = len(vectors[0])

            if dimensions == 0:
                raise ValueError("Embedding vectors must not be empty.")

            prepare_collection(
                client,
                collection_name,
                dimensions,
            )

        if any(len(vector) != dimensions for vector in vectors):
            raise ValueError("Embedding dimensions changed between chunks.")

        points = [
            models.PointStruct(
                id=str(
                    uuid5(
                        NAMESPACE_URL,
                        f"{provider.model_id}|{chunk['chunk_id']}",
                    )
                ),
                vector=vector,
                payload={
                    **chunk,
                    "embedding_model_id": provider.model_id,
                },
            )
            for chunk, vector in zip(batch, vectors, strict=True)
        ]

        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )

        logger.info(
            "Indexed %d/%d chunks",
            offset + len(batch),
            len(chunks),
        )

    return collection_name


# Select infrastructure and run indexing from the command line.
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed retrieval chunks and index them in Qdrant.",
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Path to the retrieval chunk JSON.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    provider = OllamaEmbeddingProvider()
    client = QdrantClient(
        url="http://localhost:6333",
        timeout=60,
    )

    try:
        collection = index_chunks(
            args.source,
            provider,
            client,
        )
        print(f"\nIndexed into collection: {collection}")
    finally:
        client.close()


if __name__ == "__main__":
    main()