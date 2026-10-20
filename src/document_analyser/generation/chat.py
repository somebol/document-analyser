import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict
from qdrant_client import QdrantClient

from document_analyser.config import get_settings
from document_analyser.generation.answer import answer_question
from document_analyser.generation.providers import GenerationProvider
from document_analyser.generation.summarise import summarise_chunks
from document_analyser.retrieval.embeddings import EmbeddingProvider
from document_analyser.retrieval.search import search_passages


class ChatRoute(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["document_summary", "question"]


# Choose between whole-document coverage and targeted retrieval.
def choose_route(
    message: str,
    provider: GenerationProvider,
) -> ChatRoute:
    raw = provider.generate_json(
        system_prompt=(
            "Classify the user's request; do not answer it. "
            "Choose document_summary for an overall document summary, "
            "overview, main points or a summary of the entire document. "
            "Choose question for specific questions and summaries focused "
            "on a particular topic, such as payment obligations. "
            "Return only JSON matching the schema."
        ),
        user_prompt=json.dumps(
            {
                "request": message,
                "schema": ChatRoute.model_json_schema(),
            }
        ),
        schema=ChatRoute.model_json_schema(),
    )

    return ChatRoute.model_validate_json(raw)


# Execute a bounded workflow for the selected document version.
def handle_message(
    message: str,
    chunks_path: Path,
    embeddings: EmbeddingProvider,
    generator: GenerationProvider,
) -> dict:
    if not message.strip():
        raise ValueError("Enter a question or summary request.")

    return handle_chunks(
        message,
        json.loads(chunks_path.read_text(encoding="utf-8")),
        embeddings,
        generator,
    )


# Use authoritative chunks for summaries and to validate retrieved source evidence.
def handle_chunks(
    message: str,
    chunks: list[dict],
    embeddings: EmbeddingProvider,
    generator: GenerationProvider,
) -> dict:

    if not isinstance(chunks, list) or not chunks:
        raise ValueError("The selected document has no chunks.")

    versions = {chunk["document_version_id"] for chunk in chunks}

    if len(versions) != 1:
        raise ValueError("The chunk file must contain one document version.")

    route = choose_route(message, generator)

    if route.action == "document_summary":
        result = summarise_chunks(
            chunks,
            generator,
            instruction=message,
        )

        return {
            "answer": result["summary"],
            "sources": result["sources"],
        }

    client = QdrantClient(
        url=get_settings().qdrant_url,
        timeout=get_settings().qdrant_timeout,
    )

    try:
        evidence = search_passages(
            question=message,
            provider=embeddings,
            client=client,
            limit=get_settings().retrieval_limit,
            document_version_id=next(iter(versions)),
        )
    finally:
        client.close()

    # Qdrant supplies ranking only; text and references always come from PostgreSQL.
    from dataclasses import replace

    authoritative = {chunk["chunk_id"]: chunk for chunk in chunks}
    evidence = [
        replace(
            item,
            **{
                **authoritative[item.chunk_id],
                "page_numbers": tuple(authoritative[item.chunk_id]["page_numbers"]),
            },
        )
        for item in evidence
        if item.chunk_id in authoritative
    ]
    answer = answer_question(message, evidence, generator)

    return {
        "answer": answer.model_dump(),
        "sources": {
            str(number): {
                "source_file": item.source_file,
                "page_numbers": list(item.page_numbers),
                "item_ref": item.item_ref,
                "chunk_id": item.chunk_id,
                "text": item.text,
            }
            for number, item in enumerate(evidence, start=1)
        },
    }
