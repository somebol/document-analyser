"""Separate durable-queue worker: python -m document_analyser.ingestion.worker."""

import argparse
import importlib.metadata
import logging
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from qdrant_client import QdrantClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from document_analyser.config import ProcessingConfig, get_settings
from document_analyser.ingestion.storage import Storage, get_storage
from document_analyser.persistence.database import session_factory
from document_analyser.persistence.models import (
    Chunk,
    Document,
    Job,
    Mention,
    Passage,
    Version,
    utcnow,
)
from document_analyser.persistence.queries import version_chunks
from document_analyser.retrieval.chunking import chunk_spans
from document_analyser.retrieval.embeddings import EmbeddingProvider
from document_analyser.retrieval.index_document import index_chunk_records
from document_analyser.retrieval.providers.ollama_embeddings import (
    OllamaEmbeddingProvider,
)

logger = logging.getLogger(__name__)


# Serialize queue claims using PostgreSQL row locks; model work runs outside transactions.
def claim_job(factory: sessionmaker[Session]) -> str | None:
    with factory.begin() as session:
        job = session.scalar(
            select(Job)
            .where(Job.status == "queued")
            .order_by(Job.updated_at)
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        if job is None:
            return None
        job.status, job.stage, job.updated_at = "running", "parsing", utcnow()
        return job.id


# Persist observable progress independently of expensive model operations.
def progress(
    factory: sessionmaker[Session], job_id: str, stage: str, percent: int
) -> None:
    with factory.begin() as session:
        job = session.get(Job, job_id)
        job.stage, job.progress, job.updated_at = stage, percent, utcnow()


# Persist parsed source items (including table text and full Docling provenance).
def persist_parsed(session: Session, version_id: str, document: object) -> None:
    version = session.get(Version, version_id)
    config = ProcessingConfig.model_validate(version.config)
    ordinal = 0
    chunk_ordinal = 0
    for item, _level in document.iterate_items():
        text = getattr(item, "text", "")
        if not text and getattr(getattr(item, "label", None), "value", None) == "table":
            text = item.export_to_markdown(doc=document)
        if not text.strip():
            continue
        provenance = [p.model_dump(mode="json") for p in item.prov]
        passage = Passage(
            version_id=version_id,
            ordinal=ordinal,
            item_ref=item.self_ref,
            page_numbers=sorted({p.page_no for p in item.prov}),
            provenance=provenance,
            text=text,
        )
        session.add(passage)
        session.flush()
        for start, end in chunk_spans(
            text,
            config.max_words,
            config.overlap_words,
            config.max_characters,
        ):
            session.add(
                Chunk(
                    version_id=version_id,
                    passage_id=passage.id,
                    ordinal=chunk_ordinal,
                    start=start,
                    end=end,
                    text=text[start:end],
                )
            )
            chunk_ordinal += 1
        ordinal += 1
    if chunk_ordinal == 0:
        raise ValueError("Docling found no readable text or tables in this document.")


# Retain source-linked candidates with explicit provenance; never treat them as facts.
def run_ner(factory: sessionmaker[Session], version_id: str) -> None:
    from document_analyser.extraction.document_ner import extract_passage_mentions
    from document_analyser.extraction.ner import EntityExtractor
    from document_analyser.extraction.windowing import load_tokenizer
    from document_analyser.ingestion.passages import SourcePassage

    with factory() as session:
        config = ProcessingConfig.model_validate(
            session.get(Version, version_id).config
        )
    extractor = EntityExtractor(
        config.ner_model, tuple(config.ner_labels), config.ner_threshold
    )
    tokenizer = load_tokenizer(config.ner_tokenizer)
    provenance = {
        "model": config.ner_model,
        "library": "gliner",
        "library_version": importlib.metadata.version("gliner"),
        "tokenizer": config.ner_tokenizer,
        "labels": config.ner_labels,
        "threshold": config.ner_threshold,
        "window_tokens": config.ner_window_tokens,
        "overlap_tokens": config.ner_overlap_tokens,
    }
    with factory.begin() as session:
        passages = session.scalars(
            select(Passage).where(Passage.version_id == version_id)
        ).all()
        for passage in passages:
            source = SourcePassage(
                "", passage.item_ref, tuple(passage.page_numbers), passage.text
            )
            for mention in extract_passage_mentions(
                source,
                extractor,
                tokenizer,
                window_tokens=config.ner_window_tokens,
                overlap_tokens=config.ner_overlap_tokens,
            ):
                session.add(
                    Mention(
                        passage_id=passage.id,
                        text=mention.text,
                        label=mention.label,
                        start=mention.start,
                        end=mention.end,
                        score=mention.score,
                        model_provenance=provenance,
                    )
                )
        session.get(Version, version_id).ner_status = "completed"


# Recreate vectors solely from authoritative database chunks with idempotent upserts.
def rebuild_index(
    factory: sessionmaker[Session],
    version_id: str,
    provider: EmbeddingProvider,
    client: QdrantClient,
) -> None:
    with factory.begin() as session:
        version = session.get(Version, version_id)
        if version is None:
            raise ValueError("Processing version not found")
        version.index_status = "indexing"
        version.embedding_model = provider.model_id
        chunks = version_chunks(session, version_id)
        for chunk in session.scalars(
            select(Chunk).where(Chunk.version_id == version_id)
        ):
            chunk.index_status = "pending"
    try:
        collection = index_chunk_records(chunks, provider, client)
    except Exception:
        with factory.begin() as session:
            session.get(Version, version_id).index_status = "failed"
        raise
    with factory.begin() as session:
        version = session.get(Version, version_id)
        version.index_status, version.collection = "indexed", collection
        for chunk in session.scalars(
            select(Chunk).where(Chunk.version_id == version_id)
        ):
            chunk.index_status = "indexed"


# Run one claimed version; NER errors are isolated from the RAG-critical path.
def process_job(
    factory: sessionmaker[Session],
    storage: Storage,
    job_id: str,
    provider: EmbeddingProvider,
    client: QdrantClient,
) -> None:
    try:
        from document_analyser.ingestion.parser import parse_document

        with factory() as session:
            job = session.get(Job, job_id)
            version = session.get(Version, job.version_id)
            document = session.get(Document, version.document_id)
            version_id, key, name = version.id, document.original_key, document.name
        with TemporaryDirectory() as directory:
            source = Path(directory) / ("source" + Path(name).suffix.lower())
            storage.materialize(key, source)
            parsed = parse_document(source)
            artifact = Path(directory) / "docling.json"
            parsed.save_as_json(artifact)
            artifact_key = f"versions/{version_id}/docling.json"
            with artifact.open("rb") as stream:
                storage.put(artifact_key, stream)
            progress(factory, job_id, "chunking", 30)
            with factory.begin() as session:
                version = session.get(Version, version_id)
                version.artifact_key = artifact_key
                version.config = {
                    **version.config,
                    "docling_version": importlib.metadata.version("docling"),
                }
                persist_parsed(session, version_id, parsed)
        progress(factory, job_id, "ner", 45)
        try:
            run_ner(factory, version_id)
        except Exception:
            logger.exception("NER failed for job %s; continuing with indexing", job_id)
            with factory.begin() as session:
                session.get(Version, version_id).ner_status = "failed"
                session.get(
                    Job, job_id
                ).warning = "NER failed; candidates are unavailable. RAG remains enabled. See worker logs."
        progress(factory, job_id, "embedding_and_indexing", 65)
        rebuild_index(factory, version_id, provider, client)
        with factory.begin() as session:
            job = session.get(Job, job_id)
            job.status, job.stage, job.progress, job.updated_at = (
                "completed",
                "completed",
                100,
                utcnow(),
            )
    except Exception:
        logger.exception("Ingestion failed for job %s", job_id)
        with factory.begin() as session:
            job = session.get(Job, job_id)
            job.status = "failed"
            job.error = (
                f"Processing failed during {job.stage}. See worker logs for details."
            )
            job.updated_at = utcnow()


# Process one document at a time to limit model memory on local Apple Silicon.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--reindex", metavar="VERSION_ID")
    parser.add_argument(
        "--fail-interrupted",
        action="store_true",
        help="With all workers stopped, mark interrupted jobs failed.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    factory = session_factory()
    if args.fail_interrupted:
        with factory.begin() as session:
            for job in session.scalars(select(Job).where(Job.status == "running")):
                job.status, job.error = (
                    "failed",
                    "Worker interrupted. Retry to create a new processing version.",
                )
        return
    client = QdrantClient(
        url=get_settings().qdrant_url, timeout=get_settings().qdrant_timeout
    )
    try:
        provider = OllamaEmbeddingProvider()
        if args.reindex:
            rebuild_index(factory, args.reindex, provider, client)
            return
        while True:
            job_id = claim_job(factory)
            if job_id:
                process_job(factory, get_storage(), job_id, provider, client)
            if args.once:
                break
            if job_id is None:
                time.sleep(get_settings().worker_poll_seconds)
    finally:
        client.close()


if __name__ == "__main__":
    main()
