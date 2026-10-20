"""Thin HTTP boundary; ingestion runs only in the separate worker process."""

import logging
from hashlib import sha256
from pathlib import Path
from tempfile import SpooledTemporaryFile
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from document_analyser.config import get_settings
from document_analyser.generation.answer import GroundedAnswer
from document_analyser.generation.chat import handle_chunks
from document_analyser.generation.ollama_provider import OllamaGenerationProvider
from document_analyser.ingestion.storage import get_storage
from document_analyser.persistence.database import session_factory
from document_analyser.persistence.models import Document, Job, Version, new_id
from document_analyser.persistence.queries import version_chunks
from document_analyser.retrieval.providers.ollama_embeddings import (
    OllamaEmbeddingProvider,
)

logger = logging.getLogger(__name__)
app = FastAPI(title="Document Analyser", version="0.2.0")
SUPPORTED_SUFFIXES = {
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".html",
    ".md",
    ".txt",
    ".csv",
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
}


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    document_id: str = Field(min_length=1)
    message: str = Field(min_length=1, max_length=4000)


class SourceInfo(BaseModel):
    source_file: str
    page_numbers: list[int]
    item_ref: str
    chunk_id: str
    text: str


class ChatResponse(BaseModel):
    answer: GroundedAnswer
    sources: dict[str, SourceInfo]


# Create a new immutable processing version for an upload or explicit retry.
def queue_version(session: Session, document_id: str) -> Job:
    version = Version(
        document_id=document_id,
        config={
            "pipeline": "docling-v1",
            "chunking": "words-v1",
            **get_settings().processing_config().model_dump(),
        },
    )
    session.add(version)
    session.flush()
    job = Job(version_id=version.id)
    session.add(job)
    session.flush()
    return job


# Report API availability without claiming downstream services are healthy.
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# Preserve original bytes before atomically registering a document and queued job.
@app.post("/documents", status_code=202)
def upload_document(file: Annotated[UploadFile, File()]) -> dict:
    name = Path((file.filename or "document").replace("\\", "/")).name
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(415, "Unsupported document type.")
    limit = get_settings().max_upload_bytes
    storage = get_storage()
    key = f"originals/{new_id()}{suffix}"
    digest, size = sha256(), 0
    try:
        with SpooledTemporaryFile(max_size=8 * 1024 * 1024) as stream:
            while block := file.file.read(1024 * 1024):
                size += len(block)
                if size > limit:
                    raise HTTPException(413, "Document exceeds the upload size limit.")
                digest.update(block)
                stream.write(block)
            if not size:
                raise HTTPException(400, "The uploaded document is empty.")
            stream.seek(0)
            storage.put(key, stream)
        try:
            with session_factory().begin() as session:
                document = Document(
                    name=name, original_key=key, sha256=digest.hexdigest(), size=size
                )
                session.add(document)
                session.flush()
                job = queue_version(session, document.id)
                result = {
                    "document_id": document.id,
                    "job_id": job.id,
                    "version_id": job.version_id,
                }
        except Exception:
            storage.delete(key)
            raise
        return result
    finally:
        file.file.close()


# Include processing state so the UI can show documents before indexing finishes.
@app.get("/documents")
def list_documents() -> list[dict]:
    with session_factory()() as session:
        rows = session.execute(
            select(Document, Version, Job)
            .join(Version, Version.document_id == Document.id)
            .join(Job, Job.version_id == Version.id)
            .order_by(Version.created_at.desc())
        )
        # Only the newest version is displayed for each original document.
        documents: dict[str, dict] = {}
        for document, version, job in rows:
            documents.setdefault(
                document.id,
                {
                    "document_id": document.id,
                    "name": document.name,
                    "version_id": version.id,
                    "job_id": job.id,
                    "status": job.status,
                    "stage": job.stage,
                    "progress": job.progress,
                    "error": job.error,
                    "warning": job.warning,
                    "index_status": version.index_status,
                    "ner_status": version.ner_status,
                },
            )
        return list(documents.values())


# Expose durable status without leaking source text or server filesystem paths.
@app.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    with session_factory()() as session:
        job = session.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        return {
            "job_id": job.id,
            "version_id": job.version_id,
            "status": job.status,
            "stage": job.stage,
            "progress": job.progress,
            "error": job.error,
            "warning": job.warning,
        }


# Retry from the preserved original using a fresh processing version.
@app.post("/documents/{document_id}/retry", status_code=202)
def retry_document(document_id: str) -> dict:
    with session_factory().begin() as session:
        document = session.scalar(
            select(Document).where(Document.id == document_id).with_for_update()
        )
        if document is None:
            raise HTTPException(404, "Document not found")
        active = session.scalar(
            select(Job)
            .join(Version)
            .where(
                Version.document_id == document_id,
                Job.status.in_(["queued", "running"]),
            )
        )
        if active:
            raise HTTPException(409, "This document already has an active job")
        job = queue_version(session, document_id)
        return {
            "document_id": document_id,
            "job_id": job.id,
            "version_id": job.version_id,
        }


# Keep both questions and whole-document summaries inside the existing chat flow.
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    with session_factory()() as session:
        version = session.scalar(
            select(Version)
            .where(Version.document_id == request.document_id)
            .order_by(Version.created_at.desc())
            .limit(1)
        )
        if version is None:
            raise HTTPException(404, "Document not found")
        if version.index_status != "indexed":
            raise HTTPException(
                409, "Document indexing is not complete. Check processing status."
            )
        provider = OllamaEmbeddingProvider()
        if version.embedding_model != provider.model_id:
            raise HTTPException(
                409, "Embedding configuration changed. Rebuild this version's index."
            )
        chunks = version_chunks(session, version.id)
    try:
        return ChatResponse.model_validate(
            handle_chunks(
                request.message,
                chunks,
                provider,
                OllamaGenerationProvider(),
            )
        )
    except Exception as error:
        logger.exception("Chat processing failed")
        raise HTTPException(
            500, "Chat failed. Check the API logs for details."
        ) from error
