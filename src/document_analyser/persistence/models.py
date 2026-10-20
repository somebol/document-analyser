"""Relational records; vectors are a disposable projection of Chunk rows."""

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import JSON, DateTime, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


# Generate opaque identifiers independently of filenames and document content.
def new_id() -> str:
    return str(uuid4())


# Use UTC consistently for durable processing timestamps.
def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(Text)
    original_key: Mapped[str] = mapped_column(Text, unique=True)
    sha256: Mapped[str] = mapped_column(String(64))
    size: Mapped[int]
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )


class Version(Base):
    __tablename__ = "processing_versions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), index=True)
    config: Mapped[dict] = mapped_column(JSON)
    artifact_key: Mapped[str | None] = mapped_column(Text)
    embedding_model: Mapped[str | None] = mapped_column(Text)
    collection: Mapped[str | None] = mapped_column(Text)
    index_status: Mapped[str] = mapped_column(String(24), default="pending")
    ner_status: Mapped[str] = mapped_column(String(24), default="pending")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )


class Job(Base):
    __tablename__ = "processing_jobs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    version_id: Mapped[str] = mapped_column(
        ForeignKey("processing_versions.id"), unique=True
    )
    status: Mapped[str] = mapped_column(String(24), default="queued", index=True)
    stage: Mapped[str] = mapped_column(String(32), default="queued")
    progress: Mapped[int] = mapped_column(default=0)
    error: Mapped[str | None] = mapped_column(Text)
    warning: Mapped[str | None] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )


class Passage(Base):
    __tablename__ = "passages"
    __table_args__ = (UniqueConstraint("version_id", "ordinal"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    version_id: Mapped[str] = mapped_column(
        ForeignKey("processing_versions.id"), index=True
    )
    ordinal: Mapped[int]
    item_ref: Mapped[str] = mapped_column(Text)
    page_numbers: Mapped[list] = mapped_column(JSON)
    provenance: Mapped[list] = mapped_column(JSON)
    text: Mapped[str] = mapped_column(Text)


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (UniqueConstraint("version_id", "ordinal"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    version_id: Mapped[str] = mapped_column(
        ForeignKey("processing_versions.id"), index=True
    )
    passage_id: Mapped[str] = mapped_column(ForeignKey("passages.id"))
    ordinal: Mapped[int]
    start: Mapped[int]
    end: Mapped[int]
    text: Mapped[str] = mapped_column(Text)
    index_status: Mapped[str] = mapped_column(String(24), default="pending")


class Mention(Base):
    __tablename__ = "ner_mentions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    passage_id: Mapped[str] = mapped_column(ForeignKey("passages.id"), index=True)
    text: Mapped[str] = mapped_column(Text)
    label: Mapped[str] = mapped_column(Text)
    start: Mapped[int]
    end: Mapped[int]
    score: Mapped[float]
    review_status: Mapped[str] = mapped_column(String(24), default="unreviewed")
    model_provenance: Mapped[dict] = mapped_column(JSON)
