from sqlalchemy import select
from sqlalchemy.orm import Session

from document_analyser.persistence.models import Chunk, Document, Passage, Version


# Load ordered, authoritative chunks with source metadata for chat and rebuilding.
def version_chunks(session: Session, version_id: str) -> list[dict]:
    rows = session.execute(
        select(Chunk, Passage, Document)
        .join(Passage, Chunk.passage_id == Passage.id)
        .join(Version, Chunk.version_id == Version.id)
        .join(Document, Version.document_id == Document.id)
        .where(Chunk.version_id == version_id)
        .order_by(Chunk.ordinal)
    )
    return [
        {
            "chunk_id": c.id,
            "document_version_id": c.version_id,
            "source_file": d.name,
            "item_ref": p.item_ref,
            "page_numbers": p.page_numbers,
            "start": c.start,
            "end": c.end,
            "text": c.text,
        }
        for c, p, d in rows
    ]
