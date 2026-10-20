from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from sqlalchemy import func, select

from document_analyser.api.app import app
from document_analyser.ingestion.worker import claim_job, process_job, rebuild_index
from document_analyser.persistence.models import Chunk, Document, Job, Passage, Version
from document_analyser.persistence.queries import version_chunks


class Embeddings:
    model_id = "test/embedding/v1"

    # Produce repeatable synthetic vectors for local projection tests.
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.5, 0.25] for _text in texts]


# Provide a minimal parsed document without invoking Docling or model downloads.
class Parsed:
    # Include enough text to exercise overlapping chunks and summary batching.
    def iterate_items(self):
        yield (
            SimpleNamespace(
                text="Synthetic evidence " * 180, self_ref="#/texts/0", prov=[]
            ),
            0,
        )

    # Preserve an artifact as the real parser would.
    def save_as_json(self, target):
        target.write_text('{"synthetic": true}')


# Upload crosses the API boundary and creates exactly one durable queued version.
def upload(database, storage):
    with (
        patch("document_analyser.api.app.session_factory", return_value=database),
        patch("document_analyser.api.app.get_storage", return_value=storage),
        TestClient(app) as client,
    ):
        response = client.post(
            "/documents", files={"file": ("../sample.pdf", b"synthetic-original")}
        )
        assert response.status_code == 202
        return response.json()


# An NER failure must preserve source records, complete indexing and allow rebuilding.
def test_upload_worker_ner_failure_and_rebuild(database, storage, tmp_path):
    uploaded = upload(database, storage)
    with database() as session:
        document = session.get(Document, uploaded["document_id"])
        assert document.name == "sample.pdf"
        storage.materialize(document.original_key, tmp_path / "original")
        assert (tmp_path / "original").read_bytes() == b"synthetic-original"
    job_id = claim_job(database)
    assert job_id == uploaded["job_id"]
    assert claim_job(database) is None
    client = QdrantClient(":memory:")
    # Mock only the heavyweight parser and NER; exercise real SQL and Qdrant indexing.
    with (
        patch(
            "document_analyser.ingestion.parser.parse_document", return_value=Parsed()
        ),
        patch(
            "document_analyser.ingestion.worker.run_ner",
            side_effect=RuntimeError("model unavailable"),
        ),
    ):
        process_job(database, storage, job_id, Embeddings(), client)
    with database() as session:
        job = session.get(Job, job_id)
        version = session.get(Version, job.version_id)
        assert (job.status, job.progress) == ("completed", 100)
        assert job.warning and version.ner_status == "failed"
        assert version.index_status == "indexed"
        assert session.scalar(select(func.count()).select_from(Passage)) == 1
        chunks = version_chunks(session, version.id)
        assert len(chunks) >= 3
        for chunk in session.scalars(select(Chunk)):
            passage = session.get(Passage, chunk.passage_id)
            assert chunk.text == passage.text[chunk.start : chunk.end]
        collection = version.collection
    client.delete_collection(collection)
    rebuild_index(database, uploaded["version_id"], Embeddings(), client)
    assert client.count(collection).count == len(chunks)
    rebuild_index(database, uploaded["version_id"], Embeddings(), client)
    assert client.count(collection).count == len(chunks)
    client.close()


# Validation errors cannot register documents or enqueue unusable work.
@pytest.mark.parametrize(
    ("name", "content", "status"), [("x.exe", b"x", 415), ("x.pdf", b"", 400)]
)
def test_invalid_upload(database, storage, name, content, status):
    with (
        patch("document_analyser.api.app.session_factory", return_value=database),
        patch("document_analyser.api.app.get_storage", return_value=storage),
    ):
        response = TestClient(app).post("/documents", files={"file": (name, content)})
    assert response.status_code == status
    with database() as session:
        assert session.scalar(select(func.count()).select_from(Document)) == 0


# Immutable keys and traversal checks protect stored originals.
def test_storage_guards(storage):
    storage.put("originals/x.pdf", BytesIO(b"original"))
    with pytest.raises(FileExistsError):
        storage.put("originals/x.pdf", BytesIO(b"overwrite"))
    with pytest.raises(ValueError):
        storage.put("../escape", BytesIO(b"escape"))


# Failed parsing remains visible and can be retried as a new processing version.
def test_failure_and_retry(database, storage):
    uploaded = upload(database, storage)
    job_id = claim_job(database)
    with patch(
        "document_analyser.ingestion.parser.parse_document",
        side_effect=ValueError("bad PDF"),
    ):
        process_job(database, storage, job_id, Embeddings(), QdrantClient(":memory:"))
    with patch("document_analyser.api.app.session_factory", return_value=database):
        client = TestClient(app)
        status = client.get(f"/jobs/{job_id}").json()
        assert status["status"] == "failed" and status["error"]
        retry = client.post(f"/documents/{uploaded['document_id']}/retry")
        assert retry.status_code == 202
        assert retry.json()["version_id"] != uploaded["version_id"]
        assert (
            client.post(f"/documents/{uploaded['document_id']}/retry").status_code
            == 409
        )


# Exercise real Docling conversion of synthetic markdown without PDF model downloads.
def test_real_docling_markdown(database, storage, tmp_path) -> None:
    from document_analyser.ingestion.parser import parse_document
    from document_analyser.ingestion.worker import persist_parsed

    source = tmp_path / "synthetic.md"
    source.write_text(
        "# Synthetic report\n\nAster operates in Sydney.\n\n| Item | Value |\n| --- | --- |\n| Revenue | 42 |\n"
    )
    parsed = parse_document(source)
    uploaded = upload(database, storage)
    with database.begin() as session:
        persist_parsed(session, uploaded["version_id"], parsed)
    with database() as session:
        chunks = version_chunks(session, uploaded["version_id"])
        assert any("Sydney" in c["text"] for c in chunks)
        assert any("Revenue" in c["text"] for c in chunks)


# Persist successful NER only as unreviewed candidates with model provenance.
def test_ner_candidate_provenance(database, storage) -> None:
    from document_analyser.ingestion.worker import persist_parsed, run_ner
    from document_analyser.persistence.models import Mention

    uploaded = upload(database, storage)
    with database.begin() as session:
        persist_parsed(session, uploaded["version_id"], Parsed())
    mention = SimpleNamespace(
        text="Synthetic", label="organization name", start=0, end=9, score=0.6
    )
    with (
        patch("document_analyser.extraction.ner.EntityExtractor"),
        patch("document_analyser.extraction.windowing.load_tokenizer"),
        patch(
            "document_analyser.extraction.document_ner.extract_passage_mentions",
            return_value=[mention],
        ),
    ):
        run_ner(database, uploaded["version_id"])
    with database() as session:
        candidate = session.scalar(select(Mention))
        assert candidate.review_status == "unreviewed"
        assert candidate.model_provenance["model"] == "urchade/gliner_large-v2.1"
        assert candidate.model_provenance["threshold"] == 0.5
        assert session.get(Version, uploaded["version_id"]).ner_status == "completed"


# A vector failure cannot expose partial indexing as a chat-ready document.
def test_index_failure_retains_sources(database, storage) -> None:
    from document_analyser.ingestion.worker import persist_parsed

    uploaded = upload(database, storage)
    with database.begin() as session:
        persist_parsed(session, uploaded["version_id"], Parsed())
    with (
        patch(
            "document_analyser.ingestion.worker.index_chunk_records",
            side_effect=ConnectionError("offline"),
        ),
        pytest.raises(ConnectionError),
    ):
        rebuild_index(database, uploaded["version_id"], Embeddings(), None)
    with database() as session:
        assert session.get(Version, uploaded["version_id"]).index_status == "failed"
        assert version_chunks(session, uploaded["version_id"])
    with patch("document_analyser.api.app.session_factory", return_value=database):
        response = TestClient(app).post(
            "/chat",
            json={"document_id": uploaded["document_id"], "message": "Question?"},
        )
        assert response.status_code == 409
