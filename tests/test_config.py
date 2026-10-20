"""Configuration wiring checks without live services or model downloads."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import select

from document_analyser.api.app import app, queue_version
from document_analyser.config import Settings, get_settings
from document_analyser.ingestion.storage import LocalStorage, get_storage
from document_analyser.ingestion.worker import persist_parsed, run_ner
from document_analyser.persistence.models import Chunk, Document, Version
from document_analyser.retrieval.chunking import chunk_spans


# Environment values override dotenv; list values use JSON and secrets stay masked.
def test_environment_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = tmp_path / ".env"
    env.write_text('GENERATION_MODEL=file-model\nNER_LABELS=["person"]\n')
    monkeypatch.setenv("GENERATION_MODEL", "environment-model")
    settings = Settings(_env_file=env)
    assert settings.generation_model == "environment-model"
    assert settings.ner_labels == ["person"]
    assert "analyser:analyser" not in repr(settings)


# Invalid related and individual limits fail before processing starts.
@pytest.mark.parametrize(
    "overrides",
    [
        {"chunk_max_words": 20, "chunk_overlap_words": 20},
        {"ner_window_tokens": 32, "ner_overlap_tokens": 32},
        {"generation_max_tokens": 8192},
        {"max_upload_bytes": 0},
        {"embedding_batch_size": 0},
        {"ner_threshold": 1.1},
        {"ner_labels": [""]},
        {"qdrant_timeout": -1},
    ],
)
def test_invalid_settings(overrides: dict) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **overrides)


# Each process caches validated settings until explicitly cleared or restarted.
def test_settings_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GENERATION_MODEL", "first")
    assert get_settings().generation_model == "first"
    monkeypatch.setenv("GENERATION_MODEL", "second")
    assert get_settings().generation_model == "first"
    get_settings.cache_clear()
    assert get_settings().generation_model == "second"


# Storage honours the shared root without creating directories during construction.
def test_storage_configuration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCUMENT_STORAGE_DIR", str(tmp_path / "storage"))
    storage = get_storage()
    assert isinstance(storage, LocalStorage)
    assert storage.root == tmp_path / "storage"
    assert not storage.root.exists()


# Configured models, generation options and explicit constructor overrides reach Ollama.
def test_provider_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    from document_analyser.generation.ollama_provider import OllamaGenerationProvider
    from document_analyser.retrieval.providers.ollama_embeddings import (
        OllamaEmbeddingProvider,
    )

    for key, value in {
        "OLLAMA_HOST": "http://ollama.test:11434",
        "GENERATION_MODEL": "test-generator",
        "GENERATION_TIMEOUT": "42",
        "GENERATION_TEMPERATURE": "0.25",
        "GENERATION_CONTEXT_TOKENS": "4096",
        "GENERATION_MAX_TOKENS": "512",
        "EMBEDDING_MODEL": "nomic-embed-text:test",
        "EMBEDDING_TIMEOUT": "17",
    }.items():
        monkeypatch.setenv(key, value)
    with patch("document_analyser.generation.ollama_provider.Client") as client:
        client.return_value.chat.return_value = SimpleNamespace(
            done_reason="stop", message=SimpleNamespace(content="{}")
        )
        provider = OllamaGenerationProvider()
        client.assert_called_once_with(host="http://ollama.test:11434", timeout=42)
        provider.generate_json("system", "user", {})
        args = client.return_value.chat.call_args.kwargs
        assert args["model"] == "test-generator"
        assert args["options"] == {
            "temperature": 0.25,
            "num_ctx": 4096,
            "num_predict": 512,
        }
        explicit = OllamaGenerationProvider(model="explicit", host="http://explicit")
        assert explicit.model == "explicit"
        assert client.call_args.kwargs["host"] == "http://explicit"
    with patch(
        "document_analyser.retrieval.providers.ollama_embeddings.Client"
    ) as client:
        provider = OllamaEmbeddingProvider()
        assert "nomic-embed-text:test" in provider.model_id
        client.assert_called_once_with(host="http://ollama.test:11434", timeout=17)
        OllamaEmbeddingProvider(host="http://explicit", timeout=9)
        client.assert_called_with(host="http://explicit", timeout=9)


# Queued versions retain actual chunking/NER parameters after the environment changes.
def test_processing_snapshot(database, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNK_MAX_WORDS", "4")
    monkeypatch.setenv("CHUNK_OVERLAP_WORDS", "1")
    monkeypatch.setenv("CHUNK_MAX_CHARACTERS", "20")
    monkeypatch.setenv("NER_MODEL", "synthetic-ner")
    monkeypatch.setenv("NER_LABELS", '["company"]')
    monkeypatch.setenv("NER_THRESHOLD", "0.25")
    monkeypatch.setenv("NER_WINDOW_TOKENS", "64")
    monkeypatch.setenv("NER_OVERLAP_TOKENS", "8")
    with database.begin() as session:
        document = Document(
            name="synthetic.md", original_key="synthetic", sha256="a" * 64, size=10
        )
        session.add(document)
        session.flush()
        job = queue_version(session, document.id)
        version_id = job.version_id
        snapshot = session.get(Version, version_id).config
        assert "database_url" not in snapshot
    monkeypatch.setenv("CHUNK_MAX_WORDS", "50")
    monkeypatch.setenv("NER_MODEL", "changed-ner")
    get_settings.cache_clear()
    parsed = MagicMock()
    text = "one two three four five six seven eight nine"
    parsed.iterate_items.return_value = [
        (SimpleNamespace(text=text, self_ref="#/texts/0", prov=[]), 0)
    ]
    with database.begin() as session:
        persist_parsed(session, version_id, parsed)
    with database() as session:
        chunks = session.scalars(select(Chunk).order_by(Chunk.ordinal)).all()
        assert chunks[0].text == "one two three four"
        assert chunks[1].text == "four five six seven"
        assert all(len(chunk.text) <= 20 for chunk in chunks)
    with (
        patch("document_analyser.extraction.ner.EntityExtractor") as extractor,
        patch("document_analyser.extraction.windowing.load_tokenizer") as tokenizer,
        patch(
            "document_analyser.extraction.document_ner.extract_passage_mentions",
            return_value=[],
        ) as extract,
    ):
        run_ner(database, version_id)
        extractor.assert_called_once_with("synthetic-ner", ("company",), 0.25)
        tokenizer.assert_called_once_with("microsoft/deberta-v3-large")
        assert extract.call_args.kwargs == {"window_tokens": 64, "overlap_tokens": 8}


# Runtime upload limits reject excess bytes before registering a document.
def test_upload_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "3")
    response = TestClient(app).post(
        "/documents", files={"file": ("synthetic.pdf", b"four")}
    )
    assert response.status_code == 413


# Source offsets, overlap, and character bounds remain correct for unusual text.
def test_chunk_character_limits() -> None:
    text = "a " + " " * 50 + "b " + "x" * 40
    spans = list(chunk_spans(text, 2, 1, 8))
    assert all(0 < end - start <= 8 and text[start:end].strip() for start, end in spans)
    covered = {i for start, end in spans for i in range(start, end)}
    assert all(i in covered for i, char in enumerate(text) if not char.isspace())


# Batch and retrieval limits propagate without touching a network service.
def test_batch_and_retrieval_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    from document_analyser.generation.summarise import create_batches
    from document_analyser.retrieval.index_document import index_chunk_records
    from document_analyser.retrieval.search import search_passages

    monkeypatch.setenv("SUMMARY_BATCH_CHARACTERS", "40")
    monkeypatch.setenv("RETRIEVAL_LIMIT", "7")
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "2")
    assert len(create_batches([{"text": "x" * 15}] * 3)) == 3
    client = MagicMock()
    client.query_points.return_value.points = []
    provider = MagicMock()
    provider.model_id = "test-embedding"
    search_passages("question", provider, client)
    assert client.query_points.call_args.kwargs["limit"] == 7
    search_passages("question", provider, client, limit=3)
    assert client.query_points.call_args.kwargs["limit"] == 3
    provider.embed_documents.side_effect = lambda texts: [[1.0, 0.5] for _ in texts]
    chunks = [
        {
            "chunk_id": str(i),
            "document_version_id": "v",
            "source_file": "synthetic",
            "item_ref": "x",
            "page_numbers": [],
            "start": 0,
            "end": 1,
            "text": "x",
        }
        for i in range(5)
    ]
    with patch("document_analyser.retrieval.index_document.prepare_collection"):
        index_chunk_records(chunks, provider, client)
    assert [len(call.args[0]) for call in provider.embed_documents.call_args_list] == [
        2,
        2,
        1,
    ]


# NER uses configured labels/threshold and retains explicit per-call overrides.
def test_ner_and_window_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    from document_analyser.extraction.ner import EntityExtractor
    from document_analyser.extraction.windowing import create_windows, load_tokenizer

    monkeypatch.setenv("NER_MODEL", "synthetic-model")
    monkeypatch.setenv("NER_LABELS", '["company"]')
    monkeypatch.setenv("NER_THRESHOLD", "0.25")
    monkeypatch.setenv("NER_TOKENIZER", "synthetic-tokenizer")
    monkeypatch.setenv("NER_WINDOW_TOKENS", "3")
    monkeypatch.setenv("NER_OVERLAP_TOKENS", "1")
    with patch("document_analyser.extraction.ner.GLiNER.from_pretrained") as load:
        load.return_value.predict_entities.return_value = []
        extractor = EntityExtractor()
        load.assert_called_once_with("synthetic-model")
        extractor.extract("Aster")
        load.return_value.predict_entities.assert_called_with(
            "Aster", ["company"], threshold=0.25
        )
        extractor.extract("Aster", labels=("person",), threshold=0.0)
        load.return_value.predict_entities.assert_called_with(
            "Aster", ["person"], threshold=0.0
        )
    with patch(
        "document_analyser.extraction.windowing.AutoTokenizer.from_pretrained"
    ) as load:
        tokenizer = load_tokenizer()
        load.assert_called_once_with("synthetic-tokenizer", use_fast=True)
        tokenizer.encode.side_effect = lambda text, **kwargs: text.split()
        windows = create_windows("one two three four five six", tokenizer)
        assert [window.text for window in windows] == [
            "one two three",
            "three four five",
            "five six",
        ]
        explicit = create_windows(
            "one two three four", tokenizer, max_tokens=2, overlap_tokens=0
        )
        assert [window.text for window in explicit] == ["one two", "three four"]


# The example file documents every setting and parses using the current defaults.
def test_example_config() -> None:
    from dotenv import dotenv_values

    values = dotenv_values(".env.example")
    assert set(values) == {name.upper() for name in Settings.model_fields}
    settings = Settings(_env_file=".env.example")
    assert settings.model_dump() == Settings(_env_file=None).model_dump()
