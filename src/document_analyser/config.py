from functools import lru_cache
from pathlib import Path
from typing import Self

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProcessingConfig(BaseModel):
    """Nonsecret processing parameters persisted with each immutable version."""

    max_words: int = Field(default=120, gt=0)
    overlap_words: int = Field(default=20, ge=0)
    max_characters: int = Field(default=1600, gt=0)
    ner_model: str = Field(default="urchade/gliner_large-v2.1", min_length=1)
    ner_labels: list[str] = Field(
        default_factory=lambda: [
            "person name",
            "organization name",
            "geographical location",
        ],
        min_length=1,
    )
    ner_threshold: float = Field(default=0.5, ge=0, le=1)
    ner_tokenizer: str = Field(default="microsoft/deberta-v3-large", min_length=1)
    ner_window_tokens: int = Field(default=192, gt=0)
    ner_overlap_tokens: int = Field(default=32, ge=0)

    # Reject configurations that would prevent chunk/window iteration advancing.
    @model_validator(mode="after")
    def validate_overlaps(self) -> Self:
        if self.overlap_words >= self.max_words:
            raise ValueError("chunk_overlap_words must be less than chunk_max_words")
        if self.ner_overlap_tokens >= self.ner_window_tokens:
            raise ValueError("ner_overlap_tokens must be less than ner_window_tokens")
        if any(not label.strip() for label in self.ner_labels):
            raise ValueError("ner_labels must contain nonblank labels")
        return self


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Infrastructure and storage
    database_url: SecretStr = SecretStr(
        "postgresql+psycopg://analyser:analyser@localhost:5432/analyser"
    )
    document_storage_dir: Path = Path("data/storage")
    document_api_url: str = "http://127.0.0.1:8000"
    qdrant_url: str = "http://localhost:6333"
    max_upload_bytes: int = Field(default=104_857_600, gt=0)

    # Embedding and generation models
    ollama_host: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text:v1.5"
    generation_model: str = "qwen3:8b"

    embedding_timeout: float = Field(default=120.0, gt=0)
    qdrant_timeout: float = Field(default=120.0, gt=0)
    generation_timeout: float = Field(default=300.0, gt=0)
    generation_temperature: float = Field(default=0.0, ge=0, le=2)
    generation_context_tokens: int = Field(default=8192, gt=0)
    generation_max_tokens: int = Field(default=1500, gt=0)
    embedding_batch_size: int = Field(default=16, gt=0)
    retrieval_limit: int = Field(default=5, gt=0)
    summary_batch_characters: int = Field(default=10000, gt=0)
    summary_max_levels: int = Field(default=8, gt=0)
    worker_poll_seconds: float = Field(default=2.0, gt=0)
    ui_poll_seconds: float = Field(default=3.0, gt=0)
    api_request_timeout: float = Field(default=10.0, gt=0)
    upload_request_timeout: float = Field(default=120.0, gt=0)
    chat_request_timeout: float = Field(default=900.0, gt=0)

    # Chunking
    chunk_max_words: int = Field(default=120, gt=0)
    chunk_overlap_words: int = Field(default=20, ge=0)
    chunk_max_characters: int = Field(default=1600, gt=0)

    # Unreviewed NER candidate extraction
    ner_model: str = "urchade/gliner_large-v2.1"
    ner_threshold: float = Field(default=0.5, ge=0, le=1)
    ner_labels: list[str] = Field(
        default_factory=lambda: [
            "person name",
            "organization name",
            "geographical location",
        ]
    )

    ner_tokenizer: str = Field(default="microsoft/deberta-v3-large", min_length=1)
    ner_window_tokens: int = Field(default=192, gt=0)
    ner_overlap_tokens: int = Field(default=32, ge=0)

    # Snapshot only processing parameters, never infrastructure credentials.
    def processing_config(self) -> ProcessingConfig:
        return ProcessingConfig(
            max_words=self.chunk_max_words,
            overlap_words=self.chunk_overlap_words,
            max_characters=self.chunk_max_characters,
            ner_model=self.ner_model,
            ner_labels=self.ner_labels,
            ner_threshold=self.ner_threshold,
            ner_tokenizer=self.ner_tokenizer,
            ner_window_tokens=self.ner_window_tokens,
            ner_overlap_tokens=self.ner_overlap_tokens,
        )

    # Validate related limits when configuration is loaded by any process.
    @model_validator(mode="after")
    def validate_limits(self) -> Self:
        self.processing_config()
        if self.generation_max_tokens >= self.generation_context_tokens:
            raise ValueError(
                "generation_max_tokens must be less than generation_context_tokens"
            )
        return self


# Load and validate configuration once within each application process.
@lru_cache
def get_settings() -> Settings:
    return Settings()
