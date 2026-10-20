import argparse
import json
import re
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

from document_analyser.config import get_settings
from document_analyser.ingestion.passages import load_passages


@dataclass(frozen=True)
class RetrievalChunk:
    chunk_id: str
    document_version_id: str
    source_file: str
    item_ref: str
    page_numbers: tuple[int, ...]
    start: int
    end: int
    text: str


# Yield source offsets shared by worker and CLI, with bounded chunk lengths.
def chunk_spans(
    text: str,
    max_words: int,
    overlap_words: int,
    max_characters: int,
) -> Iterator[tuple[int, int]]:
    if max_words <= 0 or max_characters <= 0:
        raise ValueError("Chunk limits must be positive")
    if not 0 <= overlap_words < max_words:
        raise ValueError("Overlap must be nonnegative and smaller than max_words")
    words = list(re.finditer(r"\S+", text))
    for first in range(0, len(words), max_words - overlap_words):
        last = min(first + max_words, len(words))
        start, end = words[first].start(), words[last - 1].end()
        for piece_start in range(start, end, max_characters):
            piece_end = min(piece_start + max_characters, end)
            if text[piece_start:piece_end].strip():
                yield piece_start, piece_end
        if last == len(words):
            break


# Create source-linked chunks without changing the original passage text.
def create_document_chunks(
    source: Path,
    max_words: int | None = None,
    overlap_words: int | None = None,
    max_characters: int | None = None,
) -> list[RetrievalChunk]:
    settings = get_settings()
    max_words = settings.chunk_max_words if max_words is None else max_words
    overlap_words = (
        settings.chunk_overlap_words if overlap_words is None else overlap_words
    )
    max_characters = (
        settings.chunk_max_characters if max_characters is None else max_characters
    )

    source = source.expanduser().resolve()
    document_version_id = sha256(source.read_bytes()).hexdigest()
    passages = load_passages(source)
    chunks: list[RetrievalChunk] = []

    for passage in passages:
        for start, end in chunk_spans(
            passage.text, max_words, overlap_words, max_characters
        ):
            text = passage.text[start:end]

            identity = (
                f"{document_version_id}|{passage.item_ref}|"
                f"{start}|{end}|words-v1|{max_words}|{overlap_words}"
            )
            chunk_id = sha256(identity.encode("utf-8")).hexdigest()

            chunks.append(
                RetrievalChunk(
                    chunk_id=chunk_id,
                    document_version_id=document_version_id,
                    source_file=passage.source_file,
                    item_ref=passage.item_ref,
                    page_numbers=passage.page_numbers,
                    start=start,
                    end=end,
                    text=text,
                )
            )

    return chunks


# Export retrieval chunks from an existing Docling JSON document.
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create source-linked retrieval chunks.",
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Path to the original Docling JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the chunk JSON.",
    )
    args = parser.parse_args()

    chunks = create_document_chunks(args.source)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            [asdict(chunk) for chunk in chunks],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Saved {len(chunks)} retrieval chunks to {args.output}")


if __name__ == "__main__":
    main()
