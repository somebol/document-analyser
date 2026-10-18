import argparse
import json
import re
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

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


# Create source-linked chunks without changing the original passage text.
def create_document_chunks(
    source: Path,
    max_words: int = 120,
    overlap_words: int = 20,
) -> list[RetrievalChunk]:
    if max_words <= 0:
        raise ValueError("max_words must be greater than zero.")

    if not 0 <= overlap_words < max_words:
        raise ValueError(
            "overlap_words must be nonnegative and less than max_words."
        )

    source = source.expanduser().resolve()
    document_version_id = sha256(source.read_bytes()).hexdigest()
    passages = load_passages(source)
    chunks: list[RetrievalChunk] = []

    for passage in passages:
        words = list(re.finditer(r"\S+", passage.text))
        first_word = 0

        while first_word < len(words):
            last_word = min(first_word + max_words, len(words))

            start = words[first_word].start()
            end = words[last_word - 1].end()
            text = passage.text[start:end]

            identity = (
                f"{document_version_id}|{passage.item_ref}|"
                f"{start}|{end}|words-v1|{max_words}|{overlap_words}"
            )
            chunk_id = sha256(
                identity.encode("utf-8")
            ).hexdigest()

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

            if last_word == len(words):
                break

            first_word = last_word - overlap_words

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