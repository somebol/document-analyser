"""Read source passages and their provenance from Docling JSON."""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from docling_core.types.doc import DoclingDocument, TextItem


@dataclass(frozen=True)
class SourcePassage:
    """A document text item with its source references."""

    source_file: str
    item_ref: str
    page_numbers: tuple[int, ...]
    text: str


def load_passages(source: Path) -> list[SourcePassage]:
    """Load text items from a previously parsed Docling document."""
    source = source.expanduser().resolve()

    if not source.is_file():
        raise FileNotFoundError(f"Parsed document not found: {source}")

    document = DoclingDocument.load_from_json(source)
    passages: list[SourcePassage] = []

    for item, _level in document.iterate_items():
        if not isinstance(item, TextItem):
            continue

        if not item.text.strip():
            continue

        pages = tuple(sorted({
            provenance.page_no
            for provenance in item.prov
        }))

        passages.append(
            SourcePassage(
                source_file=source.name,
                item_ref=item.self_ref,
                page_numbers=pages,
                text=item.text,
            )
        )

    return passages


def main() -> None:
    """Export source passages as JSON."""
    parser = argparse.ArgumentParser(
        description="Extract source passages from Docling JSON.",
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Path to the Docling JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the passage JSON output.",
    )
    args = parser.parse_args()

    passages = load_passages(args.source)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            [asdict(passage) for passage in passages],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Saved {len(passages)} passages to {args.output}")


if __name__ == "__main__":
    main()