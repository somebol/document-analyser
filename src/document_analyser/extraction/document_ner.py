import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from document_analyser.extraction.ner import EntityExtractor
from document_analyser.extraction.windowing import (
    create_windows,
    load_tokenizer,
)
from document_analyser.ingestion.passages import (
    SourcePassage,
    load_passages,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourcedMention:
    source_file: str
    item_ref: str
    page_numbers: tuple[int, ...]
    text: str
    label: str
    start: int
    end: int
    score: float


# Extract mentions and translate window offsets into passage offsets.
def extract_passage_mentions(
    passage: SourcePassage,
    extractor: EntityExtractor,
    tokenizer,
) -> list[SourcedMention]:
    windows = create_windows(passage.text, tokenizer)

    unique_mentions: dict[tuple[int, int, str], SourcedMention] = {}

    for window in windows:
        mentions = extractor.extract(window.text)

        for mention in mentions:
            if not 0 <= mention.start < mention.end <= len(window.text):
                raise ValueError(
                    f"Invalid mention offsets in {passage.item_ref}"
                )

            start = window.start + mention.start
            end = window.start + mention.end
            source_text = passage.text[start:end]

            if source_text != mention.text:
                raise ValueError(
                    f"Mention text does not match source at "
                    f"{passage.item_ref}:{start}:{end}"
                )

            sourced_mention = SourcedMention(
                source_file=passage.source_file,
                item_ref=passage.item_ref,
                page_numbers=passage.page_numbers,
                text=source_text,
                label=mention.label,
                start=start,
                end=end,
                score=mention.score,
            )

            key = (start, end, mention.label)
            existing = unique_mentions.get(key)

            if existing is None or mention.score > existing.score:
                unique_mentions[key] = sourced_mention

    return sorted(
        unique_mentions.values(),
        key=lambda mention: (mention.start, mention.end, mention.label),
    )


# Run NER on a previously parsed Docling document.
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract source-linked entities from Docling JSON.",
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Path to the original Docling JSON output.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the extracted mentions.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    passages = load_passages(args.source)
    extractor = EntityExtractor()
    tokenizer = load_tokenizer()

    all_mentions: list[SourcedMention] = []

    for number, passage in enumerate(passages, start=1):
        mentions = extract_passage_mentions(
            passage,
            extractor,
            tokenizer,
        )
        all_mentions.extend(mentions)

        logger.info(
            "Passage %d/%d: %s — %d mentions",
            number,
            len(passages),
            passage.item_ref,
            len(mentions),
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            [asdict(mention) for mention in all_mentions],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    logger.info(
        "Saved %d mentions to %s",
        len(all_mentions),
        args.output,
    )


if __name__ == "__main__":
    main()