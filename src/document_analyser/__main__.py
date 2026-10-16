"""Command-line entry point for Document Analyser."""

import argparse
import logging
from pathlib import Path

from document_analyser.ingestion.parser import parse_document

logger = logging.getLogger(__name__)


def main() -> int:
    """Parse a document and export Markdown and structured JSON."""
    argument_parser = argparse.ArgumentParser(
        description="Extract document text, tables and structure with Docling.",
    )
    argument_parser.add_argument(
        "source",
        type=Path,
        help="Path to the document to process.",
    )
    argument_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/output"),
        help="Output directory (default: data/output).",
    )
    args = argument_parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        document = parse_document(args.source)

        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        markdown_path = output_dir / f"{args.source.stem}.md"
        json_path = output_dir / f"{args.source.stem}.json"

        document.save_as_markdown(markdown_path)
        document.save_as_json(json_path)
    except Exception:
        logger.exception("Document processing failed")
        return 1

    logger.info("Markdown: %s", markdown_path)
    logger.info("Structured JSON: %s", json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())