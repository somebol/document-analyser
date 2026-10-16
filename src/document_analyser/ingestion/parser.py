"""Convert source documents into structured data using Docling."""

import logging
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling_core.types.doc import DoclingDocument

logger = logging.getLogger(__name__)


def parse_document(source: Path) -> DoclingDocument:
    """Parse a local document, preserving its structure and provenance."""
    source = source.expanduser().resolve()

    if not source.is_file():
        raise FileNotFoundError(f"Document not found: {source}")

    logger.info("Parsing %s", source.name)

    converter = DocumentConverter()
    result = converter.convert(source)

    logger.info("Completed parsing %s", source.name)
    return result.document