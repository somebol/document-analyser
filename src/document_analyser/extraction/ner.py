"""Extract named entity mentions using a local GLiNER model."""

import json
import logging
from dataclasses import asdict, dataclass

from gliner import GLiNER

from document_analyser.config import get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EntityMention:
    """An entity mention with character offsets into the input text."""

    text: str
    label: str
    start: int
    end: int
    score: float


class EntityExtractor:
    """Load a GLiNER model once and reuse it across passages."""

    # Load the configured model once and retain the extraction defaults.
    def __init__(
        self,
        model_name: str | None = None,
        labels: tuple[str, ...] | None = None,
        threshold: float | None = None,
    ) -> None:
        settings = get_settings()
        model_name = settings.ner_model if model_name is None else model_name
        self.labels = tuple(settings.ner_labels) if labels is None else labels
        self.threshold = settings.ner_threshold if threshold is None else threshold
        logger.info("Loading NER model: %s", model_name)
        self.model = GLiNER.from_pretrained(model_name)
        self.model.eval()

    # Preserve explicit labels and threshold overrides, including a zero threshold.
    def extract(
        self,
        text: str,
        labels: tuple[str, ...] | None = None,
        threshold: float | None = None,
    ) -> list[EntityMention]:
        """Extract mentions from a short passage."""
        if not text.strip():
            return []

        predictions = self.model.predict_entities(
            text,
            list(self.labels if labels is None else labels),
            threshold=self.threshold if threshold is None else threshold,
        )

        return [
            EntityMention(
                text=prediction["text"],
                label=prediction["label"],
                start=int(prediction["start"]),
                end=int(prediction["end"]),
                score=float(prediction["score"]),
            )
            for prediction in predictions
        ]


def main() -> None:
    """Run a small local NER demonstration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    text = (
        "Elena Silva was appointed director of Meridian Holdings. "
        "The company has an office in Sydney."
    )

    extractor = EntityExtractor()
    mentions = extractor.extract(text)

    print(
        json.dumps(
            [asdict(mention) for mention in mentions],
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
