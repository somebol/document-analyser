"""Extract named entity mentions using a local GLiNER model."""

import json
import logging
from dataclasses import asdict, dataclass

from gliner import GLiNER

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "urchade/gliner_multi-v2.1"
DEFAULT_LABELS = ("person", "organization", "location")


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

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        logger.info("Loading NER model: %s", model_name)
        self.model = GLiNER.from_pretrained(model_name)
        self.model.eval()

    def extract(
        self,
        text: str,
        labels: tuple[str, ...] = DEFAULT_LABELS,
        threshold: float = 0.5,
    ) -> list[EntityMention]:
        """Extract mentions from a short passage."""
        if not text.strip():
            return []

        predictions = self.model.predict_entities(
            text,
            list(labels),
            threshold=threshold,
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

    print(json.dumps(
        [asdict(mention) for mention in mentions],
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()