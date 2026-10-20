import re
from dataclasses import dataclass

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from document_analyser.config import get_settings


@dataclass(frozen=True)
class TextWindow:
    text: str
    start: int
    end: int


# Load the tokenizer used by our selected GLiNER encoder.
def load_tokenizer(model_name: str | None = None) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name if model_name is not None else get_settings().ner_tokenizer,
        use_fast=True,
    )

    if not tokenizer.is_fast:
        raise ValueError("Windowing requires a fast tokenizer.")

    return tokenizer


# Split a passage into overlapping windows, preserving source offsets.
# Split on whitespace boundaries while enforcing the tokenizer budget.
def create_windows(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> list[TextWindow]:
    settings = get_settings()
    max_tokens = settings.ner_window_tokens if max_tokens is None else max_tokens
    overlap_tokens = (
        settings.ner_overlap_tokens if overlap_tokens is None else overlap_tokens
    )
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero.")

    if not 0 <= overlap_tokens < max_tokens:
        raise ValueError("overlap_tokens must be nonnegative and less than max_tokens.")

    words = list(re.finditer(r"\S+", text))

    if not words:
        return []

    windows: list[TextWindow] = []
    start_index = 0

    while start_index < len(words):
        start = words[start_index].start()
        end_index = start_index

        # Extend the window by whole words until the budget is reached.
        while end_index < len(words):
            candidate_end = words[end_index].end()
            candidate = text[start:candidate_end]

            token_count = len(
                tokenizer.encode(
                    candidate,
                    add_special_tokens=False,
                    truncation=False,
                )
            )

            if token_count > max_tokens:
                break

            end_index += 1

        if end_index == start_index:
            raise ValueError(
                "A single whitespace-delimited word exceeds the "
                f"{max_tokens}-token budget at character {start}."
            )

        end = words[end_index - 1].end()

        windows.append(
            TextWindow(
                text=text[start:end],
                start=start,
                end=end,
            )
        )

        if end_index == len(words):
            break

        # Reuse a whole-word suffix within the overlap budget.
        next_start = end_index

        for candidate_index in range(
            end_index - 1,
            start_index,
            -1,
        ):
            overlap_text = text[words[candidate_index].start() : end]

            token_count = len(
                tokenizer.encode(
                    overlap_text,
                    add_special_tokens=False,
                    truncation=False,
                )
            )

            if token_count > overlap_tokens:
                break

            next_start = candidate_index

        start_index = next_start

    return windows


# Demonstrate overlapping windows without loading the NER model.
def main() -> None:
    text = (
        "Elena Silva was appointed director of Meridian Holdings. "
        "The company operates from Sydney. "
        "Harbour Nominees holds shares in Meridian Holdings. "
        "Elena must approve the exercise of voting rights."
    )

    tokenizer = load_tokenizer()

    windows = create_windows(
        text,
        tokenizer,
        max_tokens=24,
        overlap_tokens=8,
    )

    for number, window in enumerate(windows, start=1):
        print(f"\nWindow {number}: characters {window.start}:{window.end}")
        print(window.text)

        assert text[window.start : window.end] == window.text


if __name__ == "__main__":
    main()
