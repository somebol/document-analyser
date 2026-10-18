import argparse
import json
from pathlib import Path

from document_analyser.generation.answer import (
    SYSTEM_PROMPT,
    GroundedAnswer,
)
from document_analyser.generation.ollama_provider import (
    OllamaGenerationProvider,
)
from document_analyser.generation.providers import GenerationProvider

BATCH_CHARACTERS = 10000


# Group evidence into bounded batches without dropping any items.
def create_batches(
    evidence: list[dict],
    max_characters: int = BATCH_CHARACTERS,
) -> list[list[dict]]:
    batches: list[list[dict]] = []
    current: list[dict] = []
    current_size = 0

    for item in evidence:
        size = len(json.dumps(item, ensure_ascii=False))

        if size > max_characters:
            raise ValueError(
                "An evidence item exceeds the summary batch budget."
            )

        if current and current_size + size > max_characters:
            batches.append(current)
            current = []
            current_size = 0

        current.append(item)
        current_size += size

    if current:
        batches.append(current)

    return batches


# Request a summary and reject references outside the supplied evidence.
def summarise_evidence(
    evidence: list[dict],
    provider: GenerationProvider,
    instruction: str,
) -> GroundedAnswer:
    schema = GroundedAnswer.model_json_schema()

    prompt = json.dumps(
        {
            "question": instruction,
            "evidence": evidence,
            "output_schema": schema,
        },
        ensure_ascii=False,
    )

    raw = provider.generate_json(
        SYSTEM_PROMPT,
        prompt,
        schema,
    )
    answer = GroundedAnswer.model_validate_json(raw)

    allowed_ids = {
        evidence_id
        for item in evidence
        for evidence_id in item["evidence_ids"]
    }

    for finding in answer.findings:
        if not set(finding.evidence_ids) <= allowed_ids:
            raise ValueError("Summary contains an unknown citation ID.")

    return answer


# Summarise all chunks from one document version and retain source metadata.
def summarise_document(
    source: Path,
    provider: GenerationProvider,
) -> dict:
    chunks = json.loads(source.read_text(encoding="utf-8"))

    if not isinstance(chunks, list) or not chunks:
        raise ValueError("Expected a nonempty chunk JSON list.")

    versions = {chunk["document_version_id"] for chunk in chunks}

    if len(versions) != 1:
        raise ValueError("Select chunks from exactly one document version.")

    evidence = [
        {
            "evidence_ids": [number],
            "text": chunk["text"],
            "pages": chunk["page_numbers"],
        }
        for number, chunk in enumerate(chunks, start=1)
    ]

    batches = create_batches(evidence)
    summaries: list[GroundedAnswer] = []

    for number, batch in enumerate(batches, start=1):
        print(f"Summarising batch {number}/{len(batches)}")

        summaries.append(
            summarise_evidence(
                batch,
                provider,
                (
                    "Summarise this document section. Capture its purpose, "
                    "named parties, important dates, amounts, obligations "
                    "and qualifications where present. Cite each finding. "
                    "Do not invent missing details."
                ),
            )
        )

    if len(summaries) == 1:
        final = summaries[0]
    else:
        intermediate = [
            {
                "evidence_ids": finding.evidence_ids,
                "text": finding.statement,
            }
            for summary in summaries
            for finding in summary.findings
        ]

        if not intermediate:
            final = GroundedAnswer(
                findings=[],
                limitations="No supported findings were generated.",
            )
        elif len(create_batches(intermediate)) > 1:
            raise ValueError(
                "Combined summaries exceed the consolidation budget. "
                "Hierarchical consolidation is required for this document."
            )
        else:
            final = summarise_evidence(
                intermediate,
                provider,
                (
                    "Combine these section findings into a concise document "
                    "overview. Remove repetition and preserve important facts, "
                    "qualifications and original evidence IDs. These are "
                    "intermediate summaries, not original source passages."
                ),
            )

    limitations = list(dict.fromkeys(
        text
        for text in [
            *(summary.limitations for summary in summaries),
            final.limitations,
        ]
        if text.strip()
    ))

    final = final.model_copy(
        update={"limitations": "\n".join(limitations)}
    )

    return {
        "document_version_id": next(iter(versions)),
        "summary": final.model_dump(),
        "sources": {
            str(number): chunk
            for number, chunk in enumerate(chunks, start=1)
        },
        "section_summaries": [
            summary.model_dump() for summary in summaries
        ],
    }


# Generate and save a source-linked document summary.
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarise every chunk in a document.",
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = summarise_document(
        args.source,
        OllamaGenerationProvider(),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    answer = GroundedAnswer.model_validate(result["summary"])

    for finding in answer.findings:
        citations = " ".join(
            f"[{number}]" for number in finding.evidence_ids
        )
        print(f"- {finding.statement} {citations}")

    if answer.limitations:
        print(f"\nLimitations: {answer.limitations}")

    print(f"\nSaved summary and sources to {args.output}")


if __name__ == "__main__":
    main()