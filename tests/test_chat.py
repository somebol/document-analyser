import json
from unittest.mock import patch

from document_analyser.generation.chat import handle_chunks
from document_analyser.generation.summarise import summarise_chunks


class Generator:
    # Capture every original evidence item visited by the summarisation workflow.
    def __init__(self):
        self.visited = set()

    # Return valid, source-linked findings without depending on Ollama.
    def generate_json(self, system_prompt, user_prompt, schema):
        payload = json.loads(user_prompt)
        if "evidence" not in payload:
            return '{"action": "document_summary"}'
        for item in payload["evidence"]:
            self.visited.update(item["evidence_ids"])
        return json.dumps(
            {
                "findings": [
                    {
                        "statement": item["text"][:100],
                        "evidence_ids": item["evidence_ids"],
                    }
                    for item in payload["evidence"]
                ],
                "limitations": "",
            }
        )


# Whole-document chat must read every chunk, including material beyond search top-k.
def test_summary_covers_all_chunks():
    chunks = [
        {
            "chunk_id": str(i),
            "document_version_id": "v1",
            "source_file": "synthetic",
            "item_ref": str(i),
            "page_numbers": [i],
            "start": 0,
            "end": 1500,
            "text": (f"Evidence {i} " * 150)[:1500],
        }
        for i in range(80)
    ]
    generator = Generator()
    with patch(
        "document_analyser.generation.chat.QdrantClient",
        side_effect=AssertionError("Summary used search"),
    ):
        reply = handle_chunks("Summarise this document", chunks, None, generator)
    assert generator.visited == set(range(1, 81))
    assert len(reply["sources"]) == 80
    assert reply["answer"]["findings"]


# Noncompressing outputs must preserve section findings instead of failing on length.
def test_large_summary_fallback():
    chunks = [
        {
            "chunk_id": str(i),
            "document_version_id": "v1",
            "page_numbers": [],
            "text": "Synthetic evidence. " * 80,
        }
        for i in range(250)
    ]
    result = summarise_chunks(chunks, Generator())
    assert len(result["sources"]) == 250
    assert result["summary"]["findings"]


# Stale vector payload text must never override authoritative source records.
def test_question_uses_database_evidence(monkeypatch) -> None:
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.test:6333")
    monkeypatch.setenv("QDRANT_TIMEOUT", "19")
    from document_analyser.generation.answer import GroundedAnswer
    from document_analyser.generation.chat import ChatRoute
    from document_analyser.retrieval.search import SearchResult

    chunk = {
        "chunk_id": "c1",
        "document_version_id": "v1",
        "source_file": "original.pdf",
        "item_ref": "#/texts/0",
        "page_numbers": [2],
        "start": 0,
        "end": 13,
        "text": "Database text",
    }
    stale = SearchResult(
        **{**chunk, "text": "Untrusted projection", "page_numbers": (99,)}, score=0.9
    )
    with (
        patch(
            "document_analyser.generation.chat.choose_route",
            return_value=ChatRoute(action="question"),
        ),
        patch("document_analyser.generation.chat.QdrantClient") as client,
        patch(
            "document_analyser.generation.chat.search_passages", return_value=[stale]
        ),
        patch(
            "document_analyser.generation.chat.answer_question",
            return_value=GroundedAnswer(findings=[], limitations=""),
        ) as answer,
    ):
        reply = handle_chunks("What does it say?", [chunk], None, None)
    client.assert_called_once_with(url="http://qdrant.test:6333", timeout=19)
    assert answer.call_args.args[1][0].text == "Database text"
    assert reply["sources"]["1"]["page_numbers"] == [2]
