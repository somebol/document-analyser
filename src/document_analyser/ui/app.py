import streamlit as st
from qdrant_client import QdrantClient

from document_analyser.generation.answer import answer_question
from document_analyser.generation.ollama_provider import (
    OllamaGenerationProvider,
)
from document_analyser.retrieval.providers.ollama_embeddings import (
    OllamaEmbeddingProvider,
)
from document_analyser.retrieval.search import search_passages


# Reuse model clients across Streamlit reruns.
@st.cache_resource
def get_providers() -> tuple[
    OllamaEmbeddingProvider,
    OllamaGenerationProvider,
]:
    return (
        OllamaEmbeddingProvider(),
        OllamaGenerationProvider(),
    )


# Display the generated answer and its retrieved evidence.
def display_result(result: dict) -> None:
    answer = result["answer"]
    evidence = result["evidence"]

    st.subheader("Answer")

    if not answer.findings:
        st.info("The retrieved evidence does not establish an answer.")

    for finding in answer.findings:
        citations = " ".join(
            f"[{number}]" for number in finding.evidence_ids
        )
        st.write(f"{finding.statement} {citations}")

    if answer.limitations:
        st.warning(answer.limitations)

    st.caption(
        "Citation IDs are validated. "
        "Check the passages to confirm they support each statement."
    )

    st.subheader("Retrieved evidence")

    for number, passage in enumerate(evidence, start=1):
        pages = ", ".join(map(str, passage.page_numbers))

        with st.expander(
            f"[{number}] {passage.source_file} — "
            f"pages {pages or 'unknown'}"
        ):
            st.text(passage.text)
            st.caption(
                f"Similarity: {passage.score:.4f} | "
                f"Item: {passage.item_ref} | "
                f"Characters: {passage.start}:{passage.end}"
            )
            st.code(passage.chunk_id, language=None)


# Collect a question and run the existing RAG pipeline.
def main() -> None:
    st.set_page_config(
        page_title="Document Analyser",
        layout="wide",
    )

    st.title("Document Analyser")
    st.caption("Ask questions and inspect the supporting passages.")

    with st.form("question_form"):
        question = st.text_area(
            "Question",
            placeholder="How long should I retain receipts?",
        )

        limit = st.slider(
            "Number of passages to retrieve",
            min_value=1,
            max_value=8,
            value=5,
        )

        document_version = st.text_input(
            "Document version ID (optional)",
            help="Leave blank to search all indexed document versions.",
        )

        submitted = st.form_submit_button("Ask")

    if submitted:
        st.session_state.pop("rag_result", None)

        if not question.strip():
            st.error("Enter a question.")
        else:
            try:
                with st.spinner("Retrieving evidence and generating an answer…"):
                    embeddings, generator = get_providers()

                    client = QdrantClient(
                        url="http://localhost:6333",
                        timeout=30,
                    )

                    try:
                        evidence = search_passages(
                            question=question,
                            provider=embeddings,
                            client=client,
                            limit=limit,
                            document_version_id=(
                                document_version.strip() or None
                            ),
                        )
                    finally:
                        client.close()

                    answer = answer_question(
                        question,
                        evidence,
                        generator,
                    )

                    st.session_state["rag_result"] = {
                        "question": question,
                        "answer": answer,
                        "evidence": evidence,
                    }

            except Exception as error:
                st.error(f"Could not complete the request: {error}")

    if "rag_result" in st.session_state:
        result = st.session_state["rag_result"]
        st.write(f"**Question:** {result['question']}")
        display_result(result)


if __name__ == "__main__":
    main()