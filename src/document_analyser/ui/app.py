import httpx
import streamlit as st

from document_analyser.config import get_settings

settings = get_settings()
API_URL = settings.document_api_url.rstrip("/")


# Display structured findings and their cited source passages.
def display_reply(reply: dict) -> None:
    answer = reply["answer"]
    cited_ids: set[int] = set()

    if not answer["findings"]:
        st.info("The available evidence does not establish an answer.")

    for finding in answer["findings"]:
        cited_ids.update(finding["evidence_ids"])

        citations = " ".join(f"[{number}]" for number in finding["evidence_ids"])
        st.write(f"{finding['statement']} {citations}")

    if answer["limitations"]:
        st.caption(answer["limitations"])

    for number in sorted(cited_ids):
        source = reply["sources"][str(number)]
        pages = ", ".join(map(str, source["page_numbers"]))

        with st.expander(
            f"[{number}] {source['source_file']} — pages {pages or 'unknown'}"
        ):
            st.text(source["text"])
            st.caption(f"Source item: {source['item_ref']}")
            st.code(source["chunk_id"], language=None)


# Poll independently so long ingestion jobs update without disrupting chat input.
@st.fragment(run_every=settings.ui_poll_seconds)
def show_processing() -> None:
    try:
        response = httpx.get(
            f"{API_URL}/documents", timeout=settings.api_request_timeout
        )
        response.raise_for_status()
        documents = response.json()
    except httpx.HTTPError:
        st.warning("Processing status unavailable. Check FastAPI and PostgreSQL.")
        return
    with st.expander("Document processing", expanded=True):
        for document in documents:
            st.write(f"{document['name']}: {document['stage']} ({document['status']})")
            st.progress(document["progress"] / 100)
            if document["warning"]:
                st.warning(document["warning"])
            if document["error"]:
                st.error(document["error"])
            if document["status"] == "failed" and st.button(
                "Retry", key=document["job_id"]
            ):
                try:
                    retry = httpx.post(
                        f"{API_URL}/documents/{document['document_id']}/retry",
                        timeout=settings.api_request_timeout,
                    )
                    retry.raise_for_status()
                    st.rerun()
                except httpx.HTTPError:
                    st.error("Retry failed. Check the API logs.")
        if documents and st.button("Refresh chat document list"):
            st.rerun()


# Collect chat input and communicate exclusively through the API.
def main() -> None:
    st.set_page_config(
        page_title="Document Analyser",
        layout="wide",
    )
    st.title("Document Analyser")

    uploaded = st.sidebar.file_uploader("Upload a document")
    if uploaded is not None and st.sidebar.button("Upload and process"):
        try:
            response = httpx.post(
                f"{API_URL}/documents",
                files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                timeout=settings.upload_request_timeout,
            )
            response.raise_for_status()
            st.sidebar.success("Upload preserved. Processing queued.")
        except httpx.HTTPStatusError as error:
            st.sidebar.error(error.response.json().get("detail", "Upload failed."))
        except httpx.RequestError:
            st.sidebar.error("Upload could not reach the API.")

    show_processing()

    try:
        response = httpx.get(
            f"{API_URL}/documents",
            timeout=settings.api_request_timeout,
        )
        response.raise_for_status()
        documents = response.json()

    except httpx.HTTPError:
        st.error("Cannot load documents. Check that FastAPI is running.")
        st.stop()

    documents = [
        document for document in documents if document["index_status"] == "indexed"
    ]

    if not documents:
        st.info("Upload a document to begin. Chat becomes available after indexing.")
        st.stop()

    names = {document["document_id"]: document["name"] for document in documents}

    selected_id = st.sidebar.selectbox(
        "Document",
        options=list(names),
        format_func=lambda document_id: names[document_id],
    )

    histories = st.session_state.setdefault("document_chats", {})
    messages = histories.setdefault(selected_id, [])

    for message in messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message["content"])
            elif "error" in message:
                st.error(message["error"])
            else:
                display_reply(message["reply"])

    prompt = st.chat_input("Ask a question or type 'summarise this document'")

    if prompt:
        messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Reviewing the document…"):
                    response = httpx.post(
                        f"{API_URL}/chat",
                        json={
                            "document_id": selected_id,
                            "message": prompt,
                        },
                        timeout=settings.chat_request_timeout,
                    )
                    response.raise_for_status()
                    reply = response.json()

                display_reply(reply)
                messages.append(
                    {
                        "role": "assistant",
                        "reply": reply,
                    }
                )

            except httpx.HTTPStatusError as error:
                message = (
                    "The API rejected or could not complete the request "
                    f"(HTTP {error.response.status_code})."
                )
                st.error(message)
                messages.append(
                    {
                        "role": "assistant",
                        "error": message,
                    }
                )

            except httpx.RequestError:
                message = "The API could not be reached or the request timed out."
                st.error(message)
                messages.append(
                    {
                        "role": "assistant",
                        "error": message,
                    }
                )


if __name__ == "__main__":
    main()
