from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"


def backend_url() -> str:
    return os.getenv("BACKEND_URL", DEFAULT_BACKEND_URL).rstrip("/")


def fetch_status(client: httpx.Client) -> dict[str, Any]:
    response = client.get(f"{backend_url()}/api/status")
    response.raise_for_status()
    return response.json()


def upload_pdf(client: httpx.Client, file_name: str, file_bytes: bytes) -> dict[str, Any]:
    response = client.post(
        f"{backend_url()}/api/upload",
        files={"file": (file_name, file_bytes, "application/pdf")},
        timeout=600.0,
    )
    response.raise_for_status()
    return response.json()


def ask_question(client: httpx.Client, question: str, k: int = 4) -> dict[str, Any]:
    response = client.post(
        f"{backend_url()}/api/ask",
        json={"question": question, "k": k},
        timeout=120.0,
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    st.set_page_config(page_title="Government Scheme Navigator", layout="wide")
    st.title("Government Scheme Navigator")
    st.caption("Upload one scheme PDF, ask questions, get cited answers from that document.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "document_ready" not in st.session_state:
        st.session_state.document_ready = False
    if "document_name" not in st.session_state:
        st.session_state.document_name = None

    with httpx.Client() as client:
        try:
            status = fetch_status(client)
            st.session_state.document_ready = status.get("ready", False)
            if status.get("filename"):
                st.session_state.document_name = status["filename"]
        except httpx.HTTPError as exc:
            st.error(f"Cannot reach backend at {backend_url()}. Start uvicorn first. ({exc})")
            st.stop()

        uploaded = st.file_uploader("Upload scheme PDF", type=["pdf"])

        if uploaded is not None:
            upload_key = f"{uploaded.name}:{uploaded.size}"
            if st.session_state.get("last_upload_key") != upload_key:
                with st.spinner("Indexing PDF (first run may download embedding models)..."):
                    try:
                        result = upload_pdf(client, uploaded.name, uploaded.getvalue())
                    except httpx.HTTPStatusError as exc:
                        detail = exc.response.text
                        st.error(f"Upload failed: {detail}")
                        st.stop()
                    except httpx.HTTPError as exc:
                        st.error(f"Upload failed: {exc}")
                        st.stop()

                st.session_state.last_upload_key = upload_key
                st.session_state.document_ready = True
                st.session_state.document_name = result["filename"]
                st.session_state.messages = []
                st.success(result["message"])

        if st.session_state.document_ready and st.session_state.document_name:
            st.info(f"Indexed document: **{st.session_state.document_name}**")
        else:
            st.warning("Upload a PDF to enable questions.")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and message.get("sources"):
                    with st.expander(f"Sources ({len(message['sources'])})", expanded=False):
                        for source in message["sources"]:
                            label = f"Page {source['page']} · Chunk {source['id']}"
                            with st.expander(label, expanded=False):
                                st.markdown(source.get("text") or source.get("snippet", ""))
                if message["role"] == "assistant" and message.get("query_type"):
                    st.caption(f"Query type: {message['query_type']}")

        question_disabled = not st.session_state.document_ready
        question = st.chat_input(
            "Ask about eligibility, documents, or application process...",
            disabled=question_disabled,
        )

        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.spinner("Searching document and generating answer..."):
                try:
                    result = ask_question(client, question)
                except httpx.HTTPStatusError as exc:
                    st.error(f"Question failed: {exc.response.text}")
                    st.stop()
                except httpx.HTTPError as exc:
                    st.error(f"Question failed: {exc}")
                    st.stop()

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result.get("sources", []),
                    "query_type": result.get("query_type"),
                }
            )
            st.rerun()


if __name__ == "__main__":
    main()
