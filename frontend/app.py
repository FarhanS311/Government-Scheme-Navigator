from __future__ import annotations

import json
import os
from collections.abc import Generator, Iterator
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


def iter_ask_stream_events(
    client: httpx.Client,
    question: str,
    history: list[dict[str, str]],
    k: int = 4,
) -> Iterator[tuple[str, Any]]:
    with client.stream(
        "POST",
        f"{backend_url()}/api/ask/stream",
        json={"question": question, "k": k, "history": history},
        timeout=120.0,
    ) as response:
        response.raise_for_status()
        buffer = ""
        for chunk in response.iter_text():
            buffer += chunk
            while "\n\n" in buffer:
                block, buffer = buffer.split("\n\n", 1)
                event_type = "message"
                data_line = ""
                for line in block.split("\n"):
                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                    elif line.startswith("data:"):
                        data_line = line[5:].strip()
                if not data_line:
                    continue
                payload = json.loads(data_line)
                yield event_type, payload


def stream_answer_tokens(
    client: httpx.Client,
    question: str,
    history: list[dict[str, str]],
    k: int = 4,
) -> tuple[Generator[str, None, None], dict[str, Any]]:
    stream_meta: dict[str, Any] = {}

    def token_generator() -> Generator[str, None, None]:
        for event_type, payload in iter_ask_stream_events(client, question, history, k=k):
            if event_type == "meta":
                stream_meta.update(payload)
            elif event_type == "token":
                text = payload.get("text", "")
                if text:
                    yield text
            elif event_type == "sources":
                stream_meta["sources"] = payload

    return token_generator(), stream_meta


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
            history = [
                {"role": message["role"], "content": message["content"]}
                for message in st.session_state.messages[:-1]
            ]
            token_gen, stream_meta = stream_answer_tokens(client, question, history)

            with st.chat_message("assistant"):
                try:
                    answer = st.write_stream(token_gen)
                except httpx.HTTPStatusError as exc:
                    st.error(f"Question failed: {exc.response.text}")
                    st.stop()
                except httpx.HTTPError as exc:
                    st.error(f"Question failed: {exc}")
                    st.stop()

                sources = stream_meta.get("sources", [])
                if sources:
                    with st.expander(f"Sources ({len(sources)})", expanded=False):
                        for source in sources:
                            label = f"Page {source['page']} · Chunk {source['id']}"
                            with st.expander(label, expanded=False):
                                st.markdown(source.get("text") or source.get("snippet", ""))
                if stream_meta.get("query_type"):
                    st.caption(f"Query type: {stream_meta['query_type']}")

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": stream_meta.get("sources", []),
                    "query_type": stream_meta.get("query_type"),
                }
            )
            st.rerun()


if __name__ == "__main__":
    main()