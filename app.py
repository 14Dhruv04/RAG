import hashlib
import os
import shutil
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from parser import PDFparser, Excelparser, CSVparser
from retriever.bm25 import BM25Retriever
from retriever.semantic import SemanticRetriever
from tools.bm25_tool import create_bm25_tool
from tools.semantic_tool import create_semantic_tool
from agents.retriever_agent import RetrieverAgent
from agents.main_agent import MainAgent
from langchain_google_genai import ChatGoogleGenerativeAI
from guards.jailbreak_detection import document_sanitizer, input_check


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

st.set_page_config(
    page_title="Document RAG Chatbot",
    layout="wide",
)


# -----------------------------------------------------------------------------
# Configuration helpers
# -----------------------------------------------------------------------------

def get_data_dir() -> Path:
    configured_dir = os.getenv("DATA_DIR")
    if configured_dir:
        path = Path(configured_dir)
        return path if path.is_absolute() else BASE_DIR / path
    return BASE_DIR / "data"


def get_chroma_dir() -> Path:
    configured_dir = os.getenv("CHROMA_DIR")
    if configured_dir:
        path = Path(configured_dir)
        return path if path.is_absolute() else BASE_DIR / path
    return BASE_DIR / "chromadb"


def get_semantic_device() -> str:
    """Choose CUDA only when the installed PyTorch actually supports it."""
    requested = os.getenv("SEMANTIC_DEVICE", "auto").strip().lower()

    if requested == "cpu":
        return "cpu"

    if requested == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# -----------------------------------------------------------------------------
# Document loading
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_documents(data_dir: str):
    documents: list[dict[str, Any]] = []
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_path}")

    for file_path in sorted(data_path.iterdir()):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()

        if suffix == ".csv":
            parsed = CSVparser(str(file_path))
            parsed = document_sanitizer(parsed)
        elif suffix in {".xlsx", ".xls"}:
            parsed = Excelparser(str(file_path))
            parsed = document_sanitizer(parsed)
        elif suffix == ".pdf":
            parsed = PDFparser(str(file_path))
            parsed = document_sanitizer(parsed)
        else:
            continue

        for doc in parsed:
            enriched = dict(doc)
            enriched["source"] = file_path.name
            enriched["source_type"] = suffix
            documents.append(enriched)

    if not documents:
        raise ValueError(f"No supported documents found in {data_path}")

    return documents


@st.cache_data(show_spinner=False)
def build_semantic_documents(documents: list[dict[str, Any]], csv_rows_per_chunk: int = 75):
    """
    Keep the full corpus for BM25, but reduce the semantic corpus size by
    grouping CSV rows. This avoids generating one embedding per CSV row.
    """
    semantic_documents: list[dict[str, Any]] = []

    current_csv_source = None
    current_csv_rows: list[dict[str, Any]] = []

    def flush_csv_chunk() -> None:
        nonlocal current_csv_rows, current_csv_source
        if not current_csv_rows:
            return

        start = current_csv_rows[0].get("page_number", 0)
        end = current_csv_rows[-1].get("page_number", 0)
        content = "\n\n".join(row.get("content", "") for row in current_csv_rows)

        semantic_documents.append(
            {
                "page_number": f"{start}-{end}",
                "content": content,
                "source": current_csv_source,
                "source_type": ".csv",
            }
        )

        current_csv_rows = []
        current_csv_source = None

    for doc in documents:
        source_type = doc.get("source_type")

        if source_type == ".csv":
            if current_csv_source not in {None, doc["source"]}:
                flush_csv_chunk()

            current_csv_source = doc["source"]
            current_csv_rows.append(doc)

            if len(current_csv_rows) >= csv_rows_per_chunk:
                flush_csv_chunk()
        else:
            flush_csv_chunk()
            semantic_documents.append(doc)

    flush_csv_chunk()
    return semantic_documents


@st.cache_data(show_spinner=False)
def semantic_index_id(semantic_documents: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()

    for doc in semantic_documents:
        digest.update(str(doc.get("source", "")).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(doc.get("page_number", "")).encode("utf-8"))
        digest.update(b"\0")
        digest.update(doc.get("content", "").encode("utf-8"))
        digest.update(b"\0")

    return digest.hexdigest()[:16]

# -----------------------------------------------------------------------------
# RAG initialization
# -----------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def initialize_rag(
    data_dir: str,
    chroma_dir: str,
    model_name: str,
    csv_rows_per_chunk: int,
    semantic_device: str,
):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to your environment or .env file."
        )

    documents = load_documents(data_dir)
    semantic_documents = build_semantic_documents(
        documents,
        csv_rows_per_chunk=csv_rows_per_chunk,
    )

    bm25_retriever = BM25Retriever(documents)

    index_id = semantic_index_id(semantic_documents)
    collection_name = f"semantic_{index_id}"

    semantic_retriever = SemanticRetriever(
        semantic_documents,
        persist_directory=chroma_dir,
        collection_name=collection_name,
        device=semantic_device,
    )

    bm25_tool = create_bm25_tool(bm25_retriever)
    semantic_tool = create_semantic_tool(semantic_retriever)

    model = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
    )

    main_agent = MainAgent(
        model=model, 
        tools=[bm25_tool, semantic_tool]
    )

    return {
        "documents": documents,
        "semantic_documents": semantic_documents,
        "main_agent": main_agent,
        "semantic_collection": collection_name,
        "semantic_device": semantic_device,
    }


def render_message(message: dict[str, Any], message_index: int = 0) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        trace = message.get("trace", [])

        if message["role"] != "assistant" or not trace:
            return

        trace_id = message.get("trace_id", message_index)

        with st.expander("Agent execution trace", expanded=False):
            for index, event in enumerate(trace, start=1):
                event_type = event["type"]

                if event_type == "status":
                    st.caption(event["message"])

                elif event_type == "tool_call":
                    st.markdown(
                        f"**{index}. Tool call: `{event['name']}`**"
                    )

                    if event.get("args"):
                        st.json(event["args"])

                elif event_type == "tool_result":
                    tool_name = event["name"]
                    content = event.get("content", "")

                    st.markdown(
                        f"**{index}. Tool completed: `{tool_name}`**"
                    )

                    st.caption(
                        f"Retrieved content: "
                        f"{event.get('char_count', len(content)):,} "
                        "characters"
                    )

                    st.text_area(
                        label=f"Retrieved content from {tool_name}",
                        value=content or "No text content returned.",
                        height=250,
                        key=(
                            f"history_retrieved_content_"
                            f"{trace_id}_{index}"
                        ),
                        disabled=True,
                    )

# -----------------------------------------------------------------------------
# Main application
# -----------------------------------------------------------------------------

def main() -> None:
    st.title("Document RAG Chatbot")
    st.caption("Ask questions about the documents available in the data directory.")

    data_dir = get_data_dir()
    chroma_dir = get_chroma_dir()
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    csv_rows_per_chunk = int(os.getenv("SEMANTIC_CSV_ROWS_PER_CHUNK", "75"))
    semantic_device = get_semantic_device()

    try:
        rag = initialize_rag(
            data_dir=str(data_dir),
            chroma_dir=str(chroma_dir),
            model_name=model_name,
            csv_rows_per_chunk=csv_rows_per_chunk,
            semantic_device=semantic_device,
        )
    except Exception as exc:
        st.error(f"Failed to initialize the RAG pipeline: {exc}")
        st.info(
            "The app automatically uses CPU when CUDA is unavailable. "
            "Check GEMINI_API_KEY, the data directory, and the Chroma directory."
        )
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message_index, message in enumerate(st.session_state.messages):
        render_message(message, message_index)

    query = st.chat_input("Ask a question about your documents...")
    if not query:
        return

    if not input_check(query):
        invalid_msg = "Invalid Request"

        st.session_state.messages.append({
            "role": "user",
            "content": query,
        })

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            st.markdown(invalid_msg)

        st.session_state.messages.append({
            "role": "assistant",
            "content": invalid_msg,
        })

        return

    st.session_state.messages.append({
        "role": "user",
        "content": query,
    })

    with st.chat_message("user"):
        st.markdown(query)

        
    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        
        trace_events = []
        trace_id = len(st.session_state.messages)

        with st.expander("Agent execution trace", expanded=True):
            trace_container = st.container()

            def update_trace(event):
                """Render each event once as it arrives."""
                trace_events.append(event)

                index = len(trace_events)
                event_type = event["type"]

                with trace_container:
                    if event_type == "status":
                        st.caption(event["message"])

                    elif event_type == "tool_call":
                        st.markdown(
                            f"**{index}. Tool call: `{event['name']}`**"
                        )

                        args = event.get("args", {})
                        if args:
                            st.json(args)

                    elif event_type == "tool_result":
                        tool_name = event["name"]
                        retrieved_content = event.get("content", "")

                        st.markdown(
                            f"**{index}. Tool completed: `{tool_name}`**"
                        )

                        st.caption(
                            f"Retrieved content: "
                            f"{event.get('char_count', len(retrieved_content)):,} "
                            "characters"
                        )

                        st.text_area(
                            label=f"Retrieved content from {tool_name}",
                            value=retrieved_content or "No text content returned.",
                            height=250,
                            key=f"retrieved_content_{trace_id}_{index}",
                            disabled=True,
                        )

        try:
            answer = rag["main_agent"].response(
                query=query,
                on_event=update_trace,
            )

            answer_placeholder.markdown(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "trace": trace_events.copy(),
                "trace_id" : trace_id,
            })

        except Exception as exc:
            error_message = f"I could not process that query: {exc}"

            answer_placeholder.error(error_message)

            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message,
                "trace": trace_events.copy(),
                "trace_id" : trace_id,
            })

if __name__ == "__main__":
    main()