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


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

st.set_page_config(
    page_title="Document RAG Chatbot",
    page_icon="💬",
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
        # Never crash the application just because CUDA was requested but is
        # unavailable. Fall back to CPU.
        return "cpu"

    # auto
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
        elif suffix in {".xlsx", ".xls"}:
            parsed = Excelparser(str(file_path))
        elif suffix == ".pdf":
            parsed = PDFparser(str(file_path))
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
def build_semantic_documents(
    documents: list[dict[str, Any]], csv_rows_per_chunk: int = 75
):
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
# CPU/CUDA-safe semantic retriever
# -----------------------------------------------------------------------------

class SafeSemanticRetriever:
    """Semantic retriever with explicit CPU/CUDA device selection.

    This class mirrors the interface used by create_semantic_tool() and avoids
    failures such as: "Torch not compiled with CUDA enabled".
    """

    def __init__(
        self,
        documents,
        persist_directory="./chroma_db",
        collection_name="semantic_documents",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    ):
        from langchain_chroma import Chroma
        from langchain_core.documents import Document
        from langchain_huggingface import HuggingFaceEmbeddings

        self.documents = documents

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": device},
            encode_kwargs={
                "batch_size": 64,
                "normalize_embeddings": True,
            },
            show_progress=True,
        )

        self.langchain_documents = [
            Document(
                page_content=doc["content"],
                metadata={
                    "index": index,
                    "source": doc.get("source", ""),
                    "page_number": str(doc.get("page_number", "")),
                },
            )
            for index, doc in enumerate(documents)
        ]

        persist_path = Path(persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)

        self.vector_store = Chroma(
            persist_directory=str(persist_path),
            embedding_function=self.embeddings,
            collection_name=collection_name,
        )

        collection_data = self.vector_store.get()
        existing_ids = collection_data.get("ids") or []

        # Only build embeddings when this fingerprinted collection is empty.
        if not existing_ids:
            self.vector_store.add_documents(
                documents=self.langchain_documents,
                ids=[str(i) for i in range(len(self.langchain_documents))],
            )

    def retrieve(self, query, top_k=5):
        results = self.vector_store.similarity_search_with_score(query, k=top_k)

        return [
            {
                "content": doc.page_content,
                "score": score,
                "index": doc.metadata.get("index"),
                "retriever": "semantic",
            }
            for doc, score in results
        ]


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

    # Use our safe retriever. This works with both CPU-only and CUDA-enabled
    # PyTorch installations.
    semantic_retriever = SafeSemanticRetriever(
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

    retriever_agent = RetrieverAgent(
        model=model,
        tools=[bm25_tool, semantic_tool],
    )

    main_agent = MainAgent(model=model)

    return {
        "documents": documents,
        "semantic_documents": semantic_documents,
        "retriever_agent": retriever_agent,
        "main_agent": main_agent,
        "semantic_collection": collection_name,
        "semantic_device": semantic_device,
    }


# -----------------------------------------------------------------------------
# UI helpers
# -----------------------------------------------------------------------------

def render_sidebar(rag, data_dir: Path, chroma_dir: Path) -> None:
    with st.sidebar:
        st.header("Configuration")
        st.caption("BM25 for exact lookups + semantic retrieval for meaning-based queries.")

        st.metric("Original document chunks", len(rag["documents"]))
        st.metric("Semantic chunks", len(rag["semantic_documents"]))

        st.divider()
        st.subheader("Semantic model")
        st.write(f"Device: **{rag['semantic_device']}**")

        if rag["semantic_device"] == "cpu":
            st.caption(
                "CPU mode is active because your installed PyTorch does not provide CUDA."
            )

        st.divider()
        st.subheader("Data")
        st.caption(str(data_dir))

        st.divider()
        st.subheader("Retrieval")
        st.write("**BM25** — exact terms, names, identifiers, table values")
        st.write("**Semantic** — conceptual and paraphrased questions")

        st.divider()
        st.subheader("Index")
        st.caption(f"Chroma: {chroma_dir}")
        st.caption(f"Collection: `{rag['semantic_collection']}`")

        st.divider()
        if st.button("Reload document index", use_container_width=True):
            load_documents.clear()
            build_semantic_documents.clear()
            semantic_index_id.clear()
            initialize_rag.clear()
            st.session_state.messages = []
            st.rerun()

        if st.button("Rebuild semantic index", use_container_width=True):
            if chroma_dir.exists():
                shutil.rmtree(chroma_dir)
            load_documents.clear()
            build_semantic_documents.clear()
            semantic_index_id.clear()
            initialize_rag.clear()
            st.session_state.messages = []
            st.rerun()

        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def render_message(message: dict[str, Any]) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("retriever"):
            st.caption(f"Retriever: {message['retriever']}")
            context = message.get("context")
            if context:
                with st.expander("Retrieved context"):
                    st.text(context)


# -----------------------------------------------------------------------------
# Main application
# -----------------------------------------------------------------------------

def main() -> None:
    st.title("💬 Document RAG Chatbot")
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

    render_sidebar(rag, data_dir, chroma_dir)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        render_message(message)

    query = st.chat_input("Ask a question about your documents...")
    if not query:
        return

    st.session_state.messages.append({
        "role": "user",
        "content": query,
    })

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving relevant information and generating an answer..."):
                retrieval_result = rag["retriever_agent"].retrieve(query)
                answer = rag["main_agent"].response(
                    query=query,
                    context=retrieval_result["context"],
                )

            st.markdown(answer)
            st.caption(f"Retriever: {retrieval_result['retriever']}")

            with st.expander("Retrieved context"):
                st.text(retrieval_result["context"])

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "retriever": retrieval_result["retriever"],
                "context": retrieval_result["context"],
            })

        except Exception as exc:
            error_message = f"I could not process that query: {exc}"
            st.error(error_message)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message,
            })


if __name__ == "__main__":
    main()
