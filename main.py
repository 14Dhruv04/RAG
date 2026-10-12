"""Command-line entry point for the same LangGraph RAG agent used by app.py."""

import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from agents.main_agent import MainAgent
from parser import CSVparser, Excelparser, PDFparser
from retriever.bm25 import BM25Retriever
from retriever.semantic import SemanticRetriever
from tools.bm25_tool import create_bm25_tool
from tools.semantic_tool import create_semantic_tool


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def load_documents(data_dir: Path) -> list[dict]:
    """Parse supported files and preserve their source metadata."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    documents: list[dict] = []
    for file_path in sorted(data_dir.iterdir()):
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

        for document in parsed:
            enriched = dict(document)
            enriched["source"] = file_path.name
            enriched["source_type"] = suffix
            documents.append(enriched)

    if not documents:
        raise ValueError(f"No supported documents found in {data_dir}")

    return documents


def main() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to .env.")

    configured_data_dir = os.getenv("DATA_DIR", "data")
    data_dir = Path(configured_data_dir)
    if not data_dir.is_absolute():
        data_dir = BASE_DIR / data_dir

    chroma_dir = Path(os.getenv("CHROMA_DIR", "chromadb"))
    if not chroma_dir.is_absolute():
        chroma_dir = BASE_DIR / chroma_dir

    documents = load_documents(data_dir)
    bm25_retriever = BM25Retriever(documents)

    corpus_digest = hashlib.sha256()
    for document in documents:
        corpus_digest.update(str(document.get("source", "")).encode("utf-8"))
        corpus_digest.update(b"\0")
        corpus_digest.update(str(document.get("page_number", "")).encode("utf-8"))
        corpus_digest.update(b"\0")
        corpus_digest.update(document.get("content", "").encode("utf-8"))
        corpus_digest.update(b"\0")
    collection_name = f"semantic_cli_{corpus_digest.hexdigest()[:16]}"

    semantic_retriever = SemanticRetriever(
        documents,
        persist_directory=str(chroma_dir),
        collection_name=collection_name,
        device=os.getenv("SEMANTIC_DEVICE", "cpu"),
    )

    tools = [create_bm25_tool(bm25_retriever), create_semantic_tool(semantic_retriever)]

    model = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        google_api_key=api_key,
    )
    agent = MainAgent(model=model, tools=tools)

    query = input("Enter query: ").strip()
    if not query:
        print("No query entered.")
        return

    def print_event(event: dict) -> None:
        event_type = event.get("type")
        if event_type == "status":
            print(f"\n[{event.get('message', '')}]")
        elif event_type == "tool_call":
            print(f"\n[Tool call: {event.get('name')}]")
            print(f"Arguments: {event.get('args', {})}")
        elif event_type == "tool_result":
            print(f"\n[Tool result: {event.get('name')}]")
            print(event.get("content", ""))

    answer = agent.response(query=query, on_event=print_event)
    print("\nAnswer:")
    print(answer)


if __name__ == "__main__":
    main()
