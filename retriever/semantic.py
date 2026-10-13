import os
from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

class SemanticRetriever:
    def __init__(self, documents, persist_directory="./chromadb", collection_name="semantic_documents", embedding_model="sentence-transformers/all-MiniLM-L6-v2", device="cpu"):
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

        if not existing_ids:
            self.vector_store.add_documents(
                documents=self.langchain_documents,
                ids=[str(i) for i in range(len(self.langchain_documents))],
            )

    def retrieve(self, query, top_k=3):
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