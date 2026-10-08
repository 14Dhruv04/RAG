import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

class SemanticRetriever:
    def __init__(self, documents, persist_directory = "./chroma_db", collection_name = "semantic_documents", embedding_model = "sentence-transformers/all-MiniLM-L6-v2"):
        self.documents = documents

        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs={"device": "cpu"}, encoder_kwargs={"normalize_embeddings": True})

        self.langchain_documents = [Document(page_content = doc["content"], metadata = {"index": index}) for index, doc in enumerate(documents)]

        if os.path.exists(persist_directory):
            self.vector_store = Chroma(persist_directory=persist_directory, embedding_function=self.embeddings, collection_name=collection_name)

            collection_data = self.vector_store.get()
            if not collection_data["ids"]:
                self.vector_store.add_documents(
                    documents=self.langchain_documents,
                    ids=[str(i) for i in range(len(self.langchain_documents))]
                )
        else:
            self.vector_store = Chroma.from_documents(
                documents=self.langchain_documents,
                embedding=self.embeddings,
                collection_name=collection_name,
                persist_directory=persist_directory,
                ids=[str(i) for i in range(len(self.langchain_documents))]
            )


    def retrieve(self, query, top_k=5):
        results = self.vector_store.similarity_search_by_vector(
            query,
            k=top_k
        )

        return [
            {
                "content": doc.page_content,
                "score": score,
                "doc_id": doc.metadata["doc_id"],
                "retriever": "semantic"
            }
            for doc, score in results
        ]

