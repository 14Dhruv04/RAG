from langchain_core.tools import tool
from guards.jailbreak_detection import document_sanitizer

def create_bm25_tool(bm25_retriever, llama_guard):

    @tool
    def bm25_search(query: str) -> list:
        """
        Search the document collection using BM25 keyword-based retrieval.
        Use this for exact keywords, technical terms, names, identifiers,
        and phrases where lexical matching is important.
        """

        context = bm25_retriever.retrieve(query, top_k=5)
        context = document_sanitizer(documents=context, llama_guard=llama_guard)

        return context

    return bm25_search