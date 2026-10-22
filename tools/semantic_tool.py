from langchain_core.tools import tool
from guards.jailbreak_detection import document_sanitizer

def create_semantic_tool(semantic_retriever, llama_guard):

    @tool
    def semantic_search(query: str) -> list:
        """
        Search the document collection using semantic similarity.
        Use this for conceptual questions, paraphrases, and queries where
        understanding the meaning is more important than exact keywords.
        """

        context = semantic_retriever.retrieve(query, top_k=5)
        context = document_sanitizer(documents=context, llama_guard=llama_guard)
        
        return context

    return semantic_search