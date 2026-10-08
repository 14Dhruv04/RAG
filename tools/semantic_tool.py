from langchain_core.tools import tool

def create_semantic_tool(semantic_retriever):

    @tool
    def semantic_search(query: str) -> list:
        """
        Search the document collection using semantic similarity.
        Use this for conceptual questions, paraphrases, and queries where
        understanding the meaning is more important than exact keywords.
        """
        
        return semantic_retriever.retrieve(query, top_k=5)

    return semantic_search