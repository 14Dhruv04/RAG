from langchain_core.tools import tool

def create_semantic_tool(semantic_retriever):

    @tool
    def semantic_search(query: str) -> list:
        return semantic_retriever.retrieve(query, top_k=5)

    return semantic_search