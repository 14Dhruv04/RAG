from langchain_core.tools import tool

def create_bm25_tool(bm25_retriever):

    @tool
    def bm25_search(query: str) -> list:
        return bm25_retriever.retrieve(query, top_k=5)

    return bm25_search