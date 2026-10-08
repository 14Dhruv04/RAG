from langchain_core.tools import tool

def create_bm25_tool(bm25_retriever):

    @tool
    def bm25_search(query: str) -> list:
        """
        Search the document collection using BM25 keyword-based retrieval.
        Use this for exact keywords, technical terms, names, identifiers,
        and phrases where lexical matching is important.
        """
        
        return bm25_retriever.retrieve(query, top_k=5)

    return bm25_search