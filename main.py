import os
from dotenv import load_dotenv

from retriever.bm25 import BM25Retriever
from retriever.semantic import SemanticRetriever
from parser import PDFparser, Excelparser
from tools.bm25_tool import create_bm25_tool
from tools.semantic_tool import create_semantic_tool

from agents.retriever_agent import RetrieverAgent
from agents.main_agent import MainAgent
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(
    model = "gemini-3.5-flash-lite",
    temperature = 0,
    google_api_key = os.getenv("GEMINI_API_KEY")
)

def main():
    # Load documents from the data directory
    data_dir = os.getenv("DATA_DIR")
    documents = PDFparser(data_dir)
    

    # Initialize BM25 retriever
    bm25_retriever = BM25Retriever(documents)

    # Initialize Semantic retriever
    semantic_retriever = SemanticRetriever(documents)

    # tool creation
    bm25_tool = create_bm25_tool(bm25_retriever)
    semantic_tool = create_semantic_tool(semantic_retriever)

    retriever_tools = [bm25_tool, semantic_tool]

    for tool in retriever_tools:
        print(f"Tool Name: {tool.name}")
        print()

    retriever_agent = RetrieverAgent(
        model = model,
        tools = retriever_tools
    )

    main_agent = MainAgent(
        model = model
    )

    query = str(input("Enter query: "))

    retrieval_result = retriever_agent.retrieve(query)

    answer = main_agent.response(
        query = query,
        context = retrieval_result["context"]
    )

    print("\nAnswer:")
    print(answer)

if __name__ == "__main__":
    main()