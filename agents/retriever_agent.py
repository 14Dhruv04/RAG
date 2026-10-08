from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage

RETRIEVER_SYSTEM_PROMPT = """
You are a retrieval routing agent.

## PURPOSE
Your job is to choose the most appropriate retrieval tool for a given user query.

Use BM25 when:
- Exact keywords are important.
- The query contains technical terms, names, identifiers, or specific phrases.
- The user is looking for an exact mention in the documents.

Use semantic search when:
- The query is conceptual.
- The user is asking about meaning, explanation, reasoning, or paraphrased information.
- The exact wording may not appear in the documents.

Choose exactly one retrieval tool.
Do not answer the user's question yourself.
Your job is only to retrieve relevant context.
"""

class RetrieverAgent:
    def __init__(self, model, tools):
        self.tools = tools
        self.agent = create_agent(
            model=model,
            tools=tools,
            system_prompt=RETRIEVER_SYSTEM_PROMPT,
        )


    def retrieve(self, query):
        result = self.agent.invoke(
            {
                "messages": [
                    {
                        "role" : "user",
                        "content" : query
                    }
                ]
            }
        )

        messages = result["messages"]

        selected_tool = None

        for message in messages:
            if(isinstance(message, AIMessage)):
                if message.tool_calls:
                    selected_tool = message.tool_calls[0]["name"]
                    break

        tool_results = [
            message for message in messages if isinstance(message, ToolMessage)
        ]

        if not tool_results:
            raise RuntimeError("Retriever agent did not call a retrieval tool.")

        retrieved_context = tool_results[-1].content

        return {
            "query": query,
            "retriever" : selected_tool,
            "context" : retrieved_context
        }
