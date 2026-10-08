from langchain.agents import create_agent

MAIN_SYSTEM_INSTRUCTION = """
You are the main answer-generation agent in a RAG system.

## PURPOSE:
Answer the user's question using the retrieved context provided to you.

## Rules:
- Use the retrieved context as the primary source of information.
- Do not invent facts that are not supported by the retrieved context.
- If the retrieved context is insufficient to answer the question, clearly state that.
- Give a direct and useful answer to the user's question.
- Do not discuss the internal retrieval process unless the user explicitly asks.
"""

class MainAgent:
    def __init__(self, model):
        self.agent = create_agent(
            model = model,
            system_prompt= MAIN_SYSTEM_INSTRUCTION
        )


    def response(self, query, context):
        prompt = f"""
        Answer the user's query using the retrieved context.

        =========================
        User Query: {query}
        =========================
        Retrieved Context:
        {context}
        """

        result = self.agent.invoke(
            {
                "messages": [
                    {
                        "role" : "user",
                        "content" : prompt
                    }
                ]
            }
        )

        response = result["messages"][-1].content

        if isinstance(response, list):
            return "\n".join(
                block["text"]
                for block in response
                if isinstance(block, dict) and block.get("type") == "text"
            )

        return response
