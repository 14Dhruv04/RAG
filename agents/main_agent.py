from langchain.agents import create_agent

MAIN_SYSTEM_INSTRUCTION = """
You are the main answer-generation agent in a RAG system.
You are provided with multiple retrieval tools that you can choose between the tools to retrieve context in order to answer user queries.
The retrieval tools available for you are:
    - bm25_tool : BM25 Retriever
    - semantic_tool : Semantic Search Retriever

Use BM25 (bm25_tool) when:
- Exact keywords are important.
- The query contains technical terms, names, identifiers, or specific phrases.
- The user is looking for an exact mention in the documents.

Use semantic search (semantic_tool) when:
- The query is conceptual.
- The user is asking about meaning, explanation, reasoning, or paraphrased information.
- The exact wording may not appear in the documents.

## Workflow to follow:
- Assess the user query to determine if you need more context to answer the query or not.
- If more context is needed choose the most appropriate retrieval tool available to you.
- Use the selected tool to retrieve context.
- Use the retrieved context to answer the user query.
- If a query is general and answerable without having to retrieve extra context, answer it directly.

## Rules:
- If you use a Retrieval tool:
    - DO NOT answer the question yourself ONLY use the retrived context to answer.
    - Use the retrieved context as the primary source of information.
    - Do not invent facts that are not supported by the retrieved context.
- Give a direct and useful answer to the user's question.
- Do not discuss the internal retrieval process unless the user explicitly asks.
"""


def get_message_field(message, field, default=None):
    """Read a field from a LangChain message or a dictionary."""
    if isinstance(message, dict):
        return message.get(field, default)

    return getattr(message, field, default)


def content_to_text(content):
    """Convert LangChain message content into a readable string."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)

        return "\n".join(parts)

    return "" if content is None else str(content)


class MainAgent:
    def __init__(self, model, tools):
        self.tools = tools
        self.agent = create_agent(
            model=model,
            tools=tools,
            system_prompt=MAIN_SYSTEM_INSTRUCTION,
        )

    def response(self, query, on_event=None):
        prompt = f"""
        Retrieve relevant context to the user query using the
        available tools and answer the user's questions.

        =========================
        User Query: {query}
        =========================
        """

        def emit(event_type, **details):
            """Send observable execution events to the UI."""
            if on_event:
                on_event({
                    "type": event_type,
                    **details,
                })

        emit("status", message="Agent run started. Processing the query...")

        answer = ""

        for update in self.agent.stream(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
            },
            stream_mode="updates",
        ):
            for node_name, state in update.items():
                if not isinstance(state, dict):
                    continue

                messages = state.get("messages", [])

                if not isinstance(messages, (list, tuple)):
                    continue

                for message in messages:
                    tool_calls = (get_message_field(message, "tool_calls", []) or [])

                    for tool_call in tool_calls:
                        name = get_message_field(tool_call, "name", "unknown")
                        args = get_message_field(tool_call, "args", {})

                        emit("tool_call", name=name, args=args)

                    content = content_to_text(get_message_field(message, "content", ""))
                    message_type = get_message_field(message, "type", "")

                    if message_type == "tool" or node_name == "tools":
                        tool_name = (get_message_field(message, "name") or node_name)

                        emit(
                            "tool_result",
                            name=tool_name,
                            content = content,
                            char_count=len(content),
                        )

                    elif (not tool_calls and content.strip() and (message_type in ("ai", "assistant") or node_name == "model")):
                        answer = content

        if not answer:
            raise RuntimeError("The agent finished without producing a final answer.")

        emit("status", message="Agent run completed. Final answer generated.")

        return answer