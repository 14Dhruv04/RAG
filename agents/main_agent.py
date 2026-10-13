
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from prompts.main_agent_prompts import MAIN_SYSTEM_INSTRUCTION

def get_message_field(message, field, default=None):
    """Read a field from a LangChain message or dictionary."""
    if isinstance(message, dict):
        return message.get(field, default)

    return getattr(message, field, default)


def content_to_text(content):
    """Convert LangChain message content into readable text."""
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

        self.model_with_tools = model.bind_tools(tools)

        builder = StateGraph(MessagesState)

        builder.add_node("agent", self.call_model)
        builder.add_node("tools", ToolNode(tools))

        builder.add_edge(START, "agent")

        builder.add_conditional_edges(
            "agent",
            tools_condition,
            {
                "tools": "tools",
                "__end__": END,
            },
        )

        builder.add_edge("tools", "agent")

        self.graph = builder.compile()

    def call_model(self, state: MessagesState):
        """Invoke the model using the current graph messages."""
        response = self.model_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def response(
        self,
        query: str,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> str:
        """Run the graph and emit events for the Streamlit UI."""

        def emit(event_type, **details):
            if on_event is not None:
                on_event({
                    "type": event_type,
                    **details,
                })

        emit(
            "status",
            message="LangGraph execution started. Processing the query...",
        )

        initial_state = {
            "messages": [
                SystemMessage(content=MAIN_SYSTEM_INSTRUCTION),
                HumanMessage(content=query),
            ]
        }

        answer = ""
        tool_names_by_id = {}

        for update in self.graph.stream(
            initial_state,
            stream_mode="updates",
        ):
            for node_name, node_update in update.items():
                if not isinstance(node_update, dict):
                    continue

                messages = node_update.get("messages", [])

                if not isinstance(messages, (list, tuple)):
                    continue

                for message in messages:
                    tool_calls = (
                        get_message_field(message, "tool_calls", [])
                        or []
                    )

                    for tool_call in tool_calls:
                        name = get_message_field(
                            tool_call, "name", "unknown"
                        )
                        args = get_message_field(
                            tool_call, "args", {}
                        )
                        call_id = get_message_field(
                            tool_call, "id"
                        )

                        if call_id:
                            tool_names_by_id[call_id] = name

                        emit(
                            "tool_call",
                            name=name,
                            args=args,
                        )

                    content = content_to_text(
                        get_message_field(message, "content", "")
                    )
                    message_type = get_message_field(
                        message, "type", ""
                    )

                    if node_name == "tools" or message_type == "tool":
                        call_id = get_message_field(
                            message, "tool_call_id"
                        )

                        tool_name = (
                            get_message_field(message, "name")
                            or tool_names_by_id.get(call_id)
                            or "tool"
                        )

                        emit(
                            "tool_result",
                            name=tool_name,
                            content=content,
                            char_count=len(content),
                        )

                    elif (
                        node_name == "agent"
                        and not tool_calls
                        and content.strip()
                    ):
                        answer = content

        if not answer:
            raise RuntimeError(
                "The LangGraph workflow finished without producing a final answer."
            )

        emit(
            "status",
            message="LangGraph execution completed. Final answer generated.",
        )

        return answer