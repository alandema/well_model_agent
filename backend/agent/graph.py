import os
import sqlite3
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage
from agent.services.llm_model_factory import create_llm_model
from agent.state import AgentState
from langgraph.checkpoint.sqlite import SqliteSaver


from agent.tools.fowm_model import fowm_model
from agent.tools.web_search import web_search
from agent.tools.read_csv import read_csv
# from agent.tools.unit_conversion import convert_units

all_tools = [fowm_model, web_search, read_csv]  # , convert_units]
MAX_CALLS_PER_TOOL = 3


def route_after_model(state: AgentState):
    """Stop if the model requests the same tool more than three times."""
    last_message = state["messages"][-1]
    current_tool_calls = getattr(last_message, "tool_calls", None)
    if not current_tool_calls:
        return END

    messages_since_user = []
    for message in reversed(state["messages"]):
        if getattr(message, "type", None) == "human":
            break
        messages_since_user.append(message)

    tool_call_counts = {}
    for message in messages_since_user:
        for tool_call in getattr(message, "tool_calls", []):
            tool_name = tool_call["name"]
            tool_call_counts[tool_name] = tool_call_counts.get(
                tool_name, 0) + 1

    if any(
        tool_call_counts.get(tool_call["name"], 0) > MAX_CALLS_PER_TOOL
        for tool_call in current_tool_calls
    ):
        return "stop"

    return "tools"


def create_graph(llm_model_config: dict):
    # The factory returns a Runnable that already has the system prompt
    # and tools attached (prompt | model.bind_tools(tools)).
    principal_model = create_llm_model(
        llm_model_config.get("principal", {}), tools=all_tools
    )

    def call_model(state: AgentState):
        """Single node that calls the configured model.

        The system prompt and tools are already attached to the model in
        the factory, so this node only needs to invoke it on the messages.
        """
        response = principal_model.invoke({"messages": state["messages"]})
        return {"messages": [response]}

    def stop_after_max_tool_rounds(state: AgentState):
        return {
            "messages": [
                AIMessage(
                    content=(
                        "I stopped after reaching the maximum number of tool "
                        "calls for one tool. Please clarify the request before "
                        "trying again."
                    )
                )
            ]
        }

    # Build the graph
    builder = StateGraph(AgentState)

    builder.add_node("call_model", call_model)
    builder.add_node("tools", ToolNode(all_tools))
    builder.add_node("stop", stop_after_max_tool_rounds)

    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", route_after_model)
    builder.add_edge("tools", "call_model")
    builder.add_edge("stop", END)

    # Create checkpoints directory if it doesn't exist
    checkpoint_dir = os.path.join(
        os.path.dirname(__file__), "..", ".checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize SQLite checkpointer
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoints.sqlite")
    conn = sqlite3.connect(checkpoint_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    graph = builder.compile(checkpointer=checkpointer)
    return graph
