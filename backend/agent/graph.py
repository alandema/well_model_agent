import os
import sqlite3
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from agent.services.llm_model_factory import create_llm_model
from agent.state import AgentState
from langgraph.checkpoint.sqlite import SqliteSaver


from agent.tools.fowm import fowm_model
from agent.tools.web_search import web_search
# from agent.tools.unit_conversion import convert_units

all_tools = [fowm_model, web_search]  # , convert_units]


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

    # Build the graph
    builder = StateGraph(AgentState)

    builder.add_node("call_model", call_model)
    builder.add_node("tools", ToolNode(all_tools))

    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", tools_condition)
    builder.add_edge("tools", "call_model")

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
