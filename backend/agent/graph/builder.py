from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agent.graph.edges import route_model_output
from agent.graph.nodes.model_node import ModelNode
from agent.graph.states import AgentState


def create_graph(llm_model_config: dict):
    # The API passes the complete config file; accepting a single model entry
    # as well keeps this builder useful on its own.
    model_config = llm_model_config.get("principal", llm_model_config)

    builder = StateGraph(AgentState)
    model_node = ModelNode(model_config)

    builder.add_node("model", model_node)
    builder.add_node("tools", model_node.tool_node())

    builder.add_edge(START, "model")
    builder.add_conditional_edges(
        "model",
        route_model_output,
        {"tools": "tools", END: END},
    )
    builder.add_edge("tools", "model")

    return builder.compile(checkpointer=InMemorySaver())
