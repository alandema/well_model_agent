from langgraph.graph import END

from agent.graph.states import AgentState


def route_model_output(state: AgentState) -> str:
    """Route tool requests to the tools node or finish the turn."""
    last_message = state["messages"][-1]
    return "tools" if last_message.tool_calls else END
