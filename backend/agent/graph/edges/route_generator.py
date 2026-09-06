"""Route after the Generator node."""

from agent.graph.constants import MAX_ITERATIONS


def route_generator(state) -> str:
    if state.get("iteration", 0) >= MAX_ITERATIONS:
        return "finalize"
    last = state["messages"][-1]
    return "generator_tools" if getattr(last, "tool_calls", None) else "finalize"
