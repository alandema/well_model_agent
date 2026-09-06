"""Route after the Judge node."""

from agent.graph.constants import MAX_ITERATIONS


def route_judge(state) -> str:
    decision = state.get("decision")
    if state.get("iteration", 0) >= MAX_ITERATIONS:
        return "finalize"
    if decision == "accept":
        return "Generator"
    return "Evaluator"
