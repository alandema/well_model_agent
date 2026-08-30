MAX_ITERATIONS = 5


def route_generator(state):
    if state.get("iteration", 0) >= MAX_ITERATIONS:
        return "finalize"
    last = state["messages"][-1]
    return "generator_tools" if getattr(last, "tool_calls", None) else "finalize"


def route_evaluator_tools(state):
    if state.get("iteration", 0) >= MAX_ITERATIONS:
        return "finalize"
    last = state["messages"][-1]
    return "evaluator_tools" if getattr(last, "tool_calls", None) else "judge"


def route_judge(state):
    decision = state.get("decision")
    if state.get("iteration", 0) >= MAX_ITERATIONS:
        return "finalize"
    if decision == "accept":
        return "Generator"
    return "Evaluator"
