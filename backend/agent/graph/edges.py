MAX_ITERATIONS = 3


def route_generator(state):
    last = state["messages"][-1]
    return "generator_tools" if getattr(last, "tool_calls", None) else "finalize"


def route_evaluator_tools(state):
    last = state["messages"][-1]
    return "evaluator_tools" if getattr(last, "tool_calls", None) else "judge"


def route_judge(state):
    decision = state.get("decision")
    if state.get("iteration", 0) >= MAX_ITERATIONS:
        return "finalize"
    if decision == "accept":
        return "Generator"
    return "Evaluator"
