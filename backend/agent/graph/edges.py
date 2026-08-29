MAX_ITERATIONS = 3


def route_generator(state):
    last = state["messages"][-1]
    return "generator_tools" if getattr(last, "tool_calls", None) else "finalize"


def route_evaluator_tools(state):
    last = state["messages"][-1]
    return "evaluator_tools" if getattr(last, "tool_calls", None) else "judge"


def route_judge(state):
    evaluation = state.get("evaluation", {})
    if (evaluation.get("grade") == "acceptable" or
            state.get("iteration", 0) >= MAX_ITERATIONS):
        return "finalize"
    return "Generator"
