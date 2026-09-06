MAX_ITERATIONS = 5


def route_generator(state):
    if state.get("iteration", 0) >= MAX_ITERATIONS:
        return "finalize"
    last = state["messages"][-1]
    return "generator_tools" if getattr(last, "tool_calls", None) else "finalize"


def route_evaluator_tools(state):
    # The evaluator's traffic lives in its private channel now.
    last = state["evaluator_messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    # The EvaluatorOutput tool call is the structured final answer, not a
    # tool to execute: analysis is complete, hand over to the judge.
    if not tool_calls or any(
        tc.get("name") == "EvaluatorOutput" for tc in tool_calls
    ):
        return "judge"
    return "evaluator_tools"


def route_judge(state):
    decision = state.get("decision")
    if state.get("iteration", 0) >= MAX_ITERATIONS:
        return "finalize"
    if decision == "accept":
        return "Generator"
    return "Evaluator"
