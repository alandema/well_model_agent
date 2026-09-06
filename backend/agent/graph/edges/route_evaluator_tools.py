"""Route after the Evaluator node."""


def route_evaluator_tools(state) -> str:
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
