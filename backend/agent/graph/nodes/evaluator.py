"""Evaluator node: analyzes simulation results and emits structured instructions.

Also defines EvaluatorOutput — the structured contract this node produces.
Consumers (generator, judge, edges) import it from here.
"""

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from agent.graph.constants import MAX_ITERATIONS
from agent.graph.states import AgentState
from agent.services.config import read_system_prompt
from agent.services.llm_model_factory import create_chat_model
from agent.tools.python_repl import python_repl
from agent.tools.read_csv import read_csv
from agent.tools.summarize_csv import summarize_csv

EVALUATOR_TOOLS = [summarize_csv, read_csv, python_repl]


class EvaluatorOutput(BaseModel):
    """Small, machine-readable contract for the evaluator."""

    operational_state: Literal["steady", "slugging", "unknown"] = Field(
        description="The operational state of the simulated production run."
    )
    generator_instructions: str = Field(
        description="The instructions for the generator to follow in the next iteration."
    )
    justification: str = Field(
        description="Concise justification for the operational state, to be used as feedback for the generator."
    )


class EvaluatorNode:
    """Analyzes simulated runs and proposes the next operating point.

    Owns both of its models: with analysis tools (used during the loop) and
    EvaluatorOutput-only (used once the iteration cap is reached, so the
    final answer carries no further analysis tool calls).
    """

    def __init__(self, model_config: dict):
        system_prompt = read_system_prompt(model_config)
        self._system = SystemMessage(
            content=system_prompt) if system_prompt else None
        raw_model = create_chat_model(model_config)
        # Schema-as-tool: the structured final answer is bound as an extra
        # tool the model can call alongside ordinary analysis tools.
        self._with_analysis = raw_model.bind_tools(
            EVALUATOR_TOOLS + [EvaluatorOutput], strict=True)
        self._final_only = raw_model.bind_tools([EvaluatorOutput], strict=True)

    def __call__(self, state: AgentState) -> dict:
        if isinstance(state.get("messages")[-1], ToolMessage) and not state.get("evaluator_messages"):
            state["evaluator_messages"] = state.get("messages")[-2:]

        # When the iteration limit is reached, force the evaluator to stop
        # calling tools and produce a final message instead of routing to
        # finalize. The no-tools variant guarantees the response has no
        # analysis tool_calls, so the router sends it to the judge.
        # Give the model visibility into where it is in the iteration budget.
        # Include the minimal run ledger so the evaluator knows which
        # operating points were already simulated and how each behaved,
        # without any stale tool results in its context.
        ledger = state.get("simulated_runs") or []
        if ledger:
            ledger_lines = "\n".join(
                f"- run {i + 1}: {entry['params']} -> {entry['operational_state']}"
                for i, entry in enumerate(ledger)
            )
        else:
            ledger_lines = "- (no runs logged yet)"
        counter = HumanMessage(content=(
            f"You are on iteration {state.get('iteration', 0) + 1} of {MAX_ITERATIONS}.\n"
        ))

        previous_runs = SystemMessage(content=(
            "The following operating points have already been simulated:\n"
            f"{ledger_lines}\n"
        ))

        if state.get("iteration", 0) >= MAX_ITERATIONS:
            model = self._final_only
        else:
            model = self._with_analysis

        messages = state.get("evaluator_messages", []) + [counter]
        if self._system:
            messages = [self._system] + messages

        messages = [previous_runs] + messages

        response = model.invoke(messages)

        # When the evaluator finished with its structured EvaluatorOutput tool
        # call, parse the payload directly from the tool-call args (no regex
        # over free text), expose it downstream for the Judge/Generator, and
        # commit one ledger entry: {params, operational_state}. This is the
        # only memory kept between evaluator loops — no stale tool results.
        run_entry = None
        evaluator_output = None
        tool_calls = getattr(response, "tool_calls", None) or []
        structured_call = next(
            (tc for tc in tool_calls if tc.get("name") == "EvaluatorOutput"),
            None,
        )
        if structured_call is not None:
            evaluator_output = EvaluatorOutput(
                **structured_call["args"]).model_dump()
            pending_params = state.get("pending_params")
            if pending_params is not None:
                run_entry = {
                    "params": pending_params,
                    "operational_state": evaluator_output["operational_state"],
                }

        return {
            # Keep the evaluator's internal traffic out of the shared
            # `messages` channel; it lives in `evaluator_messages` only.
            "evaluator_messages": state.get("evaluator_messages", []) + [response],
            # Count each evaluator turn as one iteration so the
            # MAX_ITERATIONS checks in the routers actually trigger.
            "iteration": 1,
            # Commit the ledger entry when the cycle completed; consumed
            # params are dropped by setting pending_params back to None.
            # Expose the parsed structured answer for the Judge/Generator;
            # consumed by the Generator next cycle and cleared there.
            **({"evaluator_output": evaluator_output}
               if evaluator_output else {}),
            # Commit the ledger entry when the cycle completed; consumed
            # params are dropped by setting pending_params back to None.
            **({"simulated_runs": [run_entry], "pending_params": None}
               if run_entry else {}),
        }
