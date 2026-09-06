"""Judge node: accepts or rejects the evaluator's proposed instructions.

Also defines JudgeOutput — the structured contract this node produces.
"""

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Overwrite
from pydantic import BaseModel, Field

from agent.graph.constants import MAX_ITERATIONS
from agent.graph.states import AgentState
from agent.services.config import read_system_prompt
from agent.services.llm_model_factory import create_chat_model


class JudgeOutput(BaseModel):
    """Small, machine-readable contract for the judge."""

    decision: Literal["accept", "reject"] = Field(
        description="Whether the evaluator's instructions should be accepted or rejected."
    )
    justification: str = Field(
        description="Concise justification for the decision, to be used as feedback for the generator."
    )


class JudgeNode:
    """Reviews the evaluator's structured output and accepts or rejects it.

    Uses provider-native structured output (with_structured_output) so the
    response is always a parsed JudgeOutput instance.
    """

    def __init__(self, model_config: dict):
        system_prompt = read_system_prompt(model_config)
        self._system = SystemMessage(
            content=system_prompt) if system_prompt else None
        self._model = create_chat_model(model_config).with_structured_output(
            JudgeOutput)

    def __call__(self, state: AgentState) -> dict:
        output = state.get("evaluator_output") or {}
        summary = (
            f"Operational state: {output.get('operational_state', 'unknown')}\n"
            f"Justification: {output.get('justification', 'N/A')}\n"
            f"Next instruction: {output.get('generator_instructions', 'N/A')}"
        )
        messages = [HumanMessage(content=(
            "Based on the evaluator's output, decide whether to accept or reject the instructions."
            f"\n```\n{summary}\n```"
        ))]
        if self._system:
            messages = [self._system, *messages]

        response = self._model.invoke(messages)

        update = {
            "decision": response.decision,
            "justification": response.justification,
            # Reset iteration count for the next generator-evaluator cycle.
            "iteration": Overwrite(value=0)
        }
        if response.decision == "reject":
            # Feed the judge's justification back into the evaluator's
            # message list so it can re-run its analysis with the feedback.
            update["evaluator_messages"] = state.get("evaluator_messages", []) + [
                HumanMessage(content=(
                    "The judge rejected your last instruction. "
                    f"Reason: {response.justification}\n"
                    "Re-run your analysis and produce a corrected instruction."
                ))
            ]
        return update
