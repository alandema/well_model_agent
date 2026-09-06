"""Generator node: simulates operating points and proposes the next run."""

from langchain_core.messages import HumanMessage, SystemMessage

from agent.graph.constants import MAX_ITERATIONS
from agent.graph.states import AgentState
from agent.services.config import read_system_prompt
from agent.services.llm_model_factory import create_chat_model
from agent.tools.fowm_model import fowm_model
from agent.tools.multi_well_model import multi_well_model

GENERATOR_TOOLS = [fowm_model, multi_well_model]


class GeneratorNode:
    """Simulates FOWM operating points and reports the results back.

    Owns its own tool-bound model and system prompt.
    """

    def __init__(self, model_config: dict):
        system_prompt = read_system_prompt(model_config)
        self._system = SystemMessage(
            content=system_prompt) if system_prompt else None
        self._model = create_chat_model(model_config).bind_tools(
            GENERATOR_TOOLS)

    def __call__(self, state: AgentState) -> dict:
        # Work on local copies; never mutate the shared state dict.
        generator_messages = list(state.get("generator_messages") or [])
        evaluator_messages = state.get("evaluator_messages") or []

        if isinstance(state.get("messages")[-1], HumanMessage):
            # Fresh user input (first turn or a follow-up message):
            # always append it to the generator's conversation.
            generator_messages.append(state.get("messages")[-1])
        elif state.get("evaluator_output"):
            # Structured feedback from the evaluator's final EvaluatorOutput
            # tool call — no free-text parsing needed.
            output = state["evaluator_output"]
            generator_messages.append(HumanMessage(content=(
                f"Next instruction: {output.get('generator_instructions')}"
            )))
            # Reset for the next generation, expressed as a state update
            # (evaluator_messages has no reducer, so returning [] overwrites it).
            evaluator_messages = []

        messages = generator_messages
        if self._system:
            messages = [self._system, *messages]

        response = self._model.invoke(messages)

        return {
            "messages": [response],
            "generator_messages": generator_messages + [response],
            "evaluator_messages": evaluator_messages,
            # Plain field (last-write-wins): consumed feedback is cleared so a
            # stale instruction can never leak into a later cycle.
            "evaluator_output": None
        }
