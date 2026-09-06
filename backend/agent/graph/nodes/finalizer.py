"""Finalizer node: produces the user-facing answer. No tools, no schema."""

from langchain_core.messages import HumanMessage, SystemMessage

from agent.graph.states import AgentState
from agent.services.config import read_system_prompt
from agent.services.llm_model_factory import create_chat_model


class FinalizerNode:
    """Summarizes the whole run into the final assistant message."""

    def __init__(self, model_config: dict):
        system_prompt = read_system_prompt(model_config)
        self._system = SystemMessage(
            content=system_prompt) if system_prompt else None
        self._model = create_chat_model(model_config)

    def __call__(self, state: AgentState) -> dict:
        prompt = HumanMessage(content=(
            "Provide the final answer to the user. Summarize the simulated "
            "production result, the best parameter changes, safety/slugging "
            "trade-offs, and any CSV paths. Do not run another tool."
        ))
        messages = state["messages"] + [prompt]
        if self._system:
            messages = [self._system, *messages]

        response = self._model.invoke(messages)
        return {
            "messages": [response]
        }
