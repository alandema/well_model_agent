
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage

from agent.graph.states import AgentState
from agent.services.config import read_system_prompt
from agent.services.llm_model_factory import create_chat_model
from agent.tools import setpoint, read_sqlite


class ModelNode:

    def __init__(self, model_config: dict):
        self._tools = [setpoint, read_sqlite]
        system_prompt = read_system_prompt(model_config)
        self._system = SystemMessage(
            content=system_prompt) if system_prompt else None
        raw_model = create_chat_model(model_config)
        self._model = raw_model.bind_tools(self._tools)

    def __call__(self, state: AgentState) -> dict:
        messages = state["messages"]
        if self._system is not None:
            messages = [self._system, *messages]

        response = self._model.invoke(messages)
        return {"messages": [response]}

    def tool_node(self) -> ToolNode:
        return ToolNode(self._tools)
