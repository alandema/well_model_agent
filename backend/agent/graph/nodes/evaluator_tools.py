"""Evaluator tool node: runs analysis tools in the evaluator's channel."""

from langgraph.prebuilt import ToolNode

from agent.graph.nodes.evaluator import EVALUATOR_TOOLS
from agent.graph.states import AgentState


class EvaluatorToolsNode:
    """Executes evaluator tool calls inside the private message channel."""

    def __init__(self):
        self._tool_node = ToolNode(
            EVALUATOR_TOOLS, messages_key="evaluator_messages")

    def __call__(self, state: AgentState) -> dict:
        result = self._tool_node.invoke(state)
        return {
            # Tool results stay in the evaluator's private channel.
            "evaluator_messages": state.get("evaluator_messages", []) + result["evaluator_messages"],
        }
