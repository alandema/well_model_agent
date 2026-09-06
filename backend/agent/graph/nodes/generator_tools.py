"""Generator tool node: runs FOWM tools and extracts the simulated params."""

import json

from langgraph.prebuilt import ToolNode

from agent.graph.nodes.generator import GENERATOR_TOOLS
from agent.graph.states import AgentState


class GeneratorToolsNode:
    """Executes generator tool calls and captures the FOWM operating point.

    The FOWM tool echoes the resolved operating point back in its response
    ("inputs_used"). That value is grabbed from the tool result — not the
    tool call args — so defaults filled by Pydantic are included.
    """

    def __init__(self):
        # messages_key tells the ToolNode which state channel holds the
        # agent's pending tool_calls; output ToolMessages use the same key.
        self._tool_node = ToolNode(
            GENERATOR_TOOLS, messages_key="generator_messages")

    def __call__(self, state: AgentState) -> dict:
        result = self._tool_node.invoke(state)
        # ToolNode emits ToolMessages under the messages_key, so both
        # channels are fed from result["generator_messages"]: the shared
        # transcript (the Evaluator seeds from it) and the private one.
        tool_messages = result["generator_messages"]

        # ToolNode serializes dict returns into JSON strings, so try both.
        pending_params = None
        for msg in reversed(tool_messages):
            content = getattr(msg, "content", None)
            if isinstance(content, dict) and "inputs_used" in content:
                pending_params = content["inputs_used"]
                break
            if isinstance(content, str):
                try:
                    parsed = json.loads(content)
                except (json.JSONDecodeError, ValueError):
                    continue
                if isinstance(parsed, dict) and "inputs_used" in parsed:
                    pending_params = parsed["inputs_used"]
                    break

        return {
            "messages": tool_messages,
            "generator_messages": state.get("generator_messages", []) + tool_messages,
            "pending_params": pending_params,
        }
