"""Human-in-the-loop (HITL) wrapper for LangChain tools.

Wraps any tool so that before execution, the graph pauses via interrupt()
and waits for a human to approve, edit, or reject the tool call.
"""

from typing import Any
from langgraph.types import interrupt
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool


def wrap_for_hitl(tool: StructuredTool) -> StructuredTool:
    """Wrap a tool so it requires human approval before executing.

    When the agent calls the wrapped tool, the graph pauses and surfaces
    the tool name + arguments.  The human can resume with:
      - {"action": "approve"}              -> runs the tool as-is
      - {"action": "edit", "args": {...}}  -> runs with modified args
      - {"action": "reject", "reason": "..."} -> sends feedback to the LLM
    """

    original = tool.func
    original_name = tool.name
    original_description = tool.description or ""
    original_args_schema = tool.args_schema

    def _hitl_func(config: RunnableConfig, **tool_input: Any) -> str:
        """Interrupts for human approval before running the original tool."""
        response = interrupt({
            "tool": original_name,
            "args": tool_input,
            "description": f"Approve the call to '{original_name}'?",
        })

        if isinstance(response, str):
            # Simple string resume → treat as approval
            return original(**tool_input)

        action = response.get("action", "approve")

        if action == "approve":
            return original(**tool_input)

        elif action == "edit":
            edited_args = response.get("args", tool_input)
            return original(**edited_args)

        elif action == "reject":
            reason = response.get("reason", "User rejected the tool call.")
            return f"Tool call to '{original_name}' was rejected. Reason: {reason}"

        else:
            return f"Unknown action '{action}'. Tool call skipped."

    # Build a new StructuredTool that mirrors the original but uses the
    # HITL wrapper.  The config is injected so interrupt() can track state.
    hitl_tool = StructuredTool.from_function(
        func=_hitl_func,
        name=original_name,
        description=original_description + " (⚠️ Requires human approval)",
        args_schema=original_args_schema,
    )

    return hitl_tool