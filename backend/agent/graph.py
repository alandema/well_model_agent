import os
import json
import sqlite3
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage
from langchain_openrouter.chat_models import ChatOpenRouter
from langgraph.checkpoint.sqlite import SqliteSaver

from agent.tools.add import add
from agent.tools.multiply import multiply
from agent.tools.fowm_tool import run_fowm
from agent.tools.hitl_wrapper import wrap_for_hitl


# Load model configuration from prompts/config.json
_config_path = os.path.join(os.path.dirname(__file__), "prompts", "config.json")
with open(_config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

_model_cfg = config.get("model", {})

# Tools available to the agent
# hitl_tools are wrapped so they pause for human approval before executing
_raw_hitl_tools = [add, run_fowm]
hitl_tools = [wrap_for_hitl(t) for t in _raw_hitl_tools]
non_hitl_tools = [multiply]

all_tools = hitl_tools + non_hitl_tools


def call_model(state: MessagesState):
    """Single node that calls the configured model with tools bound."""
    model = ChatOpenRouter(
        model=_model_cfg.get("id", "inclusionai/ling-3.0-flash:free"),
        temperature=_model_cfg.get("temperature", 0.7),
        max_tokens=_model_cfg.get("max_tokens"),
        model_kwargs={"extra_body": {"reasoning": _model_cfg.get("reasoning", {})}}
        if _model_cfg.get("reasoning") else None,
    )
    model_with_tools = model.bind_tools(all_tools)
    messages = state["messages"]
    messages = [SystemMessage(content=_model_cfg.get("system_prompt", ""))] + list(messages)
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}


# Build the graph
builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_node("tools", ToolNode(all_tools))
builder.add_edge(START, "call_model")
builder.add_conditional_edges("call_model", tools_condition)
builder.add_edge("tools", "call_model")

# Create checkpoints directory if it doesn't exist
checkpoint_dir = os.path.join(os.path.dirname(__file__), "..", ".checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

# Initialize SQLite checkpointer
checkpoint_path = os.path.join(checkpoint_dir, "checkpoints.sqlite")
conn = sqlite3.connect(checkpoint_path, check_same_thread=False)
checkpointer = SqliteSaver(conn)

graph = builder.compile(checkpointer=checkpointer)