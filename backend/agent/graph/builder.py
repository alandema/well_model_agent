import os
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agent.graph.edges import route_evaluator_tools, route_generator, route_judge
from agent.graph.nodes import JudgeOutput, create_nodes
from agent.graph.states import AgentState
from agent.services.llm_model_factory import create_llm_model
from agent.tools.fowm_model import fowm_model
from agent.tools.multi_well_model import multi_well_model
from agent.tools.read_csv import read_csv
from agent.tools.summarize_csv import summarize_csv
from agent.tools.terminal import terminal

GENERATOR_TOOLS = [fowm_model, multi_well_model]
EVALUATOR_TOOLS = [summarize_csv, read_csv, terminal]


def create_graph(llm_model_config: dict):
    generator_model = create_llm_model(
        llm_model_config["Generator"],
        tools=GENERATOR_TOOLS
    )
    evaluator_model = create_llm_model(
        llm_model_config["Evaluator"],
        tools=EVALUATOR_TOOLS
    )
    judge_model = create_llm_model(
        llm_model_config["Judge"],
        output_schema=JudgeOutput
    )
    finalizer_model = create_llm_model(
        llm_model_config["finalizer"]
    )

    generator, evaluator, judge, finalize = create_nodes(
        generator_model=generator_model,
        evaluator_model=evaluator_model,
        judge_model=judge_model,
        finalizer_model=finalizer_model
    )

    generator_tool_node = ToolNode(GENERATOR_TOOLS)

    def generator_tools(state: AgentState):
        result = generator_tool_node.invoke(state)

        return {
            "messages": result["messages"],
            "generator_messages": state.get("generator_messages", []) + result["messages"],
        }

    evaluator_tool_node = ToolNode(EVALUATOR_TOOLS)

    def evaluator_tools(state: AgentState):
        result = evaluator_tool_node.invoke(state)

        return {
            # Tool results stay in the evaluator's private channel.
            "evaluator_messages": state.get("evaluator_messages", []) + result["messages"],
        }

    builder = StateGraph(AgentState)
    builder.add_node("Generator", generator)
    builder.add_node("generator_tools", generator_tools)
    builder.add_node("Evaluator", evaluator)
    builder.add_node("evaluator_tools", evaluator_tools)
    builder.add_node("judge", judge)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "Generator")
    builder.add_conditional_edges(
        "Generator", route_generator,
        {
            "generator_tools": "generator_tools",
            "finalize": "finalize"
        },
    )
    builder.add_edge("generator_tools", "Evaluator")
    builder.add_conditional_edges(
        "Evaluator", route_evaluator_tools,
        {
            "evaluator_tools": "evaluator_tools",
            "judge": "judge",
            # route_evaluator_tools can return this when the iteration
            # limit is hit; it must be mapped or LangGraph raises KeyError.
            "finalize": "finalize"
        },
    )
    builder.add_edge("evaluator_tools", "Evaluator")
    builder.add_conditional_edges(
        "judge", route_judge,
        {
            "Generator": "Generator",
            "Evaluator": "Evaluator",
            "finalize": "finalize"
        },
    )
    builder.add_edge("finalize", END)

    checkpoint_dir = os.path.join(
        os.path.dirname(__file__), "..", ".checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoints.sqlite")
    # Single long-lived connection (the graph is created once at startup).
    # WAL + busy_timeout make concurrent FastAPI requests safe: WAL allows
    # a reader and writer in parallel, and busy_timeout makes writers wait
    # instead of failing with "database is locked".
    conn = sqlite3.connect(
        checkpoint_path,
        check_same_thread=False,
        timeout=30,
    )
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA synchronous=NORMAL")
    return builder.compile(checkpointer=SqliteSaver(conn))
