"""Graph wiring: instantiates nodes and connects them. No model or tool
logic lives here — each node owns its own binding and system prompt."""

import os
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from agent.graph.edges import route_evaluator_tools, route_generator, route_judge
from agent.graph.nodes import (
    EvaluatorNode,
    EvaluatorToolsNode,
    FinalizerNode,
    GeneratorNode,
    GeneratorToolsNode,
    JudgeNode,
)
from agent.graph.states import AgentState


def create_graph(llm_model_config: dict):
    builder = StateGraph(AgentState)

    builder.add_node("Generator", GeneratorNode(llm_model_config["Generator"]))
    builder.add_node("generator_tools", GeneratorToolsNode())
    builder.add_node("Evaluator", EvaluatorNode(llm_model_config["Evaluator"]))
    builder.add_node("evaluator_tools", EvaluatorToolsNode())
    builder.add_node("judge", JudgeNode(llm_model_config["Judge"]))
    builder.add_node("finalize", FinalizerNode(llm_model_config["finalizer"]))

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
            "judge": "judge"
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

    return builder.compile(checkpointer=create_checkpointer())


def create_checkpointer() -> SqliteSaver:
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
    return SqliteSaver(conn)
