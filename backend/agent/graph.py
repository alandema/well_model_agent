import os
import sqlite3
from typing import Literal

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage
from agent.services.llm_model_factory import create_llm_model
from agent.state import AgentState
from langgraph.checkpoint.sqlite import SqliteSaver


from agent.tools.fowm_model import fowm_model
from agent.tools.multi_well_model import multi_well_model
from agent.tools.summarize_csv import summarize_csv
from agent.tools.read_csv import read_csv
from agent.tools.terminal import terminal
from agent.tools.web_search import is_web_search_available, web_search

GENERATOR_TOOLS = [fowm_model, multi_well_model]
EVALUATOR_TOOLS = [summarize_csv, read_csv, terminal]
if is_web_search_available():
    EVALUATOR_TOOLS.append(web_search)
MAX_ITERATIONS = 3


class ProductionEvaluation(BaseModel):
    """Small, machine-readable contract between evaluator and optimizer."""

    grade: Literal["acceptable", "needs_improvement"] = Field(
        description="Whether the simulated production result is acceptable.")
    feedback: str = Field(
        description="Specific changes the optimizer should try next.")
    recommendation: str = Field(
        description="Concise operational recommendation based on the results.")


def create_graph(llm_model_config: dict):
    generator_config = llm_model_config["Generator"]
    evaluator_config = llm_model_config["Evaluator"]
    judge_config = llm_model_config["Judge"]
    finalizer_config = llm_model_config["Finalizer"]

    generator_model = create_llm_model(generator_config, tools=GENERATOR_TOOLS)
    evaluator_tools_model = create_llm_model(
        evaluator_config, tools=EVALUATOR_TOOLS)
    evaluator_model = create_llm_model(
        judge_config, output_schema=ProductionEvaluation)
    finalizer_model = create_llm_model(finalizer_config)

    def generator(state: AgentState):
        """Generate and run the next production-improvement experiment."""
        feedback = state.get("feedback")
        instruction = (
            "Act as the Generator. Run exactly one well model using concrete "
            "parameters. Find a safe parameter change that can increase "
            "production while preventing severe slugging. Use the evaluator "
            "feedback below when present. You must call either fowm_model or "
            "multi_well_model; do not answer without running a model.\n"
            f"Evaluator feedback: {feedback or 'No previous evaluation.'}"
        )
        response = generator_model.invoke({
            "messages": state["messages"] + [HumanMessage(content=instruction)]
        })
        return {"messages": [response], "plan": response.content,
                "iteration": state.get("iteration", 0) + 1}

    def evaluator(state: AgentState):
        """Inspect and judge the Generator outcome."""
        prompt = HumanMessage(content=(
            "Act as the Evaluator. Inspect the latest Generator well-model "
            "result. Use the CSV analysis tools to examine the generated file "
            "and calculate/compare production and slugging indicators. Do not "
            "run another well model.\nGenerator plan: "
            f"{state.get('plan', '')}"
        ))
        response = evaluator_tools_model.invoke({
            "messages": state["messages"] + [prompt]
        })
        return {"messages": [response]}

    def judge(state: AgentState):
        """Return structured feedback for the next Generator iteration."""
        result = evaluator_model.invoke({
            "messages": state["messages"] + [HumanMessage(content=(
                "Judge the Evaluator's analysis and latest Generator result "
                "against the goal "
                "of increasing production while preventing severe slugging. "
                "Return structured feedback only."
            ))]
        })
        return {"feedback": result.feedback,
                "evaluation": result.model_dump()}

    def finalize(state: AgentState):
        prompt = HumanMessage(content=(
            "Provide the final answer to the user. Summarize the simulated "
            "production result, the best parameter changes, safety/slugging "
            "trade-offs, and any CSV paths. Do not run another tool."
        ))
        response = finalizer_model.invoke({
            "messages": state["messages"] + [prompt]
        })
        return {"messages": [response]}

    def route_generator(state: AgentState):
        last = state["messages"][-1]
        return "run_models" if getattr(last, "tool_calls", None) else "finalize"

    def route_evaluator_tools(state: AgentState):
        last = state["messages"][-1]
        return "evaluator_tools" if getattr(last, "tool_calls", None) else "judge"

    def route_judge(state: AgentState):
        evaluation = state.get("evaluation", {})
        if (evaluation.get("grade") == "acceptable" or
                state.get("iteration", 0) >= MAX_ITERATIONS):
            return "finalize"
        return "Generator"

    # Build the graph
    builder = StateGraph(AgentState)

    builder.add_node("Generator", generator)
    builder.add_node("generator_tools", ToolNode(GENERATOR_TOOLS))
    builder.add_node("Evaluator", evaluator)
    builder.add_node("evaluator_tools", ToolNode(EVALUATOR_TOOLS))
    builder.add_node("judge", judge)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "Generator")
    builder.add_conditional_edges(
        "Generator", route_generator,
        {"run_models": "generator_tools", "finalize": "finalize"},
    )
    builder.add_edge("generator_tools", "Evaluator")
    builder.add_conditional_edges(
        "Evaluator", route_evaluator_tools,
        {"evaluator_tools": "evaluator_tools", "judge": "judge"},
    )
    builder.add_edge("evaluator_tools", "judge")
    builder.add_conditional_edges(
        "judge", route_judge,
        {"Generator": "Generator", "finalize": "finalize"},
    )
    builder.add_edge("finalize", END)

    # Create checkpoints directory if it doesn't exist
    checkpoint_dir = os.path.join(
        os.path.dirname(__file__), "..", ".checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize SQLite checkpointer
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoints.sqlite")
    conn = sqlite3.connect(checkpoint_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    graph = builder.compile(checkpointer=checkpointer)
    return graph
