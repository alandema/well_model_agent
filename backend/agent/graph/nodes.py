from typing import Literal

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from agent.graph.state import AgentState

class JudgeOutput(BaseModel):
    """Small, machine-readable contract for the judge."""

    decision: Literal["accept", "reject"] = Field(
        description="Whether the evaluator's instructions should be accepted or rejected."
    )
    justification: str = Field(
        description="Concise justification for the decision, to be used as feedback for the generator."
    )


def create_nodes(generator_model, evaluator_model, judge_model,
                 finalizer_model):
    """Create the workflow nodes with their configured models."""

    def generator(state: AgentState):

        response = generator_model.invoke({
            "messages": state["messages"]
        })
        return {"messages": [response]}

    def evaluator(state: AgentState):
        response = evaluator_model.invoke({
            "messages": state["messages"][-1]
        })
        return {"messages": [response]}

    def judge(state: AgentState):
        response = judge_model.invoke({
            "messages": state["messages"][-1]
        })
        return {"messages": [response]}

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

    return generator, evaluator, judge, finalize
