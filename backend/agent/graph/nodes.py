from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from agent.graph.states import AgentState


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

        if isinstance(state.get("messages")[-1], HumanMessage) and not state.get("generator_messages"):
            state["generator_messages"] = [state.get("messages")[-1]]
        elif isinstance(state.get("evaluator_messages")[-1], AIMessage):
            state["generator_messages"].append(
                HumanMessage(content=state.get("evaluator_messages")[-1].content))
            state["evaluator_messages"] = []

        response = generator_model.invoke({
            "messages": state["generator_messages"]
        })

        return_state = {
            "messages": [response],
            "generator_messages": state.get("generator_messages", []) + [response],
            "evaluator_messages": state.get("evaluator_messages", [])
        }

        return return_state

    def evaluator(state: AgentState):
        if isinstance(state.get("messages")[-1], ToolMessage) and not state.get("evaluator_messages"):
            state["evaluator_messages"] = state.get("messages")[-2:]

        response = evaluator_model.invoke({
            "messages": state["evaluator_messages"]
        })

        return {
            "messages": [response],
            "evaluator_messages": state.get("evaluator_messages", []) + [response]
        }

    def judge(state: AgentState):
        response = judge_model.invoke({
            "messages": [HumanMessage(content=(
                "Based on the evaluator's output, decide whether to accept or reject the instructions."
                f"```\n{state.get('evaluator_messages', [])[-1].text if state.get('evaluator_messages', []) else 'No output yet.'}\n```"
            ))]
        })

        return {
            "decision": response.decision,
            "justification": response.justification
        }

    def finalize(state: AgentState):
        prompt = HumanMessage(content=(
            "Provide the final answer to the user. Summarize the simulated "
            "production result, the best parameter changes, safety/slugging "
            "trade-offs, and any CSV paths. Do not run another tool."
        ))
        response = finalizer_model.invoke({
            "messages": state["messages"] + [prompt]
        })
        return {
            "messages": [response]
        }

    return generator, evaluator, judge, finalize
