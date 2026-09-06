import re
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from agent.graph.edges import MAX_ITERATIONS
from agent.graph.states import AgentState
from langgraph.types import Overwrite


class JudgeOutput(BaseModel):
    """Small, machine-readable contract for the judge."""

    decision: Literal["accept", "reject"] = Field(
        description="Whether the evaluator's instructions should be accepted or rejected."
    )
    justification: str = Field(
        description="Concise justification for the decision, to be used as feedback for the generator."
    )


def create_nodes(generator_model, evaluator_model, judge_model,
                 finalizer_model, evaluator_model_no_tools=None):
    """Create the workflow nodes with their configured models."""

    def generator(state: AgentState):
        # Work on local copies; never mutate the shared state dict.
        generator_messages = list(state.get("generator_messages") or [])
        evaluator_messages = state.get("evaluator_messages") or []

        if isinstance(state.get("messages")[-1], HumanMessage):
            # Fresh user input (first turn or a follow-up message):
            # always append it to the generator's conversation.
            generator_messages.append(state.get("messages")[-1])
        elif evaluator_messages and isinstance(evaluator_messages[-1], AIMessage):
            generator_messages.append(
                HumanMessage(content=evaluator_messages[-1].content))
            # Reset for the next generation, expressed as a state update
            # (evaluator_messages has no reducer, so returning [] overwrites it).
            evaluator_messages = []

        response = generator_model.invoke({
            "messages": generator_messages
        })

        return_state = {
            "messages": [response],
            "generator_messages": generator_messages + [response],
            "evaluator_messages": evaluator_messages
        }

        return return_state

    def evaluator(state: AgentState):
        if isinstance(state.get("messages")[-1], ToolMessage) and not state.get("evaluator_messages"):
            state["evaluator_messages"] = state.get("messages")[-2:]

        # When the iteration limit is reached, force the evaluator to stop
        # calling tools and produce a final message instead of routing to
        # finalize. Re-invoking the model without bound tools guarantees the
        # response has no tool_calls, so the router sends it to the judge.
        # Give the model visibility into where it is in the iteration budget.
        # Include the minimal run ledger so the evaluator knows which
        # operating points were already simulated and how each behaved,
        # without any stale tool results in its context.
        ledger = state.get("simulated_runs") or []
        if ledger:
            ledger_lines = "\n".join(
                f"- run {i + 1}: {entry['params']} -> {entry['operational_state']}"
                for i, entry in enumerate(ledger)
            )
        else:
            ledger_lines = "- (no runs logged yet)"
        counter = HumanMessage(content=(
            f"You are on iteration {state.get('iteration', 0) + 1} of {MAX_ITERATIONS}.\n"
            "Operating points already simulated (from the run ledger):\n"
            f"{ledger_lines}"
        ))

        if state.get("iteration", 0) >= MAX_ITERATIONS and evaluator_model_no_tools is not None:
            response = evaluator_model_no_tools.invoke({
                "messages": state.get("evaluator_messages", []) + [HumanMessage(content=(
                    "You have reached the maximum number of tool-calling iterations. "
                    "Do not call any more tools. Based on the information gathered so far, "
                    "produce your final instruction now."
                ))]
            })
        else:
            response = evaluator_model.invoke({
                "messages": state["evaluator_messages"] + [counter]
            })

        # When the evaluator produced its final instruction (no tool calls),
        # capture the operational state it classified for the just-simulated
        # params and commit one ledger entry: {params, operational_state}.
        # This is the only memory kept between evaluator loops — no stale
        # tool calls or results.
        run_entry = None
        if not getattr(response, "tool_calls", None):
            pending_params = state.get("pending_params")
            # .text is a property in this langchain-core version; fall back
            # to .content if it's ever a method or missing.
            text = response.text if isinstance(
                getattr(response, "text", None), str) else str(response.content)
            match = re.search(
                r"OPERATIONAL_STATE:\s*(steady|slugging)", text, re.IGNORECASE)
            if pending_params is not None:
                run_entry = {
                    "params": pending_params,
                    "operational_state": match.group(1).lower() if match else "unknown",
                }

        return {
            # Keep the evaluator's internal traffic out of the shared
            # `messages` channel; it lives in `evaluator_messages` only.
            "evaluator_messages": state.get("evaluator_messages", []) + [response],
            # Count each evaluator turn as one iteration so the
            # MAX_ITERATIONS checks in the routers actually trigger.
            "iteration": 1,
            # Commit the ledger entry when the cycle completed; consumed
            # params are dropped by setting pending_params back to None.
            **({"simulated_runs": [run_entry], "pending_params": None}
               if run_entry else {}),
        }

    def judge(state: AgentState):
        response = judge_model.invoke({
            "messages": [HumanMessage(content=(
                "Based on the evaluator's output, decide whether to accept or reject the instructions."
                f"```\n{state.get('evaluator_messages', [])[-1].text if state.get('evaluator_messages', []) else 'No output yet.'}\n```"
            ))]
        })

        update = {
            "decision": response.decision,
            "justification": response.justification,
            # Reset iteration count for the next generator-evaluator cycle.
            "iteration": Overwrite(value=0)
        }
        if response.decision == "reject":
            # Feed the judge's justification back into the evaluator's
            # message list so it can re-run its analysis with the feedback.
            update["evaluator_messages"] = state.get("evaluator_messages", []) + [
                HumanMessage(content=(
                    "The judge rejected your last instruction. "
                    f"Reason: {response.justification}\n"
                    "Re-run your analysis and produce a corrected instruction."
                ))
            ]
        return update

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
