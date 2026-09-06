from operator import add
from typing import Annotated

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """State for the evaluator-optimizer production workflow."""

    run_id: Annotated[int, add] = 0
    iteration: Annotated[int, add] = 0
    generator_messages: list = []
    evaluator_messages: list = []
    # Parsed final answer of the evaluator (dict form of EvaluatorOutput):
    # {operational_state, generator_instructions, justification}. Consumed
    # by the Generator (as feedback) and the Judge (for review), then reset.
    evaluator_output: dict = None
    decision: str = None
    justification: str = None
    # Operating point of the most recent FOWM run, extracted from the tool
    # response by generator_tools. Plain field (last-write-wins): it always
    # holds the LATEST simulated params and is consumed (set to None) by the
    # evaluator once logged into simulated_runs.
    pending_params: dict = None
    # Minimal run ledger: one entry {params, operational_state} per completed
    # generator->evaluator cycle. Accumulated with add so the evaluator can
    # see what was already simulated without any stale tool traffic.
    simulated_runs: Annotated[list, add] = []
