from operator import add
from typing import Annotated

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """State for the evaluator-optimizer production workflow."""

    run_id: Annotated[int, add] = 0
    iteration: Annotated[int, add] = 0
    generator_messages: list = []
    evaluator_messages: list = []
    decision: str = None
    justification: str = None
