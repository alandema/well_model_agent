from operator import add
from typing import Annotated, NotRequired

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """State for the evaluator-optimizer production workflow."""

    run_id: Annotated[int, add] = 0
    iteration: Annotated[int, add]= 0
    plan: NotRequired[str]
    feedback: NotRequired[str]
    evaluation: NotRequired[dict]