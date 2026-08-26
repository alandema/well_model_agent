from operator import add
from typing import Annotated, NotRequired

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """State for the evaluator-optimizer production workflow."""

    run_id: Annotated[int, add] = 0
    iteration: Annotated[int, add] = 0
    parameters: NotRequired[dict] = None
    csv_path: NotRequired[str] = None
    instructions: NotRequired[str] = None
    decision: NotRequired[str] = None
    justification: NotRequired[str] = None
