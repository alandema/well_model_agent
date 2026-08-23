from operator import add
from typing import Annotated

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """State class for the agent, extending MessagesState."""

    run_id: Annotated[int, add] = 0