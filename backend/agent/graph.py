import os
import sqlite3
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_openrouter.chat_models import ChatOpenRouter
from langgraph.checkpoint.sqlite import SqliteSaver


def call_model(state: MessagesState):
    """Single node that calls the Gemini model."""
    model = ChatOpenRouter(model="inclusionai/ling-3.0-flash:free")
    response = model.invoke(state["messages"])
    return {"messages": [response]}


# Build the graph
builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")
builder.add_edge("call_model", END)

# Create checkpoints directory if it doesn't exist
checkpoint_dir = os.path.join(os.path.dirname(__file__), "..", ".checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

# Initialize SQLite checkpointer
checkpoint_path = os.path.join(checkpoint_dir, "checkpoints.sqlite")
conn = sqlite3.connect(checkpoint_path, check_same_thread=False)
checkpointer = SqliteSaver(conn)

graph = builder.compile(checkpointer=checkpointer)