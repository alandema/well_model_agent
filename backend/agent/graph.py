from langgraph.graph import StateGraph, MessagesState,  START, END
from langchain_openrouter.chat_models import ChatOpenRouter


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

graph = builder.compile()