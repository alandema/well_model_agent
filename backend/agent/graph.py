from langgraph.graph import StateGraph, MessagesState,  START, END
from langchain_google_genai import ChatGoogleGenerativeAI


def call_model(state: MessagesState):
    """Single node that calls the Gemini model."""
    model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
    response = model.invoke(state["messages"])
    return {"messages": [response]}


# Build the graph
builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")
builder.add_edge("call_model", END)

graph = builder.compile()