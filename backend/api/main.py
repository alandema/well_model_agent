from fastapi import FastAPI
from pydantic import BaseModel
from agent.graph import graph
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import uuid

load_dotenv()

app = FastAPI()


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    thread_id: str


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    thread_id = request.thread_id or str(uuid.uuid4())
    result = graph.invoke(
        {"messages": [HumanMessage(content=request.message)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    # Get the last message (AI response)
    ai_message = result["messages"][-1]
    return ChatResponse(response=ai_message.content, thread_id=thread_id)