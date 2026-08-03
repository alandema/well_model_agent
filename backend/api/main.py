from fastapi import FastAPI
from pydantic import BaseModel
from agent.graph import graph
from langchain_core.messages import HumanMessage

app = FastAPI()


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    result = graph.invoke({"messages": [HumanMessage(content=request.message)]})
    # Get the last message (AI response)
    ai_message = result["messages"][-1]
    return ChatResponse(response=ai_message.content)