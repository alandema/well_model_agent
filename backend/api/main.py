from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent.graph import graph
from langchain_core.messages import HumanMessage
from langgraph.types import Command
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
    interrupt: dict | None = None  # Populated when a HITL tool needs approval


class ResumeRequest(BaseModel):
    thread_id: str
    action: str  # "approve", "edit", or "reject"
    args: dict | None = None  # Edited args (only for "edit")
    reason: str | None = None  # Rejection reason (only for "reject")


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # v3 stream_events supports stream.messages + stream.interrupted / stream.interrupts
    stream = graph.stream_events(
        {"messages": [HumanMessage(content=request.message)]},
        config=config,
        version="v3",
    )

    last_message = ""
    for message in stream.messages:
        last_message += str(message.text)

    if stream.interrupted:
        interrupt_data = stream.interrupts[0].value
        return ChatResponse(
            response=last_message,
            thread_id=thread_id,
            interrupt=interrupt_data,
        )

    return ChatResponse(response=last_message, thread_id=thread_id)


@app.post("/chat/resume", response_model=ChatResponse)
async def resume_chat(request: ResumeRequest):
    """Resume a paused graph after a human approves/edits/rejects a HITL tool."""
    config = {"configurable": {"thread_id": request.thread_id}}

    if request.action == "approve":
        resume_value = {"action": "approve"}
    elif request.action == "edit":
        resume_value = {"action": "edit", "args": request.args or {}}
    elif request.action == "reject":
        resume_value = {"action": "reject", "reason": request.reason or "User rejected"}
    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {request.action}")

    stream = graph.stream_events(
        Command(resume=resume_value),
        config=config,
        version="v3",
    )

    last_message = ""
    for message in stream.messages:
        last_message += str(message.text)

    if stream.interrupted:
        interrupt_data = stream.interrupts[0].value
        return ChatResponse(
            response=last_message,
            thread_id=request.thread_id,
            interrupt=interrupt_data,
        )

    return ChatResponse(response=last_message, thread_id=request.thread_id)