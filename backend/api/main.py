import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from agent.graph import create_graph
from agent.services.config import load_config
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from dotenv import load_dotenv
import uuid

this_file_path = os.path.abspath(__file__)
load_dotenv()

app = FastAPI()

config = load_config(os.path.join(
    os.path.dirname(this_file_path), "..", "agent", "prompts", "config.json"))

graph = create_graph(
    llm_model_config=config)


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

    stream = graph.stream_events(
        {"messages": [HumanMessage(content=request.message)]},
        config=config,
        version="v3",
    )

    last_message = ""
    for message in list(stream.messages):
        last_message += str(message.text)

    if stream.interrupted:
        interrupt_data = stream.interrupts[0].value
        return ChatResponse(
            response=last_message,
            thread_id=thread_id,
            interrupt=interrupt_data,
        )

    return ChatResponse(response=last_message, thread_id=thread_id)


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream the assistant response token-by-token using LangGraph event
    streaming (stream_events version="v3", stream.messages projection).

    The response is plain text streamed as it is generated. After the text
    finishes, a final line prefixed with `__META__:` carries the thread_id
    and (optionally) an interrupt payload as JSON, so the frontend can keep
    conversation memory and handle human-in-the-loop approvals.
    """
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    def token_generator():
        stream = graph.stream_events(
            {"messages": [HumanMessage(content=request.message)]},
            config=config
        )

        for message in stream.messages:
            for token in message.text:
                yield token

        # After the run finishes (or pauses), send metadata the frontend needs.
        import json
        meta = {"thread_id": thread_id}
        if stream.interrupted:
            meta["interrupt"] = stream.interrupts[0].value
        yield f"\n__META__:{json.dumps(meta)}"

    return StreamingResponse(token_generator(), media_type="text/plain")


async def resume_chat(request: ResumeRequest):
    """Resume a paused graph after a human approves/edits/rejects a HITL tool."""
    config = {"configurable": {"thread_id": request.thread_id}}

    if request.action == "approve":
        resume_value = {"action": "approve"}
    elif request.action == "edit":
        resume_value = {"action": "edit", "args": request.args or {}}
    elif request.action == "reject":
        resume_value = {"action": "reject",
                        "reason": request.reason or "User rejected"}
    else:
        raise HTTPException(
            status_code=400, detail=f"Unknown action: {request.action}")

    stream = graph.stream_events(
        Command(resume=resume_value),
        config=config,
        version="v3",
    )

    last_message = ""
    for message in list(stream.messages):
        last_message += str(message.text)

    if stream.interrupted:
        interrupt_data = stream.interrupts[0].value
        return ChatResponse(
            response=last_message,
            thread_id=request.thread_id,
            interrupt=interrupt_data,
        )

    return ChatResponse(response=last_message, thread_id=request.thread_id)
