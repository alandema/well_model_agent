import streamlit as st
import requests
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Well Model Agent", page_icon="🌊")


# API base URL - read from environment variable or default to localhost
API_BASE_URL = os.getenv("API_BASE_URL", "http://backend:8000")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Store thread_id for conversation memory
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

# Track pending HITL interrupt
if "pending_interrupt" not in st.session_state:
    st.session_state.pending_interrupt = None

st.title("Chat with Well Agent")


# ── Helper: send a message to the backend ──
def send_message(prompt: str, thread_id: str | None) -> dict | None:
    """POST /chat and return the parsed JSON, or None on error."""
    try:
        resp = requests.post(
            f"{API_BASE_URL}/chat",
            json={"message": prompt, "thread_id": thread_id},
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            st.error(f"API returned status {resp.status_code}: {resp.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to API: {e}")
        return None


# ── Helper: resume a HITL-paused graph ──
def resume_hitl(thread_id: str, action: str, args: dict | None = None, reason: str | None = None) -> dict | None:
    """POST /chat/resume and return the parsed JSON, or None on error."""
    payload = {"thread_id": thread_id, "action": action}
    if args is not None:
        payload["args"] = args
    if reason is not None:
        payload["reason"] = reason

    try:
        resp = requests.post(
            f"{API_BASE_URL}/chat/resume",
            json=payload,
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            st.error(f"Resume API returned status {resp.status_code}: {resp.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to resume API: {e}")
        return None


# ── HITL Approval UI (shown when pending_interrupt is set) ──
if st.session_state.pending_interrupt:
    interrupt = st.session_state.pending_interrupt
    st.warning(f"⚠️ **Human Approval Required** — Tool: `{interrupt.get('tool', 'unknown')}`")

    with st.container(border=True):
        st.markdown("### Tool Arguments")
        st.json(interrupt.get("args", {}))

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("✅ Approve", use_container_width=True):
                data = resume_hitl(st.session_state.thread_id, "approve")
                st.session_state.pending_interrupt = None
                if data:
                    if data.get("interrupt"):
                        # Another HITL tool fired
                        st.session_state.pending_interrupt = data["interrupt"]
                    if data.get("response"):
                        st.session_state.messages.append(
                            {"role": "assistant", "content": data["response"]}
                        )
                    st.rerun()

        with col2:
            if st.button("✏️ Edit", use_container_width=True):
                st.session_state._editing_hitl = True
                st.rerun()

        with col3:
            if st.button("❌ Reject", use_container_width=True):
                data = resume_hitl(st.session_state.thread_id, "reject", reason="User rejected")
                st.session_state.pending_interrupt = None
                if data:
                    if data.get("interrupt"):
                        st.session_state.pending_interrupt = data["interrupt"]
                    if data.get("response"):
                        st.session_state.messages.append(
                            {"role": "assistant", "content": data["response"]}
                        )
                    st.rerun()

    # Edit mode: let the user modify args before resuming
    if st.session_state.get("_editing_hitl"):
        st.markdown("#### Edit Arguments")
        args_str = st.text_area(
            "Modify the JSON below and click Submit:",
            value=json.dumps(interrupt.get("args", {}), indent=2),
            height=120,
        )
        if st.button("Submit Edited Args", use_container_width=True):
            try:
                edited_args = json.loads(args_str)
                data = resume_hitl(st.session_state.thread_id, "edit", args=edited_args)
                st.session_state.pending_interrupt = None
                st.session_state._editing_hitl = False
                if data:
                    if data.get("interrupt"):
                        st.session_state.pending_interrupt = data["interrupt"]
                    if data.get("response"):
                        st.session_state.messages.append(
                            {"role": "assistant", "content": data["response"]}
                        )
                    st.rerun()
            except json.JSONDecodeError:
                st.error("Invalid JSON. Please fix and try again.")

    st.stop()  # Don't render the normal chat UI while HITL is pending


# ── Normal Chat UI ──
# Scrollable chat box
with st.container(height=400, border=True):
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        data = send_message(prompt, st.session_state.thread_id)

    if data:
        st.session_state.thread_id = data["thread_id"]
        if data.get("interrupt"):
            # HITL tool needs approval — show the approval UI
            if data.get("response"):
                st.session_state.messages.append(
                    {"role": "assistant", "content": data["response"]}
                )
            st.session_state.pending_interrupt = data["interrupt"]
        else:
            st.session_state.messages.append(
                {"role": "assistant", "content": data["response"]}
            )

    st.rerun()