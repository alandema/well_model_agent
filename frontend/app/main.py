import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
CHAT_URL = f"{API_BASE_URL}/chat"

st.title("Well Model Agent")

# Initialize chat history and thread id
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask the well model agent..."):
    # Add and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call the FastAPI /chat endpoint
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                payload = {
                    "message": prompt,
                    "thread_id": st.session_state.thread_id,
                }
                resp = requests.post(CHAT_URL, json=payload, timeout=120)
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as e:
                st.error(f"Request failed: {e}")
                st.stop()

        answer = data.get("response", "")
        st.session_state.thread_id = data.get("thread_id")
        st.markdown(answer)

        # Surface any human-in-the-loop interrupt for approval
        interrupt = data.get("interrupt")
        if interrupt:
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
            with st.expander("⚠️ Approval required", expanded=True):
                st.json(interrupt)
                st.info(
                    "This tool call needs approval. Use the FastAPI "
                    "`/resume` endpoint (or your approval UI) to approve, "
                    "edit, or reject it."
                )
            st.stop()

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})
