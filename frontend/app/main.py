import streamlit as st
import requests
import os
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

st.title("Chat with Well Agent")

# Scrollable chat box - all messages render inside this fixed-height container
with st.container(height=400, border=True):
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input (outside the box, stays fixed below)
if prompt := st.chat_input("Ask me anything..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get assistant response (API call with spinner below the chat box)
    with st.spinner("Thinking..."):
        try:
            response = requests.post(
                f"{API_BASE_URL}/chat",
                json={
                    "message": prompt,
                    "thread_id": st.session_state.thread_id,
                },
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                assistant_response = data["response"]
                st.session_state.thread_id = data["thread_id"]
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            else:
                error_msg = f"API returned status {response.status_code}: {response.text}"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
        except requests.exceptions.RequestException as e:
            error_msg = f"Failed to connect to API: {e}"
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

    # Rerun to re-render the chat box with new messages
    st.rerun()