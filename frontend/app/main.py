import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Well Model Agent", page_icon="🌊")

st.title("Well Model Agent")
st.write("Simple frontend for the FastAPI backend")

# API base URL - read from environment variable or default to localhost
API_BASE_URL = os.getenv("API_BASE_URL", "http://backend:8000")

# Health check
if st.button("Check API Health"):
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            st.success(f"API is healthy: {response.json()}")
        else:
            st.error(f"API returned status {response.status_code}")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to API: {e}")

# Root endpoint
if st.button("Call Root Endpoint"):
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        if response.status_code == 200:
            st.success(f"Response: {response.json()}")
        else:
            st.error(f"API returned status {response.status_code}")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to API: {e}")

# Chat interface
st.divider()
st.subheader("Chat with Gemini")

user_message = st.text_input("Your message:", placeholder="Ask me anything...")

if st.button("Send", type="primary") and user_message:
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={"message": user_message},
            timeout=30
        )
        if response.status_code == 200:
            st.success("Response:")
            st.write(response.json()["response"])
        else:
            st.error(f"API returned status {response.status_code}: {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to API: {e}")