import os

from langchain.tools import tool
from tavily import TavilyClient

print("TAVILY_API_KEY:", os.getenv("TAVILY_API_KEY", "Not Found"),
      flush=True)  # Debugging line to check if the API key is loaded
if "TAVILY_API_KEY" in os.environ:
    tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


@tool(
    description="Search the web for relevant information using Tavily's search engine.",
)
def web_search(query: str) -> str:
    if "TAVILY_API_KEY" not in os.environ:
        return "Tool unavailable. Tavily API key not found. Please set the TAVILY_API_KEY environment variable."
    return tavily_client.search(query)
