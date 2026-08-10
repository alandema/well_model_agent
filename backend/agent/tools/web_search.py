import os

from langchain.tools import tool
from tavily import TavilyClient
from pydantic import BaseModel, Field

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
tavily_client = TavilyClient(
    api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None


def is_web_search_available() -> bool:
    """Return whether the Tavily-backed tool can be exposed to the agent."""
    return tavily_client is not None


class WebSearchInput(BaseModel):
    query: str = Field(...,
                       description="The search query to be sent to the Tavily API.")


@tool(
    args_schema=WebSearchInput,
    description="Use this tool only if it is strictly necessary to attend the user's query. It performs a web search using the Tavily API and returns the results as a string.",
)
def web_search(query: str) -> str:
    if tavily_client is None:
        return "Tool unavailable. Tavily API key not found. Please set the TAVILY_API_KEY environment variable."
    return str(tavily_client.search(query))
