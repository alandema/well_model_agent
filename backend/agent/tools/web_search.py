import os

from langchain.tools import tool
from tavily import TavilyClient
from pydantic import BaseModel, Field

print("TAVILY_API_KEY:", os.getenv("TAVILY_API_KEY", "Not Found"),
      flush=True)  # Debugging line to check if the API key is loaded
if "TAVILY_API_KEY" in os.environ:
    tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


class WebSearchInput(BaseModel):
    query: str = Field(...,
                       description="The search query to be sent to the Tavily API.")


@tool(
    args_schema=WebSearchInput,
    description="Use this tool only if it is strictly necessary to attend the user's query. It performs a web search using the Tavily API and returns the results as a string.",
)
def web_search(query: str) -> str:
    if "TAVILY_API_KEY" not in os.environ:
        return "Tool unavailable. Tavily API key not found. Please set the TAVILY_API_KEY environment variable."
    return tavily_client.search(query)
