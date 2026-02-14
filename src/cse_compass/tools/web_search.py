from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class WebSearchInput(BaseModel):
    """Input for web search."""

    query: str = Field(description="Search query for web search")


def get_web_search_tool():
    """Return a web search tool using Serper API. Requires SERPER_API_KEY in .env."""
    serper = GoogleSerperAPIWrapper()

    def _search(query: str) -> str:
        return serper.run(query)

    return StructuredTool.from_function(
        name="web_search",
        description="Search the web for the latest news and information. Use for recent market news, company updates, or general web search.",
        func=_search,
        args_schema=WebSearchInput,
    )
