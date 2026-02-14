import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({
    "cse": {
        "transport": "stdio",
        "command": "npx",
        "args": ["cse-mcp"]
    }
})


async def get_cse_tools_async():
    return await client.get_tools()


def get_cse_tools():
    """Sync wrapper for use in LangChain/LangGraph."""
    return asyncio.run(get_cse_tools_async())
