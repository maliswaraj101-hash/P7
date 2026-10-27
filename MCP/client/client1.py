import asyncio

from langchain_core.messages import ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

SERVERS = {
    "math": {
        "transport": "stdio",
        "command": "/snap/bin/uv",
        "args": [
            "run",
            "fastmcp",
            "run",
            "/home/swaraj/projects/MCP/server/main.py",
            # path for the server which we want to run
        ],
    }
}


async def main():
    client = MultiServerMCPClient(SERVERS)
    tools = await client.get_tools()
    # this will get u all the tools.

    print(tools)

    llm = ChatOpenAI()
    llm_with_tools = llm.bind_tools(tools)

    prompt = "hello"
    response = await llm_with_tools.ainvoke(prompt)
    print(response)

    selected_tool = response.tool_calls[0]["name"]
    selected_tool_args = response.tool_calls[0]["args"]
    selected_tool_id = response.tool_calls[0]["id"]

    # invoke this particular tool then to get the answer.
    tool_result = await selected_tool.ainvoke(selected_tool_args)
    print(tool_result)

    tool_message = ToolMessage(tool_call_id=selected_tool_id, content=tool_result)

    final_result = await llm_with_tools.ainvoke([prompt, response, tool_message])

    print(final_result)


if __name__ == "__main__":
    asyncio.run(main())
