import os
import sys
import asyncio
import certifi
from pathlib import Path
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from logger import log_system, log_mcp_call, log_mcp_done, log_mcp_error, log_result, log_error

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
AVIATIONSTACK_API_KEY = os.getenv("AVIATIONSTACK_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Automatically find the current project folder.
PROJECT_DIR = Path(__file__).resolve().parent
WEATHER_SERVER_PATH = PROJECT_DIR / "custom_weather_mcp_server.py"

# Preserve the full environment when starting local stdio MCP servers.
AVIATION_ENV = os.environ.copy()
AVIATION_ENV["AVIATION_STACK_API_KEY"] = AVIATIONSTACK_API_KEY or ""

WEATHER_ENV = os.environ.copy()
WEATHER_ENV["OPENWEATHER_API_KEY"] = OPENWEATHER_API_KEY or ""


client = MultiServerMCPClient(
    {
       "tavily": {
            "transport": "streamable_http",
            "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}"
        },
  
        "aviationstack": {
            "transport": "stdio",
            "command": "uvx",
            "args": [
                "aviationstack-mcp"
            ],
            "env": {
                "AVIATION_STACK_API_KEY": AVIATIONSTACK_API_KEY
            }
        },
         "weather": {
            "transport": "stdio",

            # Use the same Python environment that runs app.py.
            "command": sys.executable,

            # Automatically use custom_weather_mcp_server.py
            # from the current project directory.
            "args": [
                str(WEATHER_SERVER_PATH)
            ],

            "env": WEATHER_ENV
        }
        
    }
)


# ==========================================
# Diagnostic function
# ==========================================

async def get_all_tools():
    """
    Load each MCP server separately.

    A broken server will no longer prevent the other
    working servers from loading.
    """

    all_tools = []

    for server_name in (
        "tavily",
        "aviationstack",
        "weather"
    ):
        try:
            tools = await client.get_tools(
                server_name=server_name
            )

            all_tools.extend(tools)

            print(
                f"\nAvailable tools from "
                f"{server_name} MCP:\n"
            )

            for tool in tools:
                print(tool.name)

        except Exception as error:
            print(
                f"\nCould not connect to "
                f"{server_name} MCP:\n{error}\n"
            )

    return all_tools


# ==========================================
# Tavily MCP tool
# ==========================================

search_tool = None


async def initialize_mcp():
    global search_tool
    if search_tool is not None:
        return

    log_system("Connecting to Tavily MCP (remote HTTP) …")
    tools = await client.get_tools(server_name="tavily")

    tools_by_name = {tool.name: tool for tool in tools}
    search_tool   = tools_by_name.get("tavily_search")

    if search_tool is None:
        available = ", ".join(tools_by_name.keys())
        raise RuntimeError(
            f"Tavily MCP connected but 'tavily_search' not found. Available: {available or 'none'}"
        )

    log_result("Tavily MCP ready", f"tools: {list(tools_by_name.keys())}")


async def tavily_mcp_search(query: str):
    await initialize_mcp()
    result = await search_tool.ainvoke({"query": query})
    return result


# ==========================================
# AviationStack MCP tools
# ==========================================

aviation_tools = {}


async def initialize_aviation_tools():
    global aviation_tools
    if aviation_tools:
        return

    log_system("Connecting to AviationStack MCP (stdio/uvx) …")
    tools = await client.get_tools(server_name="aviationstack")
    aviation_tools = {tool.name: tool for tool in tools}

    if not aviation_tools:
        raise RuntimeError("AviationStack MCP connected but returned no tools.")

    log_result("AviationStack MCP ready", f"tools: {list(aviation_tools.keys())}")


async def aviation_mcp_call(tool_name: str, tool_args: dict = None):
    await initialize_aviation_tools()

    tool = aviation_tools.get(tool_name)
    if tool is None:
        available = ", ".join(sorted(aviation_tools.keys()))
        raise ValueError(
            f"AviationStack tool '{tool_name}' not found. Available: {available or 'none'}"
        )

    result = await tool.ainvoke(tool_args or {})
    return result

# ==========================================
# Weather MCP tools
# ==========================================

weather_tool = None
forecast_tool = None


async def initialize_weather_tools():
    global weather_tool, forecast_tool

    if weather_tool is not None and forecast_tool is not None:
        return

    if not WEATHER_SERVER_PATH.exists():
        raise FileNotFoundError(f"Weather MCP server not found: {WEATHER_SERVER_PATH}")

    log_system(f"Connecting to Custom Weather MCP (stdio) → {WEATHER_SERVER_PATH.name} …")
    tools = await client.get_tools(server_name="weather")
    tools_by_name = {tool.name: tool for tool in tools}

    weather_tool  = tools_by_name.get("get_current_weather")
    forecast_tool = tools_by_name.get("get_forecast")

    missing = [n for n, t in [("get_current_weather", weather_tool), ("get_forecast", forecast_tool)] if t is None]
    if missing:
        raise RuntimeError(
            f"Missing Weather MCP tools: {missing}. Available: {list(tools_by_name.keys())}"
        )

    log_result("Custom Weather MCP ready", f"tools: {list(tools_by_name.keys())}")


async def weather_mcp_search(city: str):
    await initialize_weather_tools()
    result = await weather_tool.ainvoke({"city": city})
    return result


async def forecast_mcp_search(city: str):
    await initialize_weather_tools()
    result = await forecast_tool.ainvoke({"city": city})
    return result


