import os
import certifi
from dotenv import load_dotenv

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from typing import TypedDict, Annotated
import operator
import uuid
import asyncio
import concurrent.futures

import psycopg
from psycopg.rows import dict_row

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from mcp_client import tavily_mcp_search, aviation_mcp_call, forecast_mcp_search, weather_mcp_search
from langchain_openai import ChatOpenAI
from logger import (
    log_system, log_agent_start, log_agent_done,
    log_llm_call, log_llm_done,
    log_mcp_call, log_mcp_done, log_mcp_error,
    log_state_read, log_state_write,
    log_result, log_error,
    log_divider, log_graph_start, log_graph_done,
)


# =========================
# Async helper
# =========================

def run_async(coro):
    """
    Run an async coroutine from any context — sync function, thread pool,
    or already-running event loop — without nest_asyncio side-effects.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is None:
        return asyncio.run(coro)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()


# =========================
# DB / Config
# =========================

def get_database_url():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL is missing.")
    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"
    return database_url


OPEN_API_KEY = os.getenv("OPEN_API_KEY")
if not OPEN_API_KEY:
    raise ValueError("OPEN_API_KEY is missing. Please add it to your .env file.")


# =========================
# LLM
# =========================

log_system("Initialising LLM (gpt-5-nano) …")

llm = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=OPEN_API_KEY,
)

log_system("LLM ready ✔")


# =========================
# State
# =========================

class TravelState(TypedDict):
    messages:        Annotated[list[AnyMessage], operator.add]
    user_query:      str
    flight_results:  str
    hotel_results:   str
    itinerary:       str
    llm_calls:       int
    weather_results: str


# =========================
# Prompts
# =========================

IATA_EXTRACTION_PROMPT = """
Extract the departure and arrival airport IATA codes from the user's travel query.

User Query: {query}

Respond with ONLY a JSON object in this exact format (no markdown, no explanation):
{{"departure_iata": "XXX", "arrival_iata": "YYY"}}

Rules:
- Use standard 3-letter IATA airport codes (e.g. JFK, LHR, DXB, SYD, LAX).
- If the query mentions a city, use the main international airport for that city.
- If you cannot determine one or both codes, use null for that field.
"""

FLIGHT_AGENT_PROMPT = """
You are a travel flight expert.

User Query:
{query}

Departure Airport IATA: {departure_iata}
Arrival Airport IATA: {arrival_iata}

Live Flight Schedule Data:
{schedule_data}

Generate:
1. Likely departure airport (full name)
2. Likely arrival airport (full name)
3. Airlines serving this route
4. Typical flight duration
5. Estimated airfare range (note if live pricing is unavailable)
6. Peak season pricing warning
7. Booking advice

Return concise, practical travel guidance.
"""


# =========================
# LLM helpers
# =========================

def extract_iata_codes(query: str) -> dict:
    """LLM call — extract departure/arrival IATA codes from query."""
    import json

    log_llm_call("extract IATA codes from query")
    response = llm.invoke([
        SystemMessage(content="You are an expert at identifying airport IATA codes."),
        HumanMessage(content=IATA_EXTRACTION_PROMPT.format(query=query))
    ])
    log_llm_done("IATA extraction", len(response.content))

    try:
        codes = json.loads(response.content.strip())
        return {
            "departure_iata": codes.get("departure_iata"),
            "arrival_iata":   codes.get("arrival_iata"),
        }
    except Exception as e:
        log_error("IATA JSON parse", str(e))
        return {"departure_iata": None, "arrival_iata": None}


def extract_destination(query: str) -> str:
    """LLM call — extract destination city/country name from query."""
    log_llm_call("extract destination city from query")
    response = llm.invoke([
        SystemMessage(content="You extract destination city or country names from travel queries. Return only the destination name, nothing else."),
        HumanMessage(content=f"Extract only the destination city or country from this query:\n{query}")
    ])
    dest = response.content.strip()
    log_llm_done("destination extraction", len(dest))
    log_result("Destination identified", dest)
    return dest


# =========================
# ✈️  Flight Agent
# =========================

def flight_agent(state: TravelState):
    log_agent_start("Flight Agent", state["user_query"])
    log_state_read("FlightAgent", "user_query")

    query = state["user_query"]

    try:
        # ── Step 1: LLM extracts IATA codes ──────────────────────────────
        log_divider("Step 1 · LLM → extract IATA codes")
        codes          = extract_iata_codes(query)
        departure_iata = codes.get("departure_iata")
        arrival_iata   = codes.get("arrival_iata")
        log_result("IATA codes", f"Departure={departure_iata}  Arrival={arrival_iata}")

        # ── Step 2: AviationStack MCP → live departure schedule ──────────
        log_divider("Step 2 · AviationStack MCP → flight schedule")
        schedule_data = ""

        if departure_iata:
            tool_args = {
                "airport_iata_code": departure_iata,
                "schedule_type":     "departure"
            }
            log_mcp_call("AviationStack", "flight_arrival_departure_schedule", tool_args)

            try:
                dep_schedule = run_async(
                    aviation_mcp_call(
                        "flight_arrival_departure_schedule",
                        tool_args
                    )
                )
                raw = str(dep_schedule)
                schedule_data += f"\nDepartures from {departure_iata}:\n{raw[:3000]}"
                log_mcp_done("AviationStack", "flight_arrival_departure_schedule", raw[:120])
            except Exception as e:
                log_mcp_error("AviationStack", "flight_arrival_departure_schedule", str(e))
                schedule_data += f"\nDeparture schedule unavailable for {departure_iata}: {e}"
        else:
            log_error("FlightAgent", "No departure IATA — skipping MCP call")

        if not schedule_data:
            schedule_data = "No live schedule data. Provide general guidance."

        # ── Step 3: LLM generates flight guidance ────────────────────────
        log_divider("Step 3 · LLM → generate flight guidance")
        prompt = FLIGHT_AGENT_PROMPT.format(
            query=query,
            departure_iata=departure_iata or "Unknown",
            arrival_iata=arrival_iata   or "Unknown",
            schedule_data=schedule_data,
        )
        log_llm_call("generate flight recommendations")
        response    = llm.invoke([
            SystemMessage(content="You are an expert travel flight planner."),
            HumanMessage(content=prompt)
        ])
        flight_data = response.content
        log_llm_done("flight recommendations", len(flight_data))

    except Exception as e:
        log_error("FlightAgent — unexpected failure", str(e))
        flight_data = f"Flight information unavailable: {str(e)}"

    log_state_write("FlightAgent", "flight_results", flight_data[:80])
    log_agent_done("Flight Agent", f"{len(flight_data)} chars written to state")

    return {
        "flight_results": flight_data,
        "messages":       [AIMessage(content="Flight recommendations generated")],
        "llm_calls":      state.get("llm_calls", 0) + 1,
    }


# =========================
# 🏨  Hotel Agent
# =========================

def hotel_agent(state: TravelState):
    log_agent_start("Hotel Agent", state["user_query"])
    log_state_read("HotelAgent", "user_query")

    query = f"Best hotels for {state['user_query']}"

    log_divider("Tavily MCP → hotel search")
    log_mcp_call("Tavily", "tavily_search", {"query": query[:80]})

    try:
        hotel_results = run_async(tavily_mcp_search(query))
        log_mcp_done("Tavily", "tavily_search", str(hotel_results)[:120])
        log_result("Hotel data received", str(hotel_results)[:80])
    except Exception as e:
        log_mcp_error("Tavily", "tavily_search", str(e))
        hotel_results = f"Hotel search failed: {e}"

    log_state_write("HotelAgent", "hotel_results", str(hotel_results)[:80])
    log_agent_done("Hotel Agent")

    return {
        "hotel_results": hotel_results,
        "messages":      [AIMessage(content="Hotel information fetched.")],
        "llm_calls":     state.get("llm_calls", 0) + 1,
    }


# =========================
# 🌤️  Weather Agent
# =========================

def weather_agent(state: TravelState):
    log_agent_start("Weather Agent", state["user_query"])
    log_state_read("WeatherAgent", "user_query")

    # ── Step 1: LLM extracts destination city ─────────────────────────────
    log_divider("Step 1 · LLM → extract destination city")
    city = extract_destination(state["user_query"])

    # ── Step 2: Custom Weather MCP → current conditions ───────────────────
    log_divider("Step 2 · Weather MCP → current conditions")
    log_mcp_call("WeatherMCP", "get_current_weather", {"city": city})
    try:
        weather_data = run_async(weather_mcp_search(city))
        log_mcp_done("WeatherMCP", "get_current_weather", str(weather_data)[:120])
        log_result("Current weather", str(weather_data)[:80])
    except Exception as e:
        log_mcp_error("WeatherMCP", "get_current_weather", str(e))
        weather_data = f"Weather unavailable: {e}"

    # ── Step 3: Custom Weather MCP → forecast ─────────────────────────────
    log_divider("Step 3 · Weather MCP → 5-step forecast")
    log_mcp_call("WeatherMCP", "get_forecast", {"city": city})
    try:
        forecast_data = run_async(forecast_mcp_search(city))
        log_mcp_done("WeatherMCP", "get_forecast", str(forecast_data)[:120])
        log_result("Forecast data", str(forecast_data)[:80])
    except Exception as e:
        log_mcp_error("WeatherMCP", "get_forecast", str(e))
        forecast_data = f"Forecast unavailable: {e}"

    combined = f"Current Weather:\n{weather_data}\n\nForecast:\n{forecast_data}"

    log_state_write("WeatherAgent", "weather_results", str(weather_data)[:80])
    log_agent_done("Weather Agent")

    return {
        "weather_results": combined,
        "messages":        [AIMessage(content="Weather information fetched")],
    }


# =========================
# 🗓️  Itinerary Agent
# =========================

def itinerary_agent(state: TravelState):
    log_agent_start("Itinerary Agent", state["user_query"])

    log_state_read("ItineraryAgent", "flight_results")
    log_state_read("ItineraryAgent", "hotel_results")
    log_state_read("ItineraryAgent", "weather_results")

    prompt = f"""
Create a complete travel itinerary.

User Query:
{state['user_query']}

Flight Results:
{state['flight_results']}

Hotel Results:
{state['hotel_results']}

Weather Results:
{state['weather_results']}

Make the itinerary practical, budget-aware, and easy to follow.
"""

    log_divider("LLM → generate day-by-day itinerary")
    log_llm_call("generate day-by-day itinerary")
    response = llm.invoke([
        SystemMessage(content="You are an expert travel planner."),
        HumanMessage(content=prompt)
    ])
    log_llm_done("itinerary generation", len(response.content))

    log_state_write("ItineraryAgent", "itinerary", response.content[:80])
    log_agent_done("Itinerary Agent", f"{len(response.content)} chars")

    return {
        "itinerary": response.content,
        "messages":  [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# 🏆  Final Agent
# =========================

def final_agent(state: TravelState):
    log_agent_start("Final Agent", "formatting complete travel response")

    log_state_read("FinalAgent", "flight_results")
    log_state_read("FinalAgent", "hotel_results")
    log_state_read("FinalAgent", "weather_results")
    log_state_read("FinalAgent", "itinerary")

    final_prompt = f"""
Generate the final travel response for the user.

User Request:
{state['user_query']}

Flights:
{state['flight_results']}

Hotels:
{state['hotel_results']}

Weather:
{state['weather_results']}

Itinerary:
{state['itinerary']}

Format the final answer beautifully using these sections:
1. Trip Summary
2. Flight Information
3. Hotel Suggestions
4. Weather Information
5. Day-by-Day Itinerary
6. Estimated Budget
7. Final Recommendations

Important:
- Be clear and practical.
- Mention that live flight API may not provide ticket prices if pricing is unavailable.
- Include weather-based travel advice.
- Keep the response useful for real travel planning.
"""

    log_divider("LLM → format final travel response")
    log_llm_call("format complete travel response")
    response = llm.invoke([
        SystemMessage(content="You are a professional AI travel booking assistant."),
        HumanMessage(content=final_prompt)
    ])
    log_llm_done("final response formatting", len(response.content))
    log_result("Final response ready", f"{len(response.content)} chars total")

    log_agent_done("Final Agent", "all sections assembled ✈️ 🏨 🌤️ 🗓️")

    return {
        "messages":  [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Build Graph
# =========================

log_system("Building LangGraph StateGraph …")

graph = StateGraph(TravelState)

graph.add_node("flight_agent",    flight_agent)
graph.add_node("hotel_agent",     hotel_agent)
graph.add_node("weather_agent",   weather_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent",     final_agent)

graph.add_edge(START,             "flight_agent")
graph.add_edge("flight_agent",    "hotel_agent")
graph.add_edge("hotel_agent",     "weather_agent")
graph.add_edge("weather_agent",   "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent",      END)

log_system("Graph edges: START → flight → hotel → weather → itinerary → final → END")


# =========================
# PostgreSQL Checkpointer
# =========================

log_system("Connecting to PostgreSQL (AWS RDS) …")

DATABASE_URL = get_database_url()

_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    row_factory=dict_row
)

checkpointer = PostgresSaver(_conn)
checkpointer.setup()

log_system("PostgreSQL checkpointer ready ✔")

travel_graph = graph.compile(checkpointer=checkpointer)

log_system("LangGraph compiled — TripWise 360 is ready 🌍\n")


# =========================
# Entry point for FastAPI
# =========================

def run_travel_agent(user_input: str, thread_id: str | None = None):
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    log_graph_start(user_input, thread_id)

    config = {"configurable": {"thread_id": thread_id}}

    result = travel_graph.invoke(
        {
            "messages":        [HumanMessage(content=user_input)],
            "user_query":      user_input,
            "flight_results":  "",
            "hotel_results":   "",
            "weather_results": "",
            "itinerary":       "",
            "llm_calls":       0,
        },
        config=config,
    )

    llm_calls    = result.get("llm_calls", 0)
    final_answer = result["messages"][-1].content

    log_graph_done(thread_id, llm_calls)

    return {
        "thread_id":      thread_id,
        "answer":         final_answer,
        "flight_results": result.get("flight_results", ""),
        "hotel_results":  result.get("hotel_results", ""),
        "weather_results":result.get("weather_results", ""),
        "itinerary":      result.get("itinerary", ""),
        "llm_calls":      llm_calls,
    }
