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


# =========================
# Async helper
# =========================

def run_async(coro):
    """
    Run an async coroutine from any context — sync function, thread pool,
    or already-running event loop — without nest_asyncio side-effects.

    Strategy:
    - If there is NO running loop in this thread, just use asyncio.run().
    - If there IS a running loop (FastAPI / uvicorn thread), submit the
      coroutine to that loop from a fresh daemon thread so we never try
      to nest loops.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is None:
        # Plain sync context — safe to call asyncio.run() directly.
        return asyncio.run(coro)
    else:
        # We are inside an already-running loop (e.g. a LangGraph node
        # executed in uvicorn's thread pool).  Run the coroutine in a
        # brand-new event loop on a dedicated daemon thread so we never
        # block the main loop and never nest loops.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()


def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. Please add your Render PostgreSQL External Database URL to .env"
        )

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

llm = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=OPEN_API_KEY,
    # base_url="...",
    # organization="...",
    # other params...
)


# =========================
# State
# =========================

class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int
    weather_results: str

# =========================
# Flight Agent
# =========================

# def flight_agent(state: TravelState):
#     query = state["user_query"]
#     flight_data = search_flights(query)

#     return {
#         "flight_results": flight_data,
#         "messages": [
#             AIMessage(content="Flight results fetched.")
#         ],
#         "llm_calls": state.get("llm_calls", 0) + 1
#     }




# Prompt to extract IATA codes from user query
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

# Prompt to generate flight guidance from live schedule data
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


def extract_iata_codes(query: str) -> dict:
    """Use LLM to extract departure and arrival IATA codes from query."""
    import json

    response = llm.invoke([
        SystemMessage(content="You are an expert at identifying airport IATA codes."),
        HumanMessage(content=IATA_EXTRACTION_PROMPT.format(query=query))
    ])

    try:
        codes = json.loads(response.content.strip())
        return {
            "departure_iata": codes.get("departure_iata"),
            "arrival_iata": codes.get("arrival_iata"),
        }
    except Exception:
        return {"departure_iata": None, "arrival_iata": None}


def extract_destination(query: str) -> str:
    """Use LLM to extract the destination city/country from a travel query."""
    response = llm.invoke([
        SystemMessage(content="You extract destination city or country names from travel queries. Return only the destination name, nothing else."),
        HumanMessage(content=f"Extract only the destination city or country from this query:\n{query}")
    ])
    return response.content.strip()


# Flight Agent
def flight_agent(state: TravelState):
    print("\nINSIDE FLIGHT AGENT\n")

    query = state["user_query"]

    try:
        # Step 1: Extract IATA codes from the query via LLM
        codes = extract_iata_codes(query)
        departure_iata = codes.get("departure_iata")
        arrival_iata = codes.get("arrival_iata")

        print(f"\nExtracted IATA — Departure: {departure_iata}, Arrival: {arrival_iata}\n")

        # Step 2: Fetch live departure schedule (one call to stay within free-tier rate limits)
        schedule_data = ""

        if departure_iata:
            try:
                dep_schedule = run_async(
                    aviation_mcp_call(
                        "flight_arrival_departure_schedule",
                        {"airport_iata_code": departure_iata, "schedule_type": "departure"}
                    )
                )
                schedule_data += f"\nDepartures from {departure_iata}:\n{str(dep_schedule)[:3000]}"
                print("\nDEPARTURE SCHEDULE:", dep_schedule)
            except Exception as e:
                schedule_data += f"\nDeparture schedule unavailable for {departure_iata}: {e}"

        if not schedule_data:
            schedule_data = "No live schedule data available. Provide general guidance based on typical routes."

        # Step 3: Generate flight guidance with LLM
        prompt = FLIGHT_AGENT_PROMPT.format(
            query=query,
            departure_iata=departure_iata or "Unknown",
            arrival_iata=arrival_iata or "Unknown",
            schedule_data=schedule_data,
        )

        response = llm.invoke([
            SystemMessage(content="You are an expert travel flight planner."),
            HumanMessage(content=prompt)
        ])

        flight_data = response.content

    except Exception as e:
        flight_data = f"Flight information unavailable: {str(e)}"

    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(content="Flight recommendations generated")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# =========================
# Hotel Agent
# =========================

def hotel_agent(state: TravelState):
    query = f"Best hotels for {state['user_query']}"
    hotel_results = run_async(tavily_mcp_search(query))
    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(content="Hotel information fetched.")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

# =========================
# Weather Agent
# =========================

def weather_agent(state: TravelState):

    city = extract_destination(state["user_query"])

    weather_data = run_async(
        weather_mcp_search(city)
    )

    forecast_data = run_async(
        forecast_mcp_search(city)
    )

    return {
        "weather_results": f"""
        Current Weather:
        {weather_data}

        Forecast:
        {forecast_data}
        """,
        "messages": [
            AIMessage(
                content="Weather information fetched"
            )
        ]
    }

# =========================
# Itinerary Agent
# =========================

def itinerary_agent(state: TravelState):
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

    response = llm.invoke([
        SystemMessage(content="You are an expert travel planner."),
        HumanMessage(content=prompt)
    ])

    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


def final_agent(state: TravelState):
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

    response = llm.invoke([
        SystemMessage(content="You are a professional AI travel booking assistant."),
        HumanMessage(content=final_prompt)
    ])

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# =========================
# Build Graph
# =========================

graph = StateGraph(TravelState)

graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "weather_agent")
graph.add_edge("weather_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)


# =========================
# PostgreSQL Checkpointer
# =========================
DATABASE_URL = get_database_url()

_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    row_factory=dict_row
)

checkpointer = PostgresSaver(_conn)
checkpointer.setup()

travel_graph = graph.compile(checkpointer=checkpointer)



# Function for FastAPI
# =========================

def run_travel_agent(user_input: str, thread_id: str | None = None):
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = travel_graph.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",
            "llm_calls": 0
        },
        config=config
    )

    final_answer = result["messages"][-1].content

    return {
        "thread_id": thread_id,
        "answer": final_answer,
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "weather_results": result.get("weather_results", ""),
        "itinerary": result.get("itinerary", ""),
        "llm_calls": result.get("llm_calls", 0),
    }


