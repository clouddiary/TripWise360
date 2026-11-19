"""Travel-planning graph and its entry points for FastAPI.

Read in this order: TravelState → supervisor_agent → specialist agents →
human_approval_agent/revision_agent → graph edges → HTTP-facing helpers.
Nodes return partial state updates; LangGraph merges and checkpoints them.
Importing this module initializes the model client and PostgreSQL checkpointer.
"""
import os
import json
import certifi
from dotenv import load_dotenv

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from workflow_rules import GuardrailDecision, HumanReview, RevisedRequest
from typing import Any, TypedDict, Annotated
import operator
import uuid
import asyncio
import concurrent.futures

import psycopg
from psycopg.rows import dict_row

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from mcp_client import (
    tavily_mcp_search,
    aviation_mcp_call,
    forecast_mcp_search,
    weather_mcp_search
)
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
    """Require a database URL and default to TLS unless sslmode is explicit."""
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
class TravelState(TypedDict, total=False):
    # Reducer: append returned messages instead of replacing the whole history.
    """Shared graph data; total=False permits partial updates from each node."""
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    # thread_id lives in graph configuration; run_id identifies one UI request.
    run_id: str

    # Supervisor + guardrail state
    guardrail_allowed: bool
    guardrail_reason: str
    selected_agents: list[str]
    trip_constraints: dict[str, Any]
    supervisor_reasoning: str

    # Original specialist results
    flight_results: str
    hotel_results: str
    weather_results: str
    itinerary: str

    # New budget + HITL state
    budget_results: str
    approval_request: str
    approved: bool
    human_feedback: str
    final_response: str
    revision_count: int
    revision_error: str

    llm_calls: int


# =========================
# Shared helpers
# =========================
KNOWN_AGENTS = {
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
}

# Selection is dynamic, but execution follows this fixed order (no parallelism).
AGENT_ORDER = [
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
]


def log_available_state(agent: str, state: TravelState, fields):
    """Log only populated inputs; empty research fields belong to skipped agents."""
    for field in fields:
        if state.get(field) not in (None, "", [], {}):
            log_state_read(agent, field)


def _llm_text(system_prompt: str, user_prompt: str) -> str:
    """Call the shared model and return its text; this does not update graph state."""
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    return str(response.content)


def _json_from_llm(text: str) -> dict[str, Any]:
    """Extract the first complete JSON object returned by the model."""
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError("The model did not return a JSON object.")

    # This tolerant parser is only for supervisor planning. Guardrail decisions
    # use strict model_validate_json() and must not accept surrounding prose.
    return json.loads(text[start : end + 1])


def _empty_constraints() -> dict[str, Any]:
    """Create a fresh constraints dictionary for a new or revised research run."""
    return {
        "destination": "",
        "origin": "",
        "duration": "",
        "budget": "",
        "travel_style": "",
        "special_preferences": [],
    }


# =========================
# Supervisor Agent + Input Guardrail
# =========================
def supervisor_agent(state: TravelState):
    """Validate input first, then select specialists and extract trip requirements."""
    log_agent_start("Supervisor", state["user_query"])
    log_state_read("Supervisor", "user_query")
    query = state["user_query"]
    llm_calls = state.get("llm_calls", 0)

    guardrail_prompt = f"""
Determine whether the following request belongs to travel planning or travel
information. Valid requests can include destinations, flights, hotels, weather,
budgets, visas, transportation, sightseeing, food, packing, or itineraries.

Block clearly unrelated requests and requests asking for harmful or illegal
instructions. Do not block a valid travel request merely because some details
are missing.

Return strict JSON only:
{{
  "allowed": true,
  "reason": ""
}}

User request:
{query}
"""

    # A malformed or unavailable guardrail must never authorize research.
    try:
        log_llm_call("validate travel input guardrail")
        guardrail_raw = _llm_text(
            "You are the input guardrail for a travel-planning application. "
            "Return strict JSON only.",
            guardrail_prompt,
        )
        log_llm_done("input guardrail", len(guardrail_raw))
        # Never use bool(raw_value): bool("false") is True in Python. The schema
        # requires an actual JSON Boolean and rejects missing/extra fields.
        decision = GuardrailDecision.model_validate_json(guardrail_raw)
        allowed = decision.allowed
        guardrail_reason = decision.reason.strip()
        llm_calls += 1
    except Exception as exc:
        log_error("Input guardrail fallback", str(exc))
        allowed = False
        guardrail_reason = "Request validation could not be completed. Please retry your request."

    log_result("Guardrail decision", "Allowed" if allowed else "Blocked")
    log_state_write("Supervisor", "guardrail_allowed", str(allowed))
    if not allowed:
        reason = guardrail_reason or (
            "TripMate AI can only help with travel-planning requests. "
            "Please ask about a destination, flight, hotel, weather, budget, "
            "or itinerary."
        )
        log_state_write("Supervisor", "guardrail_reason", reason)
        log_agent_done("Supervisor", "request blocked; specialist agents skipped")
        return {
            "guardrail_allowed": False,
            "guardrail_reason": reason,
            "selected_agents": [],
            "trip_constraints": _empty_constraints(),
            "supervisor_reasoning": reason,
            "final_response": reason,
            "messages": [AIMessage(content=f"Guardrail blocked request: {reason}")],
            "llm_calls": llm_calls,
        }

    supervisor_prompt = f"""
You are the supervisor of a multi-agent travel-planning system.
Choose only the specialist agents needed for the request.
Do not select weather_agent for a general trip, road trip, or packing request
unless the user explicitly asks about weather, climate, temperature, or forecasts.

Available agents:
- flight_agent: flights, airports, airlines, routes, airfare, or booking advice
- hotel_agent: hotels, accommodation, neighborhoods, or places to stay
- weather_agent: weather or forecast research, only when explicitly requested
- budget_agent: cost, affordability, price limits, or budget feasibility
- itinerary_agent: creates the integrated travel plan and must always be included

Return strict JSON only using this schema:
{{
  "selected_agents": ["flight_agent", "hotel_agent", "weather_agent", "budget_agent", "itinerary_agent"],
  "trip_constraints": {{
    "destination": "",
    "origin": "",
    "duration": "",
    "budget": "",
    "travel_style": "",
    "special_preferences": []
  }},
  "reasoning": ""
}}

User request:
{query}
"""

    try:
        log_llm_call("select specialists and extract trip constraints")
        supervisor_raw = _llm_text(
            "You route work to travel specialist agents. Return strict JSON only.",
            supervisor_prompt,
        )
        log_llm_done("supervisor planning", len(supervisor_raw))
        parsed = _json_from_llm(supervisor_raw)
        requested_agents = parsed.get("selected_agents", [])
        selected_agents = [
            name for name in AGENT_ORDER
            if name in requested_agents and name in KNOWN_AGENTS
        ]

        # The itinerary agent integrates whichever specialist results were selected.
        if "itinerary_agent" not in selected_agents:
            selected_agents.append("itinerary_agent")

        constraints = _empty_constraints()
        parsed_constraints = parsed.get("trip_constraints", {})
        if isinstance(parsed_constraints, dict):
            constraints.update(parsed_constraints)

        reasoning = str(parsed.get("reasoning", "")).strip()
        llm_calls += 1
    except Exception as exc:
        log_error("Supervisor planning fallback", str(exc))
        # Original workflow behavior is preserved as the fallback.
        selected_agents = [name for name in AGENT_ORDER if name != "weather_agent"]
        constraints = _empty_constraints()
        reasoning = (
            "Supervisor parsing failed, so the original full travel workflow "
            "was selected as a safe fallback."
        )

    log_result("Selected agents", ", ".join(selected_agents))
    log_state_write("Supervisor", "selected_agents", ", ".join(selected_agents))
    log_state_write("Supervisor", "trip_constraints", str(constraints))
    log_state_write("Supervisor", "supervisor_reasoning", reasoning)
    log_agent_done("Supervisor", "agent plan ready")
    return {
        "guardrail_allowed": True,
        "guardrail_reason": guardrail_reason,
        "selected_agents": selected_agents,
        "trip_constraints": constraints,
        "supervisor_reasoning": reasoning,
        "messages": [AIMessage(content="Supervisor created the agent plan.")],
        "llm_calls": llm_calls,
    }


# =========================
# Guardrail blocked response
# =========================
def guardrail_blocked_agent(state: TravelState):
    """Return the validation reason and terminate without research or approval."""
    log_agent_start("Guardrail Blocked")
    log_state_read("GuardrailBlocked", "guardrail_reason")
    reason = state.get("final_response") or state.get("guardrail_reason") or (
        "This request was blocked by the travel input guardrail."
    )
    log_state_write("GuardrailBlocked", "final_response", reason)
    log_agent_done("Guardrail Blocked", "returning blocked response")
    return {
        "final_response": reason,
        "messages": [AIMessage(content=reason)],
    }

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
    """Attempt timetable research, then generate flight guidance with available data."""
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
    """Search through Tavily MCP and store its returned content in shared state."""
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
    """Research one extracted destination using current-weather and forecast tools."""
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
# Budget Agent - new specialist
# =========================
def budget_agent(state: TravelState):
    """Estimate feasibility from the current request and available specialist results."""
    log_agent_start("Budget Agent", state["user_query"])
    for field in ("user_query", "trip_constraints", "flight_results", "hotel_results", "weather_results"):
        log_available_state("BudgetAgent", state, (field,))
    prompt = f"""
Analyze whether this trip is realistic for the user's budget.

User Query:
{state['user_query']}

Trip Constraints:
{state.get('trip_constraints', {})}

Flight Results:
{state.get('flight_results', '')}

Hotel Results:
{state.get('hotel_results', '')}

Weather Results:
{state.get('weather_results', '')}

Return:
1. Estimated cost categories
2. Budget risk areas
3. Money-saving suggestions
4. Overall feasibility

If exact live prices are unavailable, clearly label estimates as approximate.
"""

    log_llm_call("assess travel budget and feasibility")
    response = llm.invoke(
        [
            SystemMessage(content="You are a practical travel budget analyst."),
            HumanMessage(content=prompt),
        ]
    )

    log_llm_done("budget assessment", len(response.content))
    log_state_write("BudgetAgent", "budget_results", response.content[:80])
    log_agent_done("Budget Agent", f"{len(response.content)} chars")
    return {
        "budget_results": response.content,
        "messages": [AIMessage(content="Budget assessment generated.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# 🗓️  Itinerary Agent
# =========================

def itinerary_agent(state: TravelState):
    """Create a draft for human review, not an approved final response."""
    log_agent_start("Itinerary Agent", state["user_query"])

    log_available_state("ItineraryAgent", state, (
        "user_query", "flight_results", "hotel_results", "weather_results", "budget_results"
    ))

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

Budget Results:
{state.get('budget_results', '')}

Make the itinerary practical, budget-aware, and easy to follow.
"""

    log_divider("LLM → generate day-by-day itinerary")
    log_llm_call("generate day-by-day itinerary")
    response = llm.invoke([
        SystemMessage(content="You are an expert travel planner."),
        HumanMessage(content=prompt)
    ])

    approval_request = (
        "Please review the generated draft itinerary. Approve it to create the "
        "final polished plan, or provide feedback for revision."
    )

    log_llm_done("itinerary generation", len(response.content))

    log_state_write("ItineraryAgent", "itinerary", response.content[:80])
    log_agent_done("Itinerary Agent", f"{len(response.content)} chars")

    return {
        "itinerary": response.content,
        "approval_request": approval_request,
        "messages":  [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }

# =========================
# Human-in-the-Loop approval
# =========================
def human_approval_agent(state: TravelState):
    """Pause the graph until a strict approval or revision decision is supplied."""
    log_agent_start("Human Approval")
    log_state_read("HumanApproval", "itinerary")
    log_system("HITL checkpoint reached: awaiting human review (node replays on resume).")
    # Do not wrap interrupt() in try/except. LangGraph uses it to pause execution.
    # Initial execution saves a checkpoint and returns control to the browser.
    # Command(resume=...) replays this node; interrupt() then returns the decision.
    review = interrupt(
        {
            "question": "Do you approve this itinerary?",
            "draft_itinerary": state.get("itinerary", ""),
            "approval_request": state.get("approval_request", ""),
            "selected_agents": state.get("selected_agents", []),
            "supervisor_reasoning": state.get("supervisor_reasoning", ""),
            "expected_response": {
                "approved": True,
                "feedback": "Optional revision feedback",
            },
        }
    )

    decision = HumanReview.model_validate(review)
    approved = decision.approved
    human_feedback = decision.feedback

    log_result("Human review received", "Approved" if approved else "Revision requested")
    log_state_write("HumanApproval", "approved", str(approved))
    log_state_write("HumanApproval", "human_feedback", f"{len(human_feedback)} chars")
    log_agent_done("Human Approval", "finalizing approved draft" if approved else "returning for revision")
    return {
        "approved": approved,
        "human_feedback": human_feedback,
        "messages": [AIMessage(content="Human approval step completed.")],
    }





def route_after_review(state: TravelState) -> str:
    """Choose finalization only for explicit approval; otherwise prepare a revision."""
    return "final_agent" if state.get("approved") is True else "revision"


def revision_agent(state: TravelState):
    """Merge feedback into a complete request, then replan/research from scratch."""
    log_agent_start("Revision", f"revision {state.get('revision_count', 0) + 1}")
    prompt = f"""
Rewrite the current trip request to incorporate the user's latest changes.
Return strict JSON only: {{"updated_request": "complete revised travel request"}}
Preserve unchanged requirements and explicit exclusions. Latest feedback overrides
conflicting earlier requirements. Resolve relative changes exactly once. The
updated request must stand alone: replace old destinations, durations or budgets
rather than including contradictory alternatives. Do not invent missing details.
Do not answer the request or silently remove unsafe intent; preserve its meaning
so the input guardrail can validate it before research.

Current request:
{state['user_query']}

Current constraints:
{state.get('trip_constraints', {})}

Draft itinerary (context only, not new user requirements):
{state.get('itinerary', '')}

Latest user feedback:
{state.get('human_feedback', '')}
"""
    try:
        log_llm_call("merge revision feedback into trip request")
        raw = _llm_text("You update travel requirements. Return strict JSON only.", prompt)
        log_llm_done("revision request update", len(raw))
        revised = RevisedRequest.model_validate_json(raw)
    except Exception as exc:
        log_error("Revision request update", str(exc))
        log_agent_done("Revision", "keeping previous draft for another review")
        return {
            "revision_error": "Could not prepare the revision. Please retry with feedback or approve the existing draft.",
            "approval_request": "Revision could not be prepared. Your previous draft is unchanged; review it again.",
            "approved": False,
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    log_state_write("Revision", "user_query", revised.updated_request)
    log_agent_done("Revision", "returning to guardrail and supervisor")
    # A destination or budget can change. Clear old evidence before rerunning
    # research so a revised draft cannot accidentally reuse an earlier trip.
    return {
        "user_query": revised.updated_request,
        "selected_agents": [],
        "trip_constraints": _empty_constraints(),
        "supervisor_reasoning": "",
        "flight_results": "", "hotel_results": "", "weather_results": "",
        "budget_results": "", "itinerary": "", "final_response": "",
        "approval_request": "", "approved": False,
        "guardrail_allowed": True, "guardrail_reason": "",
        "revision_count": state.get("revision_count", 0) + 1,
        "revision_error": "",
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def route_after_revision(state: TravelState) -> str:
    """Retry review of the old draft if rewriting fails; otherwise replan."""
    return "human_approval" if state.get("revision_error") else "supervisor"


# =========================
# Final Response Agent - formats an explicitly approved draft
# =========================
def final_agent(state: TravelState):
    # Defense in depth: routing should reach this node only after approval.
    """Format the approved draft without reapplying earlier relative revision requests."""
    if state.get("approved") is not True:
        raise ValueError("The current draft must be approved before finalization.")
    log_agent_start("Final Agent", "formatting approved response")
    for field in ("approved", "human_feedback", "trip_constraints", "flight_results", "hotel_results", "weather_results", "budget_results", "itinerary"):
        log_available_state("FinalAgent", state, (field,))
    review_instruction = (
        "The user approved this draft. Preserve its decisions while polishing it. "
        "Any revision feedback has already been applied in the reviewed draft. "
        "Do not apply relative changes a second time."
    )

    final_prompt = f"""
Generate the final travel response for the user.

Human Review:
{review_instruction}

User Request:
{state['user_query']}

Supervisor Constraints:
{state.get('trip_constraints', {})}

Flights:
{state.get('flight_results', '')}

Hotels:
{state.get('hotel_results', '')}

Weather:
{state.get('weather_results', '')}

Budget Analysis:
{state.get('budget_results', '')}

Draft Itinerary:
{state.get('itinerary', '')}

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
- Mention that live flight APIs may not provide ticket prices when pricing is unavailable.
- Include weather-based travel advice.
- Keep the response useful for real travel planning.
- Do not alter approved requirements or invent research for skipped agents.
"""

    log_llm_call("format final travel response with human review")
    response = llm.invoke(
        [
            SystemMessage(
                content="You are a professional AI travel booking assistant."
            ),
            HumanMessage(content=final_prompt),
        ]
    )

    log_llm_done("final response formatting", len(response.content))
    log_state_write("FinalAgent", "final_response", response.content[:80])
    log_agent_done("Final Agent", f"{len(response.content)} chars")
    return {
        "final_response": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }

# =========================
# Dynamic Supervisor Routing
# =========================
ROUTE_MAP = {
    "guardrail_blocked": "guardrail_blocked",
    "flight_agent": "flight_agent",
    "hotel_agent": "hotel_agent",
    "weather_agent": "weather_agent",
    "budget_agent": "budget_agent",
    "itinerary_agent": "itinerary_agent",
}


def _selected_agents(state: TravelState) -> list[str]:
    """Filter and order the selected names using the permitted specialist list."""
    selected = state.get("selected_agents", [])
    return [agent for agent in AGENT_ORDER if agent in selected]


def route_from_supervisor(state: TravelState) -> str:
    """Send denied requests to the stop node or start the selected specialists."""
    if not state.get("guardrail_allowed", True):
        log_system("Routing: supervisor → guardrail_blocked")
        return "guardrail_blocked"

    selected = _selected_agents(state)
    target = selected[0] if selected else "itinerary_agent"
    log_system(f"Routing: supervisor → {target}")
    return target


def route_after_agent(current_agent: str):
    """Build a router that skips specialists not selected by the supervisor."""
    def route(state: TravelState) -> str:
        selected = _selected_agents(state)
        current_index = AGENT_ORDER.index(current_agent)

        for next_agent in AGENT_ORDER[current_index + 1 :]:
            if next_agent in selected:
                log_system(f"Routing: {current_agent} → {next_agent}")
                return next_agent

        log_system(f"Routing: {current_agent} → itinerary_agent")
        return "itinerary_agent"

    return route

# =========================
# Build Graph
# =========================

log_system("Building LangGraph StateGraph …")

graph = StateGraph(TravelState)

graph.add_node("supervisor", supervisor_agent)
graph.add_node("guardrail_blocked", guardrail_blocked_agent)
graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("budget_agent", budget_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("human_approval", human_approval_agent)
graph.add_node("revision", revision_agent)
graph.add_node("final_agent", final_agent)

# add_edge is unconditional; add_conditional_edges chooses one named destination.
graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", route_from_supervisor, ROUTE_MAP)

graph.add_conditional_edges(
    "flight_agent", route_after_agent("flight_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "hotel_agent", route_after_agent("hotel_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "weather_agent", route_after_agent("weather_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "budget_agent", route_after_agent("budget_agent"), ROUTE_MAP
)

graph.add_edge("itinerary_agent", "human_approval")
# Approval finalizes. Feedback loops back through revision, guardrail and research.
graph.add_conditional_edges("human_approval", route_after_review, {
    "final_agent": "final_agent", "revision": "revision",
})
graph.add_conditional_edges("revision", route_after_revision, {
    "supervisor": "supervisor", "human_approval": "human_approval",
})
graph.add_edge("final_agent", END)
graph.add_edge("guardrail_blocked", END)

log_system("Graph: guardrail/supervisor → specialists → draft → review; revise → replan; approve → final")


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

# The same database + thread_id lets a restarted app resume a saved review.
# setup() creates the checkpointer tables; it does not create semantic memory.
checkpointer = PostgresSaver(_conn)
checkpointer.setup()

log_system("PostgreSQL checkpointer ready ✔")

travel_graph = graph.compile(checkpointer=checkpointer)

log_system("LangGraph compiled — TripWise 360 is ready 🌍\n")


# =========================
# FastAPI-facing helpers
# =========================
def _interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    """Read the first pause payload from a graph invocation result."""
    interrupts = result.get("__interrupt__", [])
    if not interrupts:
        return None

    first_interrupt = interrupts[0]
    payload = getattr(first_interrupt, "value", first_interrupt)
    return payload if isinstance(payload, dict) else {"value": payload}


def _serialize_result(
    result: dict[str, Any],
    thread_id: str,
    log_status: bool = True,
) -> dict[str, Any]:
    """Convert internal graph state into the JSON shape consumed by the UI."""
    messages = result.get("messages", [])
    last_message = messages[-1].content if messages else ""
    answer = result.get("final_response") or last_message
    interrupt_payload = _interrupt_payload(result)

    if log_status:
        if interrupt_payload is not None:
            log_system(f"Graph paused for human approval · Thread: {thread_id}")
        elif not result.get("guardrail_allowed", True):
            log_system(f"Graph stopped by input guardrail · Thread: {thread_id}")
        else:
            log_graph_done(thread_id, result.get("llm_calls", 0))

    if interrupt_payload:
        answer = interrupt_payload.get("draft_itinerary") or result.get(
            "itinerary", ""
        )

    return {
        "thread_id": thread_id,
        "user_query": result.get("user_query", ""),
        "answer": answer,
        "requires_approval": interrupt_payload is not None,
        "approval_request": (
            interrupt_payload.get("approval_request", "")
            if interrupt_payload
            else result.get("approval_request", "")
        ),
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "weather_results": result.get("weather_results", ""),
        "budget_results": result.get("budget_results", ""),
        "itinerary": (
            interrupt_payload.get("draft_itinerary", "")
            if interrupt_payload
            else result.get("itinerary", "")
        ),
        "selected_agents": result.get("selected_agents", []),
        "trip_constraints": result.get("trip_constraints", {}),
        "supervisor_reasoning": result.get("supervisor_reasoning", ""),
        "guardrail_allowed": result.get("guardrail_allowed", True),
        "guardrail_reason": result.get("guardrail_reason", ""),
        "approved": result.get("approved"),
        "revision_count": result.get("revision_count", 0),
        "revision_error": result.get("revision_error", ""),
        "human_feedback": result.get("human_feedback", ""),
        "llm_calls": result.get("llm_calls", 0),
    }



def load_travel_thread(thread_id: str) -> dict[str, Any]:
    """Read a saved trip without invoking the graph or changing checkpoints."""
    thread_id = thread_id.strip()
    if not thread_id:
        raise ValueError("Please provide a thread ID.")
    snapshot = travel_graph.get_state({"configurable": {"thread_id": thread_id}})
    if not snapshot.values:
        raise LookupError("Trip not found. Check the thread ID and database connection.")
    # Saved interrupts are attached to snapshot tasks; rebuild the same response
    # shape used by a freshly paused invoke(), without invoking any agents.
    interrupts = [item for task in snapshot.tasks for item in getattr(task, "interrupts", ())]
    if snapshot.next and not interrupts:
        raise RuntimeError("This trip is running or stopped before review. Please wait for it to finish.")
    result = dict(snapshot.values)
    if interrupts:
        result["__interrupt__"] = interrupts
    log_system(f"Opened saved trip · Thread: {thread_id}")
    return _serialize_result(result, thread_id, log_status=False)


def run_travel_agent(user_input: str, thread_id: str | None = None, run_id: str | None = None):
    """Start a new travel-planning run and pause at human approval."""
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    log_graph_start(user_input, thread_id)
    config = {"configurable": {"thread_id": thread_id}}

    result = travel_graph.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_query": user_input,
            "run_id": run_id or uuid.uuid4().hex,
            "guardrail_allowed": True,
            "guardrail_reason": "",
            "selected_agents": [],
            "trip_constraints": _empty_constraints(),
            "supervisor_reasoning": "",
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "budget_results": "",
            "itinerary": "",
            "approval_request": "",
            "approved": False,
            "human_feedback": "",
            "final_response": "",
            "revision_count": 0,
            "revision_error": "",
            "llm_calls": 0,
        },
        config=config,
    )

    return _serialize_result(result, thread_id)


def resume_travel_agent(
    thread_id: str,
    approved: bool,
    feedback: str = "",
):
    """Resume the paused LangGraph thread after human review."""
    if not thread_id:
        log_error("HITL resume", "thread_id is missing")
        raise ValueError("thread_id is required to resume a travel plan.")

    log_system(f"Resuming human review · Thread: {thread_id} · Approved: {approved} · Feedback: {len(feedback.strip())} chars")
    decision = HumanReview(approved=approved, feedback=feedback)
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = travel_graph.get_state(config)
    if not any(getattr(task, "interrupts", ()) for task in snapshot.tasks):
        raise ValueError("This trip is not waiting for human review.")
    result = travel_graph.invoke(
        Command(
            resume={
                "approved": decision.approved,
                "feedback": decision.feedback,
            }
        ),
        config=config,
    )

    return _serialize_result(result, thread_id)


def get_travel_progress(thread_id: str, run_id: str | None = None):
    """Describe execution from persisted graph checkpoints, not UI timers."""
    snapshot = travel_graph.get_state({"configurable": {"thread_id": thread_id}})
    state = snapshot.values or {}
    if not state or (run_id and state.get("run_id") != run_id):
        return {"agents": [{"name": "supervisor", "status": "running"}]}

    # The checkpoint records the next scheduled nodes and completed outputs.
    # Polling can miss very short steps; a completed node may contain a tool error.
    next_nodes = set(snapshot.next)
    interrupted = any(getattr(task, "interrupts", ()) for task in snapshot.tasks)
    selected = _selected_agents(state)
    agents = ([{"name": "revision", "status": "running"}] if "revision" in next_nodes else []) + [{"name": "supervisor", "status": "running" if "supervisor" in next_nodes else "completed"}]
    fields = {
        "flight_agent": "flight_results", "hotel_agent": "hotel_results",
        "weather_agent": "weather_results", "budget_agent": "budget_results",
        "itinerary_agent": "itinerary",
    }
    if not state.get("guardrail_allowed", True):
        agents.append({"name": "guardrail_blocked", "status": "blocked"})
    else:
        for name in selected:
            status = "completed" if state.get(fields[name]) else "running" if name in next_nodes else "queued"
            agents.append({"name": name, "status": status})
        review_status = "waiting" if interrupted else "completed" if state.get("human_feedback") or state.get("approved") or state.get("final_response") else "queued"
        agents.append({"name": "human_approval", "status": review_status})
        agents.append({"name": "final_agent", "status": "completed" if state.get("final_response") else "running" if "final_agent" in next_nodes else "queued"})
    return {"agents": agents, "reason": state.get("supervisor_reasoning", "")}
