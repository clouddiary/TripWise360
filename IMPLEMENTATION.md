# TripWise 360 AI — Implementation Snapshot

> Multi-Agent Travel Planner built with LangGraph, LangChain, FastAPI, PostgreSQL, Tavily and AviationStack.

---

## Project Structure

```
TripWise360/
├── app.py                      # FastAPI server — routes & entry point
├── backend.py                  # LangGraph graph, agents, state, checkpointer
├── requirements.txt            # Python dependencies (pinned versions)
├── Dockerfile                  # Multi-stage Docker build for ECS deployment
├── .env                        # Secrets & config (never committed)
├── tools/
│   ├── __init__.py
│   ├── flight_tool.py          # AviationStack flight search + NLP route parser
│   └── tavily_tool.py          # Tavily web search wrapper
├── templates/
│   └── index.html              # Frontend UI (Jinja2 template)
└── static/
    ├── style.css               # UI styles
    └── script.js               # Frontend JS (fetch, markdown, PDF download)
```

---

## Environment Variables (`.env`)

```env
OPEN_API_KEY=<OpenAI API key>
AVIATIONSTACK_API_KEY=<AviationStack API key>
TAVILY_API_KEY=<Tavily API key>
DEFAULT_ORIGIN_IATA=USA
DATABASE_URL=postgresql://<user>:<password>@<host>/<db>
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=<LangSmith API key>
LANGSMITH_PROJECT=TripWise360
```

---

## Dependencies (`requirements.txt`)

```
langgraph==1.2.2
langchain==1.3.2
langchain-groq==1.1.3
langchain-community==0.4.2
langchain-tavily==0.2.18
psycopg[binary]==3.3.4
psycopg_pool==3.3.1
python-dotenv==1.2.2
tavily-python==0.7.24
requests==2.34.2
langgraph-checkpoint-postgres==3.1.0
airportsdata==20260315
pycountry==26.2.16
fastapi==0.136.3
uvicorn==0.48.0
jinja2==3.1.6
langchain-openai==1.7.3
langchain-mcp-adapters==0.3.0
nest_asyncio==1.6.0
mcp==1.28.1
certifi==2026.6.17
```

---

## Backend (`backend.py`)

### LangGraph State

```python
class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]  # append-only message history
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int
```

### LLM

```python
llm = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0,
    max_retries=2,
    api_key=OPEN_API_KEY,
)
```

### Agents

**1. flight_agent**
- Calls `search_flights(query)` from `tools/flight_tool.py`
- Writes to `flight_results` channel
- No LLM call (tool-only)

**2. hotel_agent**
- Calls `tavily_search("Best hotels for <query>")` from `tools/tavily_tool.py`
- Writes to `hotel_results` channel
- No LLM call (tool-only)

**3. itinerary_agent**
- Calls `llm.invoke()` with flight + hotel results
- System prompt: "You are an expert travel planner."
- Writes to `itinerary` channel

**4. final_agent**
- Calls `llm.invoke()` with all previous results
- Produces the final formatted response with 6 sections:
  1. Trip Summary
  2. Flight Information
  3. Hotel Suggestions
  4. Day-by-Day Itinerary
  5. Estimated Budget
  6. Final Recommendations

### Graph

```
START → flight_agent → hotel_agent → itinerary_agent → final_agent → END
```

### PostgreSQL Checkpointer

```python
_conn = psycopg.connect(DATABASE_URL, autocommit=True, row_factory=dict_row)
checkpointer = PostgresSaver(_conn)
checkpointer.setup()
travel_graph = graph.compile(checkpointer=checkpointer)
```

LangGraph creates 4 tables in the database:
- `checkpoints` — one row per graph step (agent transition)
- `checkpoint_blobs` — state values per channel (msgpack-serialized)
- `checkpoint_writes` — per-agent write-ahead log
- `checkpoint_migrations` — schema version tracking (v0–v9)

### Entry Function

```python
def run_travel_agent(user_input: str, thread_id: str | None = None) -> dict:
    # Generates thread_id if not provided
    # Invokes travel_graph with full TravelState
    # Returns: thread_id, answer, flight_results, hotel_results, itinerary, llm_calls
```

---

## FastAPI Server (`app.py`)

### Routes

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serves `index.html` |
| `POST` | `/api/travel` | Accepts `{ message, thread_id }`, returns travel plan |
| `GET` | `/health` | Health check |
| `GET` | `/favicon.ico` | Returns empty JSON (suppress 404) |

### Request / Response

```json
// POST /api/travel
Request:  { "message": "Plan a 7 day Japan trip...", "thread_id": null }

Response: {
  "success": true,
  "thread_id": "user_<uuid>",
  "answer": "...",
  "flight_results": "...",
  "hotel_results": "...",
  "itinerary": "...",
  "llm_calls": 4
}
```

### App Config

```python
app = FastAPI(
    title="TripWise 360 AI",
    description="LangGraph Multi-Agent Travel Planner with FastAPI Frontend",
    version="1.0.0"
)
```

---

## Tools

### `tools/flight_tool.py`

Wraps the [AviationStack API](https://aviationstack.com) with full NLP route parsing.

**Key functions:**
- `search_flights(query, limit=10)` — main entry point
- `parse_route(query)` → `(dep_iata, arr_iata)` — extracts route from natural language
- `resolve_location_to_iata(location)` — converts country/city name to IATA code
- `find_location_mentions(query)` — detects country/city names in free text
- `get_best_airport_for_country(country_code)` — picks the main international airport
- `format_flight(flight)` — formats a single flight dict into readable text

**Lookup data used:**
- `airportsdata` — full IATA airport database
- `pycountry` — ISO country codes
- `COUNTRY_ALIASES` — manual aliases (usa→US, uk→GB, dubai→AE, etc.)
- `COUNTRY_MAIN_AIRPORT` — preferred hub per country (IN→DEL, US→JFK, etc.)
- `CITY_MAIN_AIRPORT` — preferred airport per city (tokyo→NRT, london→LHR, etc.)

**Route parsing logic (in order):**
1. Global keywords → return `(None, None)` for live worldwide flights
2. Explicit IATA codes in query (e.g. `DAC to NRT`)
3. `from X to Y` pattern
4. `to Y from X` pattern
5. `from X` or `to Y` pattern
6. Country/city mention scanning
7. Fallback to `DEFAULT_ORIGIN_IATA` + detected destination

---

### `tools/tavily_tool.py`

Wraps the [Tavily](https://tavily.com) search API.

```python
def tavily_search(query: str) -> str:
    # Searches with max_results=5
    # Returns formatted string: "1. **Title**\n   url\n   snippet..."
    # Snippets truncated to 300 chars
```

**Note:** Pass keyword-focused queries (e.g. `"best hotels New York City"`), not conversational questions, to get relevant results.

---

## Frontend

### `templates/index.html`

- Page title: **TripWise 360 AI**
- Badge: `✈️ TripWise 360 AI — A Multi-Agent Travel Planner with LangGraph`
- Textarea for travel query input
- Quick prompt buttons: Japan Trip, Dubai Trip, Thailand Trip, Global Flights
- Result section with Copy and Download PDF buttons
- Footer: `Built with FastAPI, LangGraph, OpenAI, PostgreSQL, Tavily and AviationStack`
- External scripts: `marked.js` (markdown rendering), `html2pdf.js` (PDF export)

### `static/script.js`

Key functions:
- `sendMessage()` — POSTs to `/api/travel`, stores `thread_id` in `localStorage`
- `showResult(answer, threadId)` — renders markdown via `marked.parse()`
- `copyResult()` — copies plain text to clipboard
- `downloadPDF()` — exports result as A4 PDF via `html2pdf.js`
- `setPrompt(text)` — fills textarea with a quick prompt
- `Ctrl+Enter` keyboard shortcut → triggers `sendMessage()`

Thread continuity: `thread_id` is persisted in `localStorage` so follow-up queries within the same browser session maintain conversation context via LangGraph's checkpointer.

---

## Docker (`Dockerfile`)

Multi-stage build:

```
Stage 1 (builder):  python:3.12-slim
  - Installs gcc, libpq-dev
  - pip install --prefix=/install -r requirements.txt

Stage 2 (runtime):  python:3.12-slim
  - Installs libpq5, ca-certificates (runtime only)
  - Copies /install from builder
  - Copies app source: app.py, backend.py, tools/, static/, templates/
  - Runs as non-root user (appuser)
  - Exposes port 8000
  - CMD: uvicorn app:app --host 0.0.0.0 --port 8000 --workers 2
```

Secrets are injected via ECS Task Definition environment variables — `.env` is never copied into the image.

---

## LangSmith Tracing

Tracing is enabled automatically via environment variables — no code changes needed.

```
LANGSMITH_TRACING=true        → activates LangChainTracer callback
LANGSMITH_API_KEY=...         → authenticates with LangSmith
LANGSMITH_ENDPOINT=...        → target API endpoint
LANGSMITH_PROJECT=TripWise360 → groups all runs under this project
```

Every `travel_graph.invoke()` produces a trace tree:
```
TravelGraph.invoke()
  ├── flight_agent
  ├── hotel_agent
  ├── itinerary_agent
  │     └── ChatOpenAI.invoke()
  └── final_agent
        └── ChatOpenAI.invoke()
```

---

## How to Run Locally

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your keys to .env (see Environment Variables section above)

# 4. Run the server
python app.py
# → http://127.0.0.1:8000
```
