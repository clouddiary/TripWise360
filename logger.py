"""
logger.py — Colored console logging for TripWise 360 demo.

Color scheme:
  SYSTEM   → bold white      (startup, graph, state)
  AGENT    → bold cyan       (agent entry/exit)
  LLM      → bold magenta    (every LLM call)
  MCP      → bold yellow     (MCP tool calls)
  RESULT   → bold green      (successful data returned)
  ERROR    → bold red        (failures / fallbacks)
  STATE    → bold blue       (shared state read/write)
  DIVIDER  → dim white       (visual separators)
"""

import time

# ── ANSI codes ──────────────────────────────────────────────────────────────

RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"

# Regular colours
RED     = "\033[31m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
BLUE    = "\033[34m"
MAGENTA = "\033[35m"
CYAN    = "\033[36m"
WHITE   = "\033[37m"

# Bright colours
BRED    = "\033[91m"
BGREEN  = "\033[92m"
BYELLOW = "\033[93m"
BBLUE   = "\033[94m"
BMAGENTA= "\033[95m"
BCYAN   = "\033[96m"
BWHITE  = "\033[97m"

# Background
BG_BLUE  = "\033[44m"
BG_CYAN  = "\033[46m"


def _now():
    return time.strftime("%H:%M:%S")


# ── Public helpers ───────────────────────────────────────────────────────────

def log_system(msg: str):
    """White — app startup, graph build, DB connection."""
    print(f"{DIM}{_now()}{RESET}  {BOLD}{BWHITE}[SYSTEM  ]{RESET}  {WHITE}{msg}{RESET}")


def log_agent_start(agent_name: str, query_preview: str = ""):
    """Cyan banner — fired when an agent node begins."""
    bar = "─" * 60
    print(f"\n{BOLD}{BCYAN}{bar}{RESET}")
    print(f"{DIM}{_now()}{RESET}  {BOLD}{BCYAN}[AGENT ▶ ]{RESET}  {BCYAN}{agent_name.upper()}{RESET}  {DIM}{query_preview[:80]}{RESET}")
    print(f"{BOLD}{BCYAN}{bar}{RESET}")


def log_agent_done(agent_name: str, detail: str = ""):
    """Cyan — agent completed."""
    print(f"{DIM}{_now()}{RESET}  {BOLD}{BCYAN}[AGENT ✔ ]{RESET}  {BCYAN}{agent_name.upper()} done{RESET}  {DIM}{detail}{RESET}\n")


def log_llm_call(purpose: str, model: str = "gpt-5-nano"):
    """Magenta — every time the LLM is invoked."""
    print(f"{DIM}{_now()}{RESET}  {BOLD}{BMAGENTA}[LLM  🧠 ]{RESET}  {MAGENTA}Calling {model}{RESET}  {DIM}→ {purpose}{RESET}")


def log_llm_done(purpose: str, chars: int = 0):
    """Magenta — LLM returned."""
    size = f"  {DIM}({chars} chars){RESET}" if chars else ""
    print(f"{DIM}{_now()}{RESET}  {BOLD}{BMAGENTA}[LLM  ✔  ]{RESET}  {MAGENTA}{purpose} complete{RESET}{size}")


def log_mcp_call(server: str, tool: str, args: dict = None):
    """Yellow — MCP tool being invoked."""
    args_str = f"  {DIM}{args}{RESET}" if args else ""
    print(f"{DIM}{_now()}{RESET}  {BOLD}{BYELLOW}[MCP  🔧 ]{RESET}  {YELLOW}{server}{RESET} → {BYELLOW}{tool}{RESET}{args_str}")


def log_mcp_done(server: str, tool: str, preview: str = ""):
    """Yellow — MCP tool returned data."""
    prev = f"  {DIM}{preview[:100]}{'…' if len(preview) > 100 else ''}{RESET}" if preview else ""
    print(f"{DIM}{_now()}{RESET}  {BOLD}{BYELLOW}[MCP  ✔  ]{RESET}  {YELLOW}{server}{RESET} ← {BYELLOW}{tool}{RESET}{prev}")


def log_mcp_error(server: str, tool: str, error: str):
    """Red — MCP tool failed."""
    print(f"{DIM}{_now()}{RESET}  {BOLD}{BRED}[MCP  ✖  ]{RESET}  {RED}{server}.{tool} FAILED{RESET}  {DIM}{str(error)[:120]}{RESET}")


def log_state_read(agent: str, field: str):
    """Blue — agent reading from shared state."""
    print(f"{DIM}{_now()}{RESET}  {BOLD}{BBLUE}[STATE 📖]{RESET}  {BLUE}{agent}{RESET} reading  {BBLUE}{field}{RESET}")


def log_state_write(agent: str, field: str, preview: str = ""):
    """Blue — agent writing to shared state."""
    prev = f"  {DIM}{str(preview)[:80]}{'…' if len(str(preview)) > 80 else ''}{RESET}" if preview else ""
    print(f"{DIM}{_now()}{RESET}  {BOLD}{BBLUE}[STATE 📝]{RESET}  {BLUE}{agent}{RESET} writing  {BBLUE}{field}{RESET}{prev}")


def log_result(label: str, detail: str = ""):
    """Green — a meaningful result was obtained."""
    det = f"  {DIM}{detail[:100]}{'…' if len(detail) > 100 else ''}{RESET}" if detail else ""
    print(f"{DIM}{_now()}{RESET}  {BOLD}{BGREEN}[RESULT ✦]{RESET}  {GREEN}{label}{RESET}{det}")


def log_error(context: str, error: str):
    """Red — something went wrong (non-fatal fallback)."""
    print(f"{DIM}{_now()}{RESET}  {BOLD}{BRED}[ERROR  ✖]{RESET}  {RED}{context}{RESET}  {DIM}{str(error)[:160]}{RESET}")


def log_divider(label: str = ""):
    """Dim separator for visual grouping."""
    if label:
        pad = max(0, 56 - len(label))
        print(f"{DIM}{'─' * 4}  {label}  {'─' * pad}{RESET}")
    else:
        print(f"{DIM}{'─' * 64}{RESET}")


def log_graph_start(query: str, thread_id: str):
    """Bold white banner — LangGraph invocation begins."""
    print(f"\n{BOLD}{BG_BLUE}{BWHITE}{'═' * 64}{RESET}")
    print(f"{BOLD}{BG_BLUE}{BWHITE}  🌍 TRIPWISE 360  ·  NEW REQUEST{RESET}")
    print(f"{BOLD}{BG_BLUE}{BWHITE}  Thread : {thread_id}{RESET}")
    print(f"{BOLD}{BG_BLUE}{BWHITE}  Query  : {query[:80]}{'…' if len(query) > 80 else ''}{RESET}")
    print(f"{BOLD}{BG_BLUE}{BWHITE}{'═' * 64}{RESET}\n")


def log_graph_done(thread_id: str, llm_calls: int):
    """Bold green banner — full pipeline finished."""
    print(f"\n{BOLD}{BGREEN}{'═' * 64}{RESET}")
    print(f"{BOLD}{BGREEN}  ✅ PLAN COMPLETE  ·  Thread: {thread_id}  ·  LLM calls: {llm_calls}{RESET}")
    print(f"{BOLD}{BGREEN}{'═' * 64}{RESET}\n")
