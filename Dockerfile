# ─────────────────────────────────────────────
# Stage 1: builder — install Python dependencies
# ─────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# Build deps for psycopg binary
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt


# ─────────────────────────────────────────────
# Stage 2: runtime — lean final image
# ─────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Runtime deps:
#   libpq5        — psycopg (PostgreSQL client)
#   ca-certificates — SSL / certifi
#   curl          — used by uv installer below
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv + uvx so aviationstack-mcp and the custom weather server
# can be launched as stdio MCP sub-processes at runtime.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:/root/.local/bin:$PATH"

# Copy installed Python packages from builder stage
COPY --from=builder /install /usr/local

# ── Application source ───────────────────────
COPY app.py                      ./app.py
COPY backend.py                  ./backend.py
COPY mcp_client.py               ./mcp_client.py
COPY custom_weather_mcp_server.py ./custom_weather_mcp_server.py
COPY logger.py                   ./logger.py
COPY static/                     ./static/
COPY templates/                  ./templates/

# Non-root user for security
RUN useradd --no-create-home --shell /bin/false appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Secrets must be injected at runtime via environment variables
# (ECS Task Definition / docker run --env-file).
# Never bake .env into the image.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
