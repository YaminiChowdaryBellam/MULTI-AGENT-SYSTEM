"""
FastAPI REST API — wraps the LangGraph clinical research assistant as HTTP endpoints.
Mirrors RAG-medical-system's src/api.py pattern (rate limiting, health check,
global exception handler).

Endpoints:
  POST /query   — submit a clinical research question, get a synthesized, cited answer
  GET  /health  — system status check

Run locally:
  uvicorn api:app --reload --port 8001

Example request:
  curl -X POST http://localhost:8001/query \
       -H "Content-Type: application/json" \
       -d '{"query": "What are the treatment options for atrial fibrillation?"}'
"""

import time
import uuid

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from slowapi import Limiter, _rate_limit_exceeded_handler  # noqa: E402
from slowapi.errors import RateLimitExceeded  # noqa: E402
from slowapi.util import get_remote_address  # noqa: E402

from graph.build import get_graph, run_graph  # noqa: E402

# Rate limiter — 10 requests/min per IP. Groq's free-tier TPM limits are the
# real bottleneck (see graph/llm.py's retry/backoff); this just keeps a single
# client from hammering past them before backoff has a chance to help.
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Multi-Agent Clinical Research Assistant API",
    description=(
        "Ask clinical/medical research questions. Routed across PubMed, openFDA, "
        "ClinicalTrials.gov, an internal RAG knowledge base, and live web search, "
        "with PHI redaction and scope guardrails on every request."
    ),
    version="1.0.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Compile the graph once at startup, not on every request.
get_graph()


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="The clinical research question to answer",
        examples=["What are the treatment options for atrial fibrillation?"],
    )
    thread_id: str | None = Field(
        None,
        description=(
            "Reuse a prior response's thread_id to ask a follow-up in the same "
            "checkpointed session; omit to start a new one."
        ),
    )


class QueryResponse(BaseModel):
    query: str
    answer: str
    thread_id: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    engine: str


@app.post("/query", response_model=QueryResponse)
@limiter.limit("10/minute")
async def query(request: Request, body: QueryRequest):
    """
    Submit a clinical research question and receive a synthesized, cited answer.

    Runs the full LangGraph pipeline: PHI redaction and scope guardrails, agent
    routing, parallel specialist dispatch, a confidence-gated reflection retry,
    citation-enforced synthesis, and a human-review gate for low-confidence answers.
    """
    query_text = body.query.strip()
    if not query_text:
        raise HTTPException(status_code=422, detail="Query cannot be empty.")

    thread_id = body.thread_id or str(uuid.uuid4())
    start = time.perf_counter()
    answer = run_graph(query_text, thread_id=thread_id)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    return QueryResponse(query=query_text, answer=answer, thread_id=thread_id, latency_ms=latency_ms)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Returns system status — useful for deployment monitoring."""
    return HealthResponse(status="ok", engine="langgraph")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
    )
