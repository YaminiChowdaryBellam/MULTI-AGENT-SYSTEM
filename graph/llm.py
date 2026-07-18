import json
import logging
import os
import re
import time

import requests
from groq import Groq, RateLimitError
from langfuse import get_client, observe

_client = Groq(api_key=os.environ["GROQ_API_KEY"])
GROQ_MODEL = "llama-3.1-8b-instant"

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# Step 6.1 — tiered models: route "cheap" tier calls (routing + guard checks)
# to a quantized local Ollama model instead of Groq when enabled; "expensive"
# tier (synthesis) always uses Groq — it's the one call where output quality
# matters most, and by then the evidence is already narrowed down so there's
# only ever one such call per turn (vs up to 5 cheap-tier calls).
TIERED_MODELS_ENABLED = os.getenv("TIERED_MODELS", "false").lower() == "true"

# Langfuse's client checks `is None`, not truthiness, to decide whether a key
# was provided — an empty `LANGFUSE_PUBLIC_KEY=` line in .env (as opposed to
# leaving it commented out) would otherwise attempt, and fail, a real network
# call instead of cleanly no-op'ing. Normalize empty-string env vars to unset.
for _langfuse_key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
    if os.environ.get(_langfuse_key) == "":
        del os.environ[_langfuse_key]

# get_client() isn't cached when unauthenticated, so without this every node's
# @observe call would re-log "Langfuse client initialized without public_key"
# to stderr on every single request when LANGFUSE_PUBLIC_KEY isn't set — the
# expected, supported way to run this project without a Langfuse account.
if not os.getenv("LANGFUSE_PUBLIC_KEY"):
    logging.getLogger("langfuse").setLevel(logging.ERROR)

# Approximate Groq list pricing for llama-3.1-8b-instant, per 1M tokens — verify
# against console.groq.com/settings/billing before trusting cost figures for
# anything beyond relative comparisons; override via env if it drifts.
COST_PER_1M_INPUT_TOKENS = float(os.getenv("GROQ_COST_PER_1M_INPUT_TOKENS", "0.05"))
COST_PER_1M_OUTPUT_TOKENS = float(os.getenv("GROQ_COST_PER_1M_OUTPUT_TOKENS", "0.08"))

# Updated after every call_groq()/call_ollama() — lets callers (e.g. the A/B
# routing experiment, the tiering benchmark) read the most recent call's real
# backend/usage/cost without changing either function's `-> str` contract.
LAST_USAGE = {"backend": None, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost_usd": 0.0}


@observe(as_type="generation", name="groq_call")
def call_groq(prompt: str, retries: int = 3) -> str:
    """
    Parallel fan-out to specialist nodes means several Groq calls can land in
    the same instant, which bursts past free-tier TPM limits far more easily
    than the old orchestrator's sequential for-loop did. Retry with backoff
    rather than letting the whole graph run fail on a transient 429.

    Wrapped as a Langfuse generation — captures latency automatically and
    token usage/cost explicitly below. A no-op if LANGFUSE_PUBLIC_KEY/
    LANGFUSE_SECRET_KEY aren't set (see graph/llm.py's Langfuse client init).
    """
    for attempt in range(retries):
        try:
            response = _client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            usage = response.usage
            if usage:
                cost = (
                    usage.prompt_tokens / 1_000_000 * COST_PER_1M_INPUT_TOKENS
                    + usage.completion_tokens / 1_000_000 * COST_PER_1M_OUTPUT_TOKENS
                )
                LAST_USAGE.update({
                    "backend": "groq",
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                    "cost_usd": round(cost, 8),
                })
                get_client().update_current_generation(
                    model=GROQ_MODEL,
                    usage_details={
                        "input": usage.prompt_tokens,
                        "output": usage.completion_tokens,
                        "total": usage.total_tokens,
                    },
                    cost_details={"total": round(cost, 8)},
                )
            return response.choices[0].message.content.strip()
        except RateLimitError as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"[graph.llm] Rate limited by Groq, retry {attempt + 1}/{retries} after {wait}s — {e}")
            time.sleep(wait)


@observe(as_type="generation", name="ollama_call")
def call_ollama(prompt: str, retries: int = 3, timeout: int = 60) -> str:
    """
    Local quantized model (see OLLAMA_MODEL) — zero marginal cost and no
    per-minute rate limit, at the cost of weaker output quality than Groq's
    hosted model. Requires `ollama serve` running locally; used for the
    "cheap" tier (routing + guard checks) when TIERED_MODELS=true.
    """
    for attempt in range(retries):
        try:
            resp = requests.post(
                f"{OLLAMA_HOST}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            prompt_tokens = data.get("prompt_eval_count", 0)
            completion_tokens = data.get("eval_count", 0)
            LAST_USAGE.update({
                "backend": "ollama",
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "cost_usd": 0.0,
            })
            get_client().update_current_generation(
                model=OLLAMA_MODEL,
                usage_details={
                    "input": prompt_tokens,
                    "output": completion_tokens,
                    "total": prompt_tokens + completion_tokens,
                },
                cost_details={"total": 0.0},
            )
            return data["message"]["content"].strip()
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"[graph.llm] Ollama call failed, retry {attempt + 1}/{retries} after {wait}s — {e}")
            time.sleep(wait)


def call_llm(prompt: str, tier: str = "cheap", retries: int = 3) -> str:
    """
    tier="cheap"  (classify/router/confidence_gate/reflect/scope-check): the
        local Ollama model when TIERED_MODELS=true, otherwise Groq — same
        behavior as before tiering existed.
    tier="expensive" (synthesize): always Groq, regardless of TIERED_MODELS.
    """
    if tier == "cheap" and TIERED_MODELS_ENABLED:
        return call_ollama(prompt, retries=retries)
    return call_groq(prompt, retries=retries)


def extract_json(raw: str) -> dict | None:
    """
    LLMs often wrap JSON in markdown code fences — strip those before parsing.
    Returns None if no valid JSON object could be extracted, distinct from a
    validly parsed `{}` — callers that need to tell "the model failed to
    respond usefully" apart from "the model deliberately said 'nothing here'"
    check for None; callers that don't care can just do `extract_json(x) or {}`.
    """
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    cleaned = match.group(0) if match else raw
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None
