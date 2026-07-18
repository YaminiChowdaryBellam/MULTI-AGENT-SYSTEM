import json
import logging
import os
import re
import time

from groq import Groq, RateLimitError
from langfuse import get_client, observe

_client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "llama-3.1-8b-instant"

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

# Updated after every call_groq() — lets callers (e.g. the A/B routing
# experiment) read the most recent call's real token usage without changing
# call_groq's `-> str` contract everywhere it's already used.
LAST_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


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
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            usage = response.usage
            if usage:
                LAST_USAGE["prompt_tokens"] = usage.prompt_tokens
                LAST_USAGE["completion_tokens"] = usage.completion_tokens
                LAST_USAGE["total_tokens"] = usage.total_tokens
                cost = (
                    usage.prompt_tokens / 1_000_000 * COST_PER_1M_INPUT_TOKENS
                    + usage.completion_tokens / 1_000_000 * COST_PER_1M_OUTPUT_TOKENS
                )
                get_client().update_current_generation(
                    model=MODEL,
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
