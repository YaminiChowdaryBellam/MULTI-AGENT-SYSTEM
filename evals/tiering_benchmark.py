"""
Step 6.1 — benchmark cost/query and latency before vs after model tiering.

Runs the same sample of gold-set queries through the full graph twice: once
with TIERED_MODELS off (today's default — every LLM call goes to Groq) and
once with it on (classify/router/confidence_gate/reflect/scope-check go to
the local quantized Ollama model; only synthesize still uses Groq).

Wraps graph.llm.call_groq/call_ollama for the duration of each run to count
calls and accumulate real cost (via LAST_USAGE, which both functions update)
without touching the production call path.

Run: python -m evals.tiering_benchmark [--limit N]
"""

import argparse
import time
from unittest.mock import patch

from dotenv import load_dotenv

load_dotenv()

from evals.gold_set import GOLD_SET  # noqa: E402
from graph import llm  # noqa: E402
from graph.build import get_graph  # noqa: E402


def _run_benchmark(entries: list[dict], tiered: bool, thread_prefix: str) -> dict:
    calls = {"groq": 0, "ollama": 0, "total_cost_usd": 0.0}
    original_groq = llm.call_groq
    original_ollama = llm.call_ollama

    def wrapped_groq(prompt, retries=3):
        result = original_groq(prompt, retries=retries)
        calls["groq"] += 1
        calls["total_cost_usd"] += llm.LAST_USAGE["cost_usd"]
        return result

    def wrapped_ollama(prompt, retries=3, timeout=60):
        result = original_ollama(prompt, retries=retries, timeout=timeout)
        calls["ollama"] += 1
        calls["total_cost_usd"] += llm.LAST_USAGE["cost_usd"]  # always 0.0 — local compute
        return result

    per_query_latency = []
    with patch.object(llm, "call_groq", wrapped_groq), \
         patch.object(llm, "call_ollama", wrapped_ollama), \
         patch.object(llm, "TIERED_MODELS_ENABLED", tiered):
        app = get_graph()
        for i, entry in enumerate(entries):
            start = time.perf_counter()
            app.invoke(
                {"query": entry["query"]},
                config={"configurable": {"thread_id": f"{thread_prefix}-{i}"}},
            )
            per_query_latency.append(time.perf_counter() - start)

    n = len(entries) or 1
    return {
        "mean_latency_s": round(sum(per_query_latency) / n, 4),
        "total_cost_usd": round(calls["total_cost_usd"], 6),
        "mean_cost_usd_per_query": round(calls["total_cost_usd"] / n, 8),
        "groq_calls": calls["groq"],
        "ollama_calls": calls["ollama"],
        "n_queries": n,
    }


def run_tiering_benchmark(gold_set: list[dict] | None = None, limit: int = 5) -> dict:
    entries = [e for e in (gold_set if gold_set is not None else GOLD_SET) if e["e2e"]][:limit]
    before = _run_benchmark(entries, tiered=False, thread_prefix="bench-before")
    after = _run_benchmark(entries, tiered=True, thread_prefix="bench-after")
    return {"before_tiering": before, "after_tiering": after}


def _print_report(result: dict) -> None:
    before, after = result["before_tiering"], result["after_tiering"]
    print("=" * 68)
    print("MODEL TIERING BENCHMARK  (Step 6.1)")
    print("=" * 68)
    print(f"{'metric':<28}{'before (all-Groq)':>20}{'after (tiered)':>20}")
    print(f"{'n_queries':<28}{before['n_queries']:>20}{after['n_queries']:>20}")
    print(f"{'mean_latency_s':<28}{before['mean_latency_s']:>20.4f}{after['mean_latency_s']:>20.4f}")
    print(f"{'total_cost_usd':<28}{before['total_cost_usd']:>20.6f}{after['total_cost_usd']:>20.6f}")
    print(f"{'mean_cost_usd_per_query':<28}{before['mean_cost_usd_per_query']:>20.8f}{after['mean_cost_usd_per_query']:>20.8f}")
    print(f"{'groq_calls':<28}{before['groq_calls']:>20}{after['groq_calls']:>20}")
    print(f"{'ollama_calls':<28}{before['ollama_calls']:>20}{after['ollama_calls']:>20}")

    if before["total_cost_usd"] > 0:
        cost_reduction = 1 - (after["total_cost_usd"] / before["total_cost_usd"])
        print(f"\nCost reduction: {cost_reduction:.1%}")
    latency_delta = after["mean_latency_s"] - before["mean_latency_s"]
    print(f"Latency delta: {latency_delta:+.4f}s per query")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark model tiering (Step 6.1)")
    parser.add_argument("--limit", type=int, default=5, help="Number of e2e gold-set queries to run per configuration")
    args = parser.parse_args()
    _print_report(run_tiering_benchmark(limit=args.limit))
