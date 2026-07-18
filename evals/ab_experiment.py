"""
Step 5.4 — A/B experiment: LLM router (graph.nodes.router_node) vs
embedding-similarity router (graph.embedding_router.embedding_route).
Logs accuracy, latency, and cost per variant over the gold set, plus a
simple drift check comparing each variant's agent-selection distribution.

Run: python -m evals.ab_experiment
"""

import time
from collections import Counter

from dotenv import load_dotenv

load_dotenv()

from evals.gold_set import GOLD_SET  # noqa: E402
from evals.routing_eval import score_one  # noqa: E402
from graph import llm  # noqa: E402
from graph.embedding_router import embedding_route  # noqa: E402
from graph.nodes import router_node  # noqa: E402


def _run_variant(entries: list[dict], route_fn, track_cost: bool) -> dict:
    per_query = []
    total_cost = 0.0
    for entry in entries:
        start = time.perf_counter()
        routing = route_fn(entry["query"])
        latency = time.perf_counter() - start

        if track_cost:
            total_cost += llm.LAST_USAGE["cost_usd"]

        actual = set(routing.keys())
        expected = set(entry["expected_agents"])
        scores = score_one(expected, actual)
        per_query.append({
            "id": entry["id"], "actual_agents": sorted(actual),
            "latency_s": round(latency, 4), **scores,
        })

    n = len(per_query) or 1
    return {
        "accuracy": round(sum(p["exact_match"] for p in per_query) / n, 4),
        "precision": round(sum(p["precision"] for p in per_query) / n, 4),
        "recall": round(sum(p["recall"] for p in per_query) / n, 4),
        "f1": round(sum(p["f1"] for p in per_query) / n, 4),
        "mean_latency_s": round(sum(p["latency_s"] for p in per_query) / n, 4),
        "total_cost_usd": round(total_cost, 6),
        "n_queries": len(per_query),
        "per_query": per_query,
    }


def _llm_route(query: str) -> dict:
    return router_node({"redacted_query": query, "history": []})["routing"]


def _agent_distribution(per_query: list[dict], agents: list[str]) -> dict[str, float]:
    counts = Counter(agent for p in per_query for agent in p["actual_agents"])
    total = sum(counts.values()) or 1
    return {agent: round(counts.get(agent, 0) / total, 4) for agent in agents}


def _total_variation_distance(dist_a: dict[str, float], dist_b: dict[str, float]) -> float:
    agents = set(dist_a) | set(dist_b)
    return round(0.5 * sum(abs(dist_a.get(a, 0.0) - dist_b.get(a, 0.0)) for a in agents), 4)


def run_ab_experiment(gold_set: list[dict] | None = None, limit: int | None = None) -> dict:
    entries = gold_set if gold_set is not None else GOLD_SET
    if limit is not None:
        entries = entries[:limit]

    llm_result = _run_variant(entries, _llm_route, track_cost=True)
    embedding_result = _run_variant(entries, embedding_route, track_cost=False)

    agents = sorted({a for e in entries for a in e["expected_agents"]})
    llm_dist = _agent_distribution(llm_result["per_query"], agents)
    embedding_dist = _agent_distribution(embedding_result["per_query"], agents)
    drift = _total_variation_distance(llm_dist, embedding_dist)

    return {
        "llm_router": llm_result,
        "embedding_router": embedding_result,
        "agent_distribution": {"llm_router": llm_dist, "embedding_router": embedding_dist},
        "distribution_drift_tvd": drift,
    }


def _print_report(result: dict, drift_threshold: float = 0.15) -> None:
    llm_r, emb_r = result["llm_router"], result["embedding_router"]
    print("=" * 68)
    print("A/B EXPERIMENT: LLM router vs embedding-similarity router  (Step 5.4)")
    print("=" * 68)
    print(f"{'metric':<20}{'llm_router':>18}{'embedding_router':>22}")
    print(f"{'n_queries':<20}{llm_r['n_queries']:>18}{emb_r['n_queries']:>22}")
    print(f"{'accuracy':<20}{llm_r['accuracy']:>17.1%} {emb_r['accuracy']:>21.1%}")
    print(f"{'precision':<20}{llm_r['precision']:>17.1%} {emb_r['precision']:>21.1%}")
    print(f"{'recall':<20}{llm_r['recall']:>17.1%} {emb_r['recall']:>21.1%}")
    print(f"{'f1':<20}{llm_r['f1']:>17.1%} {emb_r['f1']:>21.1%}")
    print(f"{'mean_latency_s':<20}{llm_r['mean_latency_s']:>18.4f}{emb_r['mean_latency_s']:>22.4f}")
    print(f"{'total_cost_usd':<20}{llm_r['total_cost_usd']:>18.6f}{emb_r['total_cost_usd']:>22.6f}")

    print("\nAgent selection distribution (share of all selections made):")
    dist = result["agent_distribution"]
    agents = sorted(set(dist["llm_router"]) | set(dist["embedding_router"]))
    for agent in agents:
        print(f"  {agent:<16} llm={dist['llm_router'].get(agent, 0):>6.1%}   "
              f"embedding={dist['embedding_router'].get(agent, 0):>6.1%}")

    drift = result["distribution_drift_tvd"]
    flag = " <-- DRIFT" if drift > drift_threshold else ""
    print(f"\nDistribution drift (total variation distance): {drift:.4f}{flag}")


if __name__ == "__main__":
    _print_report(run_ab_experiment())
