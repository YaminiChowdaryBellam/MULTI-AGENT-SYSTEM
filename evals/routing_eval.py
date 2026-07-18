"""
Step 4.1 — routing accuracy.

Calls graph.nodes.router_node directly rather than running the full graph,
so this eval costs exactly one Groq call per query and never touches PHI
redaction, specialist APIs, or synthesis — it's purely testing the router's
agent-selection judgment against the gold set's expected_agents labels.
"""

from evals.gold_set import GOLD_SET
from graph.nodes import router_node


def score_one(expected: set[str], actual: set[str]) -> dict:
    if not expected and not actual:
        return {"exact_match": True, "precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not actual or not expected:
        return {"exact_match": expected == actual, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    overlap = expected & actual
    precision = len(overlap) / len(actual)
    recall = len(overlap) / len(expected)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"exact_match": expected == actual, "precision": precision, "recall": recall, "f1": f1}


def evaluate_routing(gold_set: list[dict] | None = None, limit: int | None = None) -> dict:
    entries = gold_set if gold_set is not None else GOLD_SET
    if limit is not None:
        entries = entries[:limit]

    per_query = []
    for entry in entries:
        result = router_node({"redacted_query": entry["query"], "history": []})
        actual = set(result["routing"].keys())
        expected = set(entry["expected_agents"])
        scores = score_one(expected, actual)
        per_query.append({
            "id": entry["id"],
            "query": entry["query"],
            "category": entry["category"],
            "expected_agents": sorted(expected),
            "actual_agents": sorted(actual),
            **scores,
        })

    n = len(per_query) or 1
    return {
        "routing_accuracy": round(sum(p["exact_match"] for p in per_query) / n, 4),
        "routing_precision": round(sum(p["precision"] for p in per_query) / n, 4),
        "routing_recall": round(sum(p["recall"] for p in per_query) / n, 4),
        "routing_f1": round(sum(p["f1"] for p in per_query) / n, 4),
        "n_queries": len(per_query),
        "per_query": per_query,
    }
