"""
Step 4.2 — LLM-as-judge faithfulness + deterministic citation coverage,
end-to-end (full graph, real specialist + Groq calls) on the ~25 gold set
entries flagged e2e=True.

Calls graph.build.get_graph().invoke() directly rather than run_graph(), so
eval traffic never lands in logs/audit.jsonl or logs/review_queue.jsonl —
those are for real user requests only.
"""

from evals.gold_set import GOLD_SET
from graph.build import get_graph
from graph.guardrails import DISCLAIMER, citation_coverage
from graph.llm import call_groq, extract_json
from graph.nodes import AGENT_REGISTRY

KNOWN_SOURCES = {name.upper() for name in AGENT_REGISTRY}


def _judge_faithfulness(query: str, agent_outputs: dict, answer: str) -> dict:
    if not agent_outputs:
        return {"faithfulness": 0.0, "unsupported_claims": [], "notes": "No specialist evidence was gathered."}

    evidence = "\n\n".join(f"[{k.upper()}]\n{v}" for k, v in agent_outputs.items())
    prompt = (
        "You are grading whether a clinical assistant's answer is faithful to the evidence "
        "gathered by its specialist agents — NOT whether the answer is medically correct in "
        "an absolute sense, only whether it is supported by the evidence below.\n\n"
        f"Question: '{query}'\n\n"
        f"Evidence gathered by specialists:\n{evidence}\n\n"
        f"Answer to grade:\n{answer}\n\n"
        "Score what fraction of the answer's claims are supported by the evidence.\n"
        'Return ONLY JSON: {"faithfulness": <0.0-1.0>, "unsupported_claims": ["..."], "notes": "<one sentence>"}'
    )
    result = extract_json(call_groq(prompt)) or {}
    try:
        faithfulness = float(result.get("faithfulness", 0.0))
    except (TypeError, ValueError):
        faithfulness = 0.0
    return {
        "faithfulness": faithfulness,
        "unsupported_claims": result.get("unsupported_claims", []),
        "notes": result.get("notes", ""),
    }


def evaluate_judge(gold_set: list[dict] | None = None, limit: int | None = None) -> dict:
    entries = [e for e in (gold_set if gold_set is not None else GOLD_SET) if e["e2e"]]
    if limit is not None:
        entries = entries[:limit]

    app = get_graph()
    per_query = []
    for entry in entries:
        thread_id = f"eval-{entry['id']}"
        state = app.invoke({"query": entry["query"]}, config={"configurable": {"thread_id": thread_id}})
        # output_guard already appended DISCLAIMER to final_answer — it's a fixed
        # system message, not an LLM claim, so it must not count as an uncited one.
        answer = state.get("final_answer", "").replace(DISCLAIMER, "").strip()
        agent_outputs = state.get("agent_outputs", {})

        judged = _judge_faithfulness(entry["query"], agent_outputs, answer)
        coverage, coverage_stats = citation_coverage(answer, KNOWN_SOURCES)

        per_query.append({
            "id": entry["id"],
            "query": entry["query"],
            "category": entry["category"],
            "needs_review": state.get("needs_review", False),
            "confidence": state.get("confidence"),
            "faithfulness": judged["faithfulness"],
            "unsupported_claims": judged["unsupported_claims"],
            "citation_coverage": coverage,
            **coverage_stats,
        })

    n = len(per_query) or 1
    return {
        "faithfulness": round(sum(p["faithfulness"] for p in per_query) / n, 4),
        "citation_coverage": round(sum(p["citation_coverage"] for p in per_query) / n, 4),
        "review_rate": round(sum(p["needs_review"] for p in per_query) / n, 4),
        "n_queries": len(per_query),
        "per_query": per_query,
    }
