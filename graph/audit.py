"""
Audit trail: a timestamped entry per graph node event (appended to
GraphState["audit"] as the run progresses), plus one JSONL line per *request*
written after the graph finishes, replayable end-to-end from raw query
through redactions, routing, agent outputs, retries, and the final answer.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "audit.jsonl"


def audit_entry(node: str, **fields) -> dict:
    return {"node": node, "timestamp": datetime.now(timezone.utc).isoformat(), **fields}


def write_audit_record(state: dict, thread_id: str) -> dict:
    """Assembles and appends the full per-request record; returns it (useful for tests/demos)."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "thread_id": thread_id,
        "query": state.get("query", ""),
        "redacted_query": state.get("redacted_query", ""),
        "phi_redactions": state.get("phi_redactions", []),
        "injection_flags": state.get("injection_flags", []),
        "refused": state.get("is_refused", False),
        "conversational": state.get("is_conversational", False),
        "routing": state.get("routing", {}),
        "agent_outputs": state.get("agent_outputs", {}),
        "retry_count": state.get("retry_count", 0),
        "confidence": state.get("confidence"),
        "final_answer": state.get("final_answer", ""),
        "trace": state.get("audit", []),
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def read_audit_log(path: Path = LOG_PATH) -> list[dict]:
    """Replays the JSONL log back into a list of request records, oldest first."""
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]
