"""
Human-in-the-loop review queue: when the confidence gate is still "low"
after the reflection retry is exhausted, the drafted answer isn't returned
to the user — it's queued here for a human reviewer instead.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

QUEUE_PATH = Path(__file__).resolve().parent.parent / "logs" / "review_queue.jsonl"


def write_review_record(state: dict, thread_id: str) -> dict:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "thread_id": thread_id,
        "query": state.get("query", ""),
        "redacted_query": state.get("redacted_query", ""),
        "routing": state.get("routing", {}),
        "agent_outputs": state.get("agent_outputs", {}),
        "confidence": state.get("confidence"),
        "confidence_reason": state.get("confidence_reason", ""),
        "retry_count": state.get("retry_count", 0),
        "drafted_answer": state.get("drafted_answer") or state.get("final_answer", ""),
        "status": "pending",
    }
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with QUEUE_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def read_review_queue(path: Path = QUEUE_PATH) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]
