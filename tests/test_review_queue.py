"""
Tests for graph/review_queue.py and the human_review_gate node's wiring into
run_graph(): a drafted answer that's still low-confidence after the
reflection retry is exhausted must never reach the user directly — it's
queued for human review instead.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "test-key")

from graph.review_queue import read_review_queue, write_review_record  # noqa: E402

IN_SCOPE = '{"out_of_scope": false, "reason": ""}'


class TestReviewQueue:

    def test_write_and_read_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "review_queue.jsonl"
            with patch("graph.review_queue.QUEUE_PATH", path):
                state = {
                    "query": "vague clinical question",
                    "redacted_query": "vague clinical question",
                    "routing": {"pubmed": "vague query"},
                    "agent_outputs": {"pubmed": "weak or empty result"},
                    "confidence": "low",
                    "confidence_reason": "still weak",
                    "retry_count": 1,
                    "drafted_answer": "Final answer despite weak evidence [PUBMED].",
                    "final_answer": "flagged for review placeholder",
                }
                write_review_record(state, thread_id="t1")

            records = read_review_queue(path)
            assert len(records) == 1
            assert records[0]["status"] == "pending"
            assert records[0]["drafted_answer"] == "Final answer despite weak evidence [PUBMED]."
            assert records[0]["confidence"] == "low"
            assert records[0]["thread_id"] == "t1"


class TestRunGraphReviewQueueWiring:

    @patch("graph.nodes.AGENT_REGISTRY", {"pubmed": lambda q: "weak or empty result"})
    @patch("graph.guardrails.call_llm", return_value=IN_SCOPE)
    @patch("graph.nodes.call_llm")
    def test_low_confidence_answer_is_queued_not_returned(self, mock_call_groq, mock_guard_groq):
        mock_call_groq.side_effect = [
            '{"type": "research"}',
            '{"pubmed": "vague query"}',
            '{"confidence": "low", "reason": "weak evidence"}',
            '{"pubmed": "more specific query"}',
            '{"confidence": "low", "reason": "still weak"}',
            "Final answer despite weak evidence [PUBMED].",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            queue_path = Path(tmp) / "review_queue.jsonl"
            audit_path = Path(tmp) / "audit.jsonl"
            with patch("graph.review_queue.QUEUE_PATH", queue_path), \
                 patch("graph.audit.LOG_PATH", audit_path):
                from graph.build import run_graph
                answer = run_graph("vague clinical question", thread_id="review-1")

            queued = read_review_queue(queue_path)

        assert "flagged this question for a specialist's review" in answer
        assert "Final answer despite weak evidence" not in answer
        assert len(queued) == 1
        assert queued[0]["drafted_answer"] == "Final answer despite weak evidence [PUBMED]."
        assert queued[0]["thread_id"] == "review-1"
