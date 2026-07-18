"""
Tests for the Step 6.1 tiering benchmark harness. Mocks graph.llm.call_groq
and graph.llm.call_ollama directly with a content-inspecting responder (the
exact call sequence depends on branching graph logic — confidence, retries —
so a positional side_effect list would be too fragile to maintain here).
"""

import os
from unittest.mock import patch

os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "test-key")

from evals.tiering_benchmark import run_tiering_benchmark  # noqa: E402

IN_SCOPE = '{"out_of_scope": false, "reason": ""}'


def _fake_llm_responder(prompt: str, retries: int = 3, timeout: int = 60) -> str:
    if "scope classifier" in prompt:
        return IN_SCOPE
    if "classifier for a clinical research" in prompt:
        return '{"type": "research"}'
    if "routing assistant" in prompt:
        return '{"pubmed": "warfarin"}'
    if "evidence-quality reviewer" in prompt:
        return '{"confidence": "high", "reason": "ok"}'
    if "helpful clinical research assistant" in prompt:
        return "Warfarin is an anticoagulant [PUBMED]."
    return "{}"


class TestTieringBenchmark:

    @patch("graph.nodes.AGENT_REGISTRY", {"pubmed": lambda q: "warfarin is an anticoagulant"})
    @patch("graph.llm.call_ollama", side_effect=_fake_llm_responder)
    @patch("graph.llm.call_groq", side_effect=_fake_llm_responder)
    def test_before_uses_only_groq_after_uses_ollama_for_cheap_calls(self, mock_groq, mock_ollama):
        gold_set = [{
            "id": "T1", "category": "pubmed_only", "e2e": True,
            "query": "What does the literature say about warfarin?",
            "expected_agents": ["pubmed"],
        }]

        result = run_tiering_benchmark(gold_set=gold_set, limit=1)

        assert result["before_tiering"]["n_queries"] == 1
        assert result["after_tiering"]["n_queries"] == 1

        # "before" (TIERED_MODELS off): every cheap+expensive call goes to Groq.
        assert result["before_tiering"]["ollama_calls"] == 0
        assert result["before_tiering"]["groq_calls"] >= 4  # scope, classify, router, confidence_gate, synthesize

        # "after" (TIERED_MODELS on): only synthesize should still hit Groq.
        assert result["after_tiering"]["groq_calls"] == 1
        assert result["after_tiering"]["ollama_calls"] >= 3
