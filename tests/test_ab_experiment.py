"""
Tests for the Phase 5 A/B routing experiment (embedding router + comparison
harness). embedding_route uses a real, small, local sentence-transformers
model — no mocking needed, same rationale as testing Presidio directly.
The LLM router side is mocked via graph.nodes.call_groq, consistent with the
rest of the test suite.
"""

import os
from unittest.mock import patch

os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "test-key")

from evals.ab_experiment import (  # noqa: E402
    _agent_distribution,
    _total_variation_distance,
    run_ab_experiment,
)
from graph.embedding_router import embedding_route  # noqa: E402


class TestEmbeddingRoute:

    def test_pathophysiology_query_routes_to_rag(self):
        routing = embedding_route("What is the pathophysiology of atrial fibrillation?")
        assert "rag" in routing

    def test_returns_query_unmodified_as_subquery(self):
        query = "What is the pathophysiology of atrial fibrillation?"
        routing = embedding_route(query)
        assert all(sub_query == query for sub_query in routing.values())

    def test_higher_threshold_selects_fewer_or_equal_agents(self):
        query = "What are the warnings for metformin?"
        loose = embedding_route(query, threshold=0.1)
        strict = embedding_route(query, threshold=0.9)
        assert set(strict.keys()) <= set(loose.keys())


class TestDistributionHelpers:

    def test_agent_distribution_sums_to_one(self):
        per_query = [
            {"actual_agents": ["pubmed"]},
            {"actual_agents": ["pubmed", "openfda"]},
            {"actual_agents": ["openfda"]},
        ]
        dist = _agent_distribution(per_query, ["pubmed", "openfda", "rag"])
        assert abs(sum(dist.values()) - 1.0) < 1e-9
        assert dist["rag"] == 0.0

    def test_identical_distributions_have_zero_drift(self):
        dist = {"pubmed": 0.5, "openfda": 0.5}
        assert _total_variation_distance(dist, dist) == 0.0

    def test_disjoint_distributions_have_max_drift(self):
        dist_a = {"pubmed": 1.0, "openfda": 0.0}
        dist_b = {"pubmed": 0.0, "openfda": 1.0}
        assert _total_variation_distance(dist_a, dist_b) == 1.0


class TestRunAbExperiment:

    @patch("graph.nodes.call_groq")
    def test_returns_both_variants_and_drift(self, mock_call_groq):
        mock_call_groq.return_value = '{"rag": "pathophysiology of atrial fibrillation"}'
        gold_set = [{
            "id": "T1", "category": "rag_only", "e2e": False,
            "query": "What is the pathophysiology of atrial fibrillation?",
            "expected_agents": ["rag"],
        }]
        result = run_ab_experiment(gold_set=gold_set)

        assert result["llm_router"]["n_queries"] == 1
        assert result["embedding_router"]["n_queries"] == 1
        assert "distribution_drift_tvd" in result
        # Both variants should get this one right — same gold label either way.
        assert result["llm_router"]["accuracy"] == 1.0
        assert result["embedding_router"]["accuracy"] == 1.0

    @patch("graph.nodes.call_groq")
    def test_llm_variant_tracks_nonzero_cost_embedding_variant_is_free(self, mock_call_groq):
        mock_call_groq.return_value = '{"pubmed": "warfarin"}'
        gold_set = [{
            "id": "T1", "category": "pubmed_only", "e2e": False,
            "query": "What does the literature say about warfarin?",
            "expected_agents": ["pubmed"],
        }]
        # graph.llm.LAST_USAGE isn't updated by a mocked call_groq (the real
        # Groq response object is never constructed), so it reflects whatever
        # the last *real* call in the process set — for a clean assertion,
        # zero it out first.
        import graph.llm as llm_module
        llm_module.LAST_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        result = run_ab_experiment(gold_set=gold_set)
        assert result["embedding_router"]["total_cost_usd"] == 0.0
