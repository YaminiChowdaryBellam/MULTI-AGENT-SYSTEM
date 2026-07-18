"""
Tests for the Phase 4 evaluation harness (evals/). All Groq calls are
mocked — these tests validate the harness's scoring/aggregation logic, not
model quality itself (that's what actually running `python -m evals`
against live APIs is for).
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "test-key")

from evals.results import compare, latest_results, write_results  # noqa: E402
from evals.routing_eval import evaluate_routing  # noqa: E402

IN_SCOPE = '{"out_of_scope": false, "reason": ""}'

MINI_GOLD_SET = [
    {"id": "T1", "category": "pubmed_only", "e2e": True,
     "query": "What does the literature say about warfarin?", "expected_agents": ["pubmed"]},
    {"id": "T2", "category": "tavily_only", "e2e": True,
     "query": "What's the latest drug recall news?", "expected_agents": ["tavily"]},
    {"id": "T3", "category": "edge_case", "e2e": False,
     "query": "What's the weather?", "expected_agents": []},
]


# ---------------------------------------------------------------------------
# Routing eval
# ---------------------------------------------------------------------------

class TestRoutingEval:

    @patch("graph.nodes.call_llm")
    def test_exact_match_scores_perfectly(self, mock_call_groq):
        mock_call_groq.side_effect = [
            '{"pubmed": "warfarin"}',
            '{"tavily": "drug recall news"}',
            '{}',
        ]
        result = evaluate_routing(gold_set=MINI_GOLD_SET)
        assert result["n_queries"] == 3
        # T1, T2, and T3 (a validly parsed, deliberately empty {}) all match exactly.
        assert result["routing_accuracy"] == 1.0

    @patch("graph.nodes.call_llm")
    def test_unparseable_response_falls_back_to_all_agents(self, mock_call_groq):
        """Genuine parse failure (not a valid {}) should still trigger the safety net."""
        mock_call_groq.return_value = "I cannot decide which agents to use."
        gold_set = [{"id": "T1", "category": "c", "e2e": False, "query": "q", "expected_agents": []}]
        result = evaluate_routing(gold_set=gold_set)
        from agents.orchestrator import AGENT_REGISTRY
        assert set(result["per_query"][0]["actual_agents"]) == set(AGENT_REGISTRY.keys())

    @patch("graph.nodes.call_llm")
    def test_partial_overlap_scores_between_zero_and_one(self, mock_call_groq):
        mock_call_groq.return_value = '{"pubmed": "x", "openfda": "y"}'
        gold_set = [{"id": "T1", "category": "c", "e2e": False,
                     "query": "q", "expected_agents": ["pubmed", "tavily"]}]
        result = evaluate_routing(gold_set=gold_set)
        p = result["per_query"][0]
        assert p["exact_match"] is False
        assert 0 < p["precision"] < 1
        assert 0 < p["recall"] < 1

    @patch("graph.nodes.call_llm")
    def test_limit_truncates_gold_set(self, mock_call_groq):
        mock_call_groq.return_value = '{"pubmed": "x"}'
        result = evaluate_routing(gold_set=MINI_GOLD_SET, limit=1)
        assert result["n_queries"] == 1


# ---------------------------------------------------------------------------
# Judge eval
# ---------------------------------------------------------------------------

class TestJudgeEval:

    @patch("evals.judge_eval.call_groq")
    @patch("graph.nodes.AGENT_REGISTRY", {"pubmed": lambda q: "warfarin is an anticoagulant"})
    @patch("graph.guardrails.call_llm", return_value=IN_SCOPE)
    @patch("graph.nodes.call_llm")
    def test_faithfulness_and_coverage_are_computed(
        self, mock_nodes_groq, mock_guard_groq, mock_judge_groq
    ):
        mock_nodes_groq.side_effect = [
            '{"type": "research"}',
            '{"pubmed": "warfarin"}',
            '{"confidence": "high", "reason": "solid"}',
            "Warfarin is an anticoagulant used to prevent blood clots [PUBMED].",
        ]
        mock_judge_groq.return_value = '{"faithfulness": 0.9, "unsupported_claims": [], "notes": "well supported"}'

        from evals.judge_eval import evaluate_judge
        gold_set = [{"id": "T1", "category": "pubmed_only", "e2e": True,
                     "query": "What is warfarin?", "expected_agents": ["pubmed"]}]
        result = evaluate_judge(gold_set=gold_set)

        assert result["n_queries"] == 1
        assert result["faithfulness"] == 0.9
        assert result["citation_coverage"] == 1.0
        assert result["review_rate"] == 0.0

    @patch("evals.judge_eval.call_groq")
    @patch("graph.nodes.AGENT_REGISTRY", {"pubmed": lambda q: "weak result"})
    @patch("graph.guardrails.call_llm", return_value=IN_SCOPE)
    @patch("graph.nodes.call_llm")
    def test_needs_review_is_reported(self, mock_nodes_groq, mock_guard_groq, mock_judge_groq):
        mock_nodes_groq.side_effect = [
            '{"type": "research"}',
            '{"pubmed": "vague"}',
            '{"confidence": "low", "reason": "weak"}',
            '{"pubmed": "vague 2"}',
            '{"confidence": "low", "reason": "still weak"}',
            "Draft answer [PUBMED].",
        ]
        mock_judge_groq.return_value = '{"faithfulness": 0.5, "unsupported_claims": [], "notes": ""}'

        from evals.judge_eval import evaluate_judge
        gold_set = [{"id": "T2", "category": "pubmed_only", "e2e": True,
                     "query": "vague clinical question", "expected_agents": ["pubmed"]}]
        result = evaluate_judge(gold_set=gold_set)

        assert result["review_rate"] == 1.0
        assert result["per_query"][0]["needs_review"] is True

    @patch("evals.judge_eval.call_groq", return_value='{"faithfulness": 1.0, "unsupported_claims": [], "notes": ""}')
    def test_only_e2e_flagged_entries_are_evaluated(self, mock_judge_groq):
        from evals.judge_eval import evaluate_judge
        # None of MINI_GOLD_SET's non-e2e entry should ever reach the graph.
        with patch("graph.guardrails.call_llm", return_value=IN_SCOPE), \
             patch("graph.nodes.AGENT_REGISTRY", {"pubmed": lambda q: "x"}), \
             patch("graph.nodes.call_llm", side_effect=[
                 '{"type": "research"}', '{"pubmed": "x"}',
                 '{"confidence": "high", "reason": "ok"}', "Answer [PUBMED].",
                 '{"type": "research"}', '{"tavily": "x"}',
                 '{"confidence": "high", "reason": "ok"}', "Answer [TAVILY].",
             ]):
            result = evaluate_judge(gold_set=MINI_GOLD_SET)
        assert result["n_queries"] == 2  # only T1 and T2 are e2e=True


# ---------------------------------------------------------------------------
# Results tracking / regression comparison
# ---------------------------------------------------------------------------

class TestResultsTracking:

    def test_write_and_read_latest(self):
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            write_results({"routing_accuracy": 0.8}, results_dir=results_dir)
            path2 = write_results({"routing_accuracy": 0.9}, results_dir=results_dir)
            latest = latest_results(results_dir=results_dir, exclude=None)
            assert latest["routing_accuracy"] == 0.9
            excluded_latest = latest_results(results_dir=results_dir, exclude=path2)
            assert excluded_latest["routing_accuracy"] == 0.8

    def test_no_previous_results_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            assert latest_results(results_dir=Path(tmp)) is None

    def test_compare_flags_regression(self):
        current = {"faithfulness": 0.70}
        previous = {"faithfulness": 0.90}
        comparison = compare(current, previous, tolerance=0.02)
        assert comparison["faithfulness"]["regressed"] is True
        assert comparison["faithfulness"]["delta"] == -0.2

    def test_compare_within_tolerance_is_not_a_regression(self):
        current = {"faithfulness": 0.89}
        previous = {"faithfulness": 0.90}
        comparison = compare(current, previous, tolerance=0.02)
        assert comparison["faithfulness"]["regressed"] is False

    def test_compare_lower_is_better_metric(self):
        current = {"review_rate": 0.30}
        previous = {"review_rate": 0.10}
        comparison = compare(current, previous, tolerance=0.02)
        # review_rate going UP is the regression, not down.
        assert comparison["review_rate"]["regressed"] is True

    def test_compare_with_no_previous_returns_empty(self):
        assert compare({"routing_accuracy": 0.8}, None) == {}


# ---------------------------------------------------------------------------
# CLI exit code (this is what makes CI actually block a regressing PR)
# ---------------------------------------------------------------------------

class TestMainExitCode:

    @patch("evals.__main__.write_results")
    @patch("evals.__main__.latest_results")
    @patch("evals.__main__.evaluate_routing")
    def test_regressing_run_exits_nonzero(self, mock_routing, mock_latest, mock_write):
        mock_routing.return_value = {
            "routing_accuracy": 0.50, "routing_precision": 0.5, "routing_recall": 0.5,
            "routing_f1": 0.5, "n_queries": 10, "per_query": [],
        }
        mock_latest.return_value = {"routing_accuracy": 0.90, "timestamp": "t", "git_sha": "abc"}

        from evals.__main__ import main
        exit_code = main(["--skip-judge", "--no-write"])

        assert exit_code == 1
        mock_write.assert_not_called()  # --no-write means no new baseline gets committed

    @patch("evals.__main__.write_results")
    @patch("evals.__main__.latest_results")
    @patch("evals.__main__.evaluate_routing")
    def test_improving_run_exits_zero_and_writes_results(self, mock_routing, mock_latest, mock_write):
        mock_routing.return_value = {
            "routing_accuracy": 0.90, "routing_precision": 0.9, "routing_recall": 0.9,
            "routing_f1": 0.9, "n_queries": 10, "per_query": [],
        }
        mock_latest.return_value = {"routing_accuracy": 0.50, "timestamp": "t", "git_sha": "abc"}
        mock_write.return_value = Path("/tmp/fake_results.json")

        from evals.__main__ import main
        exit_code = main(["--skip-judge"])

        assert exit_code == 0
        mock_write.assert_called_once()

    @patch("evals.__main__.latest_results", return_value=None)
    @patch("evals.__main__.evaluate_routing")
    def test_baseline_run_with_no_prior_results_exits_zero(self, mock_routing, mock_latest):
        mock_routing.return_value = {
            "routing_accuracy": 0.5, "routing_precision": 0.5, "routing_recall": 0.5,
            "routing_f1": 0.5, "n_queries": 10, "per_query": [],
        }
        from evals.__main__ import main
        exit_code = main(["--skip-judge", "--no-write"])
        assert exit_code == 0
