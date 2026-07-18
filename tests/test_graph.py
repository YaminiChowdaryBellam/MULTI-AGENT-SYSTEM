"""
Tests for the LangGraph agentic core (graph/).

All Groq calls go through graph.llm.call_groq, but it's imported separately
into graph.nodes (classify/router/confidence_gate/reflect/synthesize) and
graph.guardrails (the input guard's out-of-scope check), so both call sites
need mocking. Every graph run passes through input_guard first, so its
out-of-scope check always fires — tests mock it to "in scope" unless they're
specifically testing guardrail behavior (see tests/test_guardrails.py).
Specialist agents are replaced wholesale via graph.nodes.AGENT_REGISTRY so no
real API/LLM calls happen inside them.
"""

import os
from unittest.mock import patch

os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "test-key")

from graph.build import build_graph  # noqa: E402

IN_SCOPE = '{"out_of_scope": false, "reason": ""}'


class TestClassifyShortCircuit:

    @patch("graph.guardrails.call_llm", return_value=IN_SCOPE)
    @patch("graph.nodes.call_llm")
    def test_conversational_query_skips_router_and_specialists(self, mock_call_groq, mock_guard_groq):
        mock_call_groq.return_value = '{"type": "conversational", "reply": "Hi there!"}'
        app = build_graph()
        result = app.invoke(
            {"query": "Hello!"}, config={"configurable": {"thread_id": "conv-1"}}
        )
        assert result["final_answer"] == "Hi there!"
        # Only classify's LLM call should have fired — router/confidence_gate/synthesize never ran.
        assert mock_call_groq.call_count == 1


class TestFanOutAndSynthesis:

    @patch("graph.nodes.AGENT_REGISTRY", {
        "pubmed": lambda q: f"pubmed answer for {q}",
        "tavily": lambda q: f"tavily answer for {q}",
    })
    @patch("graph.guardrails.call_llm", return_value=IN_SCOPE)
    @patch("graph.nodes.call_llm")
    def test_research_query_dispatches_in_parallel_and_synthesizes(self, mock_call_groq, mock_guard_groq):
        mock_call_groq.side_effect = [
            '{"type": "research"}',
            '{"pubmed": "afib treatment", "tavily": "afib news"}',
            '{"confidence": "high", "reason": "solid evidence"}',
            "Final synthesized answer [PUBMED][TAVILY].",
        ]
        app = build_graph()
        result = app.invoke(
            {"query": "afib treatment options"}, config={"configurable": {"thread_id": "research-1"}}
        )
        assert "Final synthesized answer" in result["final_answer"]
        assert "[PUBMED]" in result["final_answer"] and "[TAVILY]" in result["final_answer"]
        assert set(result["agent_outputs"].keys()) == {"pubmed", "tavily"}
        assert result["confidence"] == "high"


class TestEmptyRoutingSafety:
    """
    The router prompt tells the model an empty {} is a valid answer for a
    fully off-topic query. A zero-element Send list would leave nothing to
    advance the graph, so this must resolve safely (no crash, no hang) via
    confidence_gate's existing "no output" handling instead.
    """

    @patch("graph.guardrails.call_llm", return_value=IN_SCOPE)
    @patch("graph.nodes.call_llm")
    def test_zero_agents_selected_reaches_a_final_answer_without_crashing(
        self, mock_call_groq, mock_guard_groq
    ):
        mock_call_groq.side_effect = [
            '{"type": "research"}',
            '{}',                     # router finds nothing relevant
            '{}',                     # reflect also finds nothing relevant
            "I don't have enough information to answer this.",  # synthesize, run on empty evidence
        ]
        app = build_graph()
        result = app.invoke(
            {"query": "What is the weather today?"}, config={"configurable": {"thread_id": "empty-route-1"}}
        )
        assert result["routing"] == {}
        assert result["agent_outputs"] == {}
        # Still-low confidence after the exhausted retry means it's queued for review, not returned as-is.
        assert result["needs_review"] is True
        assert "flagged this question for a specialist's review" in result["final_answer"]


class TestReflectionLoop:

    @patch("graph.nodes.AGENT_REGISTRY", {"pubmed": lambda q: "weak or empty result"})
    @patch("graph.guardrails.call_llm", return_value=IN_SCOPE)
    @patch("graph.nodes.call_llm")
    def test_low_confidence_triggers_exactly_one_retry(self, mock_call_groq, mock_guard_groq):
        mock_call_groq.side_effect = [
            '{"type": "research"}',
            '{"pubmed": "vague query"}',
            '{"confidence": "low", "reason": "weak evidence"}',
            '{"pubmed": "more specific query"}',
            '{"confidence": "low", "reason": "still weak"}',
            "Final answer despite weak evidence [PUBMED].",
        ]
        app = build_graph()
        result = app.invoke(
            {"query": "vague clinical question"}, config={"configurable": {"thread_id": "reflect-1"}}
        )
        assert result["retry_count"] == 1
        # Still low confidence after the retry budget is exhausted -> withheld from the
        # user and flagged for human review instead (Step 4.3), not returned as-is.
        assert result["needs_review"] is True
        assert result["drafted_answer"] == "Final answer despite weak evidence [PUBMED]."
        assert "flagged this question for a specialist's review" in result["final_answer"]
        # classify, router, gate#1, reflect, gate#2, synthesize = 6 calls — never a 2nd reflect.
        assert mock_call_groq.call_count == 6

    @patch("graph.nodes.AGENT_REGISTRY", {"pubmed": lambda q: "strong result"})
    @patch("graph.guardrails.call_llm", return_value=IN_SCOPE)
    @patch("graph.nodes.call_llm")
    def test_high_confidence_skips_reflection(self, mock_call_groq, mock_guard_groq):
        mock_call_groq.side_effect = [
            '{"type": "research"}',
            '{"pubmed": "clear query"}',
            '{"confidence": "high", "reason": "solid"}',
            "Final answer [PUBMED].",
        ]
        app = build_graph()
        result = app.invoke(
            {"query": "clear clinical question"}, config={"configurable": {"thread_id": "reflect-2"}}
        )
        assert result["retry_count"] == 0
        assert mock_call_groq.call_count == 4


class TestSessionCheckpointing:

    @patch("graph.nodes.AGENT_REGISTRY", {
        "pubmed": lambda q: f"pubmed: {q}",
        "tavily": lambda q: f"tavily: {q}",
    })
    @patch("graph.guardrails.call_llm", return_value=IN_SCOPE)
    @patch("graph.nodes.call_llm")
    def test_followup_turn_does_not_leak_stale_agent_outputs(self, mock_call_groq, mock_guard_groq):
        mock_call_groq.side_effect = [
            # Turn 1 — routes to pubmed only
            '{"type": "research"}',
            '{"pubmed": "afib treatment"}',
            '{"confidence": "high", "reason": "ok"}',
            "Turn 1 answer [PUBMED].",
            # Turn 2 — follow-up, same thread, routes to tavily only
            '{"type": "research"}',
            '{"tavily": "afib latest news"}',
            '{"confidence": "high", "reason": "ok"}',
            "Turn 2 answer [TAVILY].",
        ]
        app = build_graph()
        config = {"configurable": {"thread_id": "session-1"}}

        result1 = app.invoke({"query": "afib treatment"}, config=config)
        assert set(result1["agent_outputs"].keys()) == {"pubmed"}

        result2 = app.invoke({"query": "any recent news on afib?"}, config=config)
        # The stale pubmed output from turn 1 must not leak into turn 2.
        assert set(result2["agent_outputs"].keys()) == {"tavily"}
        # But turn history should accumulate for follow-up context.
        assert len(result2["history"]) == 2
        assert "Turn 1 answer" in result2["history"][0]
        assert "Turn 2 answer" in result2["history"][1]
