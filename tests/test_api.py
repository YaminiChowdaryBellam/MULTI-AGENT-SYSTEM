"""
Tests for the FastAPI service (api.py). Mocks api.run_graph so these run
offline — TestClient/starlette handles the HTTP layer.
"""

import os
from unittest.mock import patch

os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "test-key")

from fastapi.testclient import TestClient  # noqa: E402

from api import app  # noqa: E402

client = TestClient(app)
# TestClient re-raises unhandled app exceptions into the test by default, which
# would bypass api.py's registered Exception handler — disable that so this
# test actually observes the HTTP 500 response a real client would get.
client_no_raise = TestClient(app, raise_server_exceptions=False)


class TestHealth:

    def test_health_returns_ok(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok", "engine": "langgraph"}


class TestQueryEndpoint:

    @patch("api.run_graph")
    def test_query_returns_answer(self, mock_run_graph):
        mock_run_graph.return_value = "Warfarin is an anticoagulant [OPENFDA]."
        resp = client.post("/query", json={"query": "What is warfarin?"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["query"] == "What is warfarin?"
        assert body["answer"] == "Warfarin is an anticoagulant [OPENFDA]."
        assert "thread_id" in body
        assert body["latency_ms"] >= 0

    @patch("api.run_graph")
    def test_omitting_thread_id_generates_a_new_one(self, mock_run_graph):
        mock_run_graph.return_value = "answer"
        resp1 = client.post("/query", json={"query": "What is warfarin?"})
        resp2 = client.post("/query", json={"query": "What is warfarin?"})
        assert resp1.json()["thread_id"] != resp2.json()["thread_id"]

    @patch("api.run_graph")
    def test_provided_thread_id_is_reused(self, mock_run_graph):
        mock_run_graph.return_value = "answer"
        resp = client.post("/query", json={"query": "follow-up question", "thread_id": "session-abc"})
        assert resp.json()["thread_id"] == "session-abc"
        mock_run_graph.assert_called_with("follow-up question", thread_id="session-abc")

    def test_empty_query_is_rejected(self):
        resp = client.post("/query", json={"query": ""})
        assert resp.status_code == 422

    def test_missing_query_field_is_rejected(self):
        resp = client.post("/query", json={})
        assert resp.status_code == 422

    def test_overlong_query_is_rejected(self):
        resp = client.post("/query", json={"query": "a" * 1001})
        assert resp.status_code == 422

    @patch("api.run_graph")
    def test_graph_exception_returns_500_not_a_crash(self, mock_run_graph):
        mock_run_graph.side_effect = RuntimeError("boom")
        resp = client_no_raise.post("/query", json={"query": "What is warfarin?"})
        assert resp.status_code == 500
        assert "boom" in resp.json()["detail"]
