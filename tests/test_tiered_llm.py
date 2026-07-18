"""
Tests for Phase 6.1's tiered model layer (graph/llm.py): call_llm's
cheap/expensive dispatch, and call_ollama's request handling + usage tracking.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
import requests

os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "test-key")

import graph.llm as llm  # noqa: E402


class TestCallLlmDispatch:

    @patch("graph.llm.call_ollama")
    @patch("graph.llm.call_groq")
    def test_cheap_tier_uses_groq_when_tiering_disabled(self, mock_groq, mock_ollama):
        with patch("graph.llm.TIERED_MODELS_ENABLED", False):
            mock_groq.return_value = "groq answer"
            result = llm.call_llm("prompt", tier="cheap")
        assert result == "groq answer"
        mock_groq.assert_called_once()
        mock_ollama.assert_not_called()

    @patch("graph.llm.call_ollama")
    @patch("graph.llm.call_groq")
    def test_cheap_tier_uses_ollama_when_tiering_enabled(self, mock_groq, mock_ollama):
        with patch("graph.llm.TIERED_MODELS_ENABLED", True):
            mock_ollama.return_value = "ollama answer"
            result = llm.call_llm("prompt", tier="cheap")
        assert result == "ollama answer"
        mock_ollama.assert_called_once()
        mock_groq.assert_not_called()

    @patch("graph.llm.call_ollama")
    @patch("graph.llm.call_groq")
    def test_expensive_tier_always_uses_groq(self, mock_groq, mock_ollama):
        mock_groq.return_value = "groq answer"
        for tiering in (True, False):
            with patch("graph.llm.TIERED_MODELS_ENABLED", tiering):
                result = llm.call_llm("prompt", tier="expensive")
            assert result == "groq answer"
        mock_ollama.assert_not_called()


class TestCallOllama:

    @patch("graph.llm.requests.post")
    def test_returns_content_and_updates_last_usage(self, mock_post):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {
            "message": {"content": '{"type": "research"}'},
            "prompt_eval_count": 42,
            "eval_count": 8,
        }
        mock_post.return_value = resp

        result = llm.call_ollama("some prompt")

        assert result == '{"type": "research"}'
        assert llm.LAST_USAGE["backend"] == "ollama"
        assert llm.LAST_USAGE["prompt_tokens"] == 42
        assert llm.LAST_USAGE["completion_tokens"] == 8
        assert llm.LAST_USAGE["total_tokens"] == 50
        assert llm.LAST_USAGE["cost_usd"] == 0.0

    @patch("graph.llm.requests.post")
    def test_posts_to_configured_host_and_model(self, mock_post):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"message": {"content": "ok"}, "prompt_eval_count": 1, "eval_count": 1}
        mock_post.return_value = resp

        llm.call_ollama("hello")

        call_kwargs = mock_post.call_args
        assert call_kwargs.args[0] == f"{llm.OLLAMA_HOST}/api/chat"
        assert call_kwargs.kwargs["json"]["model"] == llm.OLLAMA_MODEL
        assert call_kwargs.kwargs["json"]["stream"] is False

    @patch("graph.llm.time.sleep")
    @patch("graph.llm.requests.post")
    def test_retries_and_raises_after_exhausting_attempts(self, mock_post, mock_sleep):
        mock_post.side_effect = requests.RequestException("connection refused")
        with pytest.raises(requests.RequestException):
            llm.call_ollama("prompt", retries=2)
        assert mock_post.call_count == 2
