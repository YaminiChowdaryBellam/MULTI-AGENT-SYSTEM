"""
Tests for Phase 6.1's tiered model layer (graph/llm.py): call_llm's
cheap/expensive dispatch, and call_ollama's request handling + usage tracking.
"""

import os
from unittest.mock import MagicMock, patch

import httpx
import pytest
import requests
from groq import APIConnectionError, BadRequestError, RateLimitError

os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "test-key")

import graph.llm as llm  # noqa: E402


def _groq_response(content: str, prompt_tokens=10, completion_tokens=5) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens
    resp.usage = usage
    return resp


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


class TestCallGroqRetry:
    """
    First caught running in GitHub Actions (not locally): a plain
    groq.APIConnectionError with no rate limit involved crashed the eval
    harness, because call_groq only retried RateLimitError. Broadened to
    RETRYABLE_GROQ_ERRORS — verify each transient type is actually retried,
    and that a genuine client error (bad request) is NOT retried.
    """

    @patch("graph.llm.time.sleep")
    @patch("graph.llm._client")
    def test_retries_on_rate_limit_error(self, mock_client, mock_sleep):
        response = httpx.Response(429, request=httpx.Request("POST", "https://api.groq.com"))
        mock_client.chat.completions.create.side_effect = [
            RateLimitError("rate limited", response=response, body=None),
            _groq_response("ok"),
        ]
        result = llm.call_groq("prompt")
        assert result == "ok"
        assert mock_client.chat.completions.create.call_count == 2

    @patch("graph.llm.time.sleep")
    @patch("graph.llm._client")
    def test_retries_on_connection_error(self, mock_client, mock_sleep):
        request = httpx.Request("POST", "https://api.groq.com")
        mock_client.chat.completions.create.side_effect = [
            APIConnectionError(request=request),
            _groq_response("ok"),
        ]
        result = llm.call_groq("prompt")
        assert result == "ok"
        assert mock_client.chat.completions.create.call_count == 2

    @patch("graph.llm.time.sleep")
    @patch("graph.llm._client")
    def test_does_not_retry_non_transient_errors(self, mock_client, mock_sleep):
        response = httpx.Response(400, request=httpx.Request("POST", "https://api.groq.com"))
        mock_client.chat.completions.create.side_effect = BadRequestError(
            "bad request", response=response, body=None
        )
        with pytest.raises(BadRequestError):
            llm.call_groq("prompt")
        assert mock_client.chat.completions.create.call_count == 1
