import sys
from types import SimpleNamespace

from truthfulqa_bench.conditions import RunCondition
from truthfulqa_bench.prompting import BinaryPrompt
from truthfulqa_bench.providers import (
    OpenAIClient,
    api_error_summary,
    call_with_retries,
    openai_max_output_tokens,
    openai_min_request_interval_seconds,
    retry_after_from_message,
)


class FakeResponses:
    def __init__(self):
        self.request = None

    def create(self, **request):
        self.request = request
        return SimpleNamespace(
            output_text="A",
            usage=SimpleNamespace(input_tokens=100, output_tokens=1),
        )


class FakeOpenAI:
    latest = None

    def __init__(self):
        self.responses = FakeResponses()
        FakeOpenAI.latest = self


def test_openai_client_sends_reasoning_effort_without_tools(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    condition = RunCondition(
        condition_id="openai_gpt-5.5_high",
        provider="openai",
        model_id="gpt-5.5",
        model_label="GPT-5.5 (high)",
        reasoning_effort="high",
    )
    client = OpenAIClient(condition=condition, question_set_id="test-set")

    result = client.complete_choice(
        BinaryPrompt(row_id=1, order="correct_first", prompt="Question?\n\nA. Yes\n\nB. No", correct_choice="A"),
        "Health",
    )

    request = FakeOpenAI.latest.responses.request
    assert request["model"] == "gpt-5.5"
    assert request["max_output_tokens"] == 512
    assert request["reasoning"] == {"effort": "high"}
    assert "temperature" not in request
    assert "tools" not in request
    assert "web_search" not in request
    assert result.condition_id == "openai_gpt-5.5_high"
    assert result.reasoning_effort == "high"


def test_call_with_retries_uses_retry_after_message(monkeypatch):
    sleeps = []
    attempts = {"count": 0}

    class RateLimit(Exception):
        status_code = 429

    def flaky_call():
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RateLimit("Please try again in 20s.")
        return "ok"

    monkeypatch.setattr("truthfulqa_bench.providers.time.sleep", sleeps.append)

    assert call_with_retries(flaky_call) == "ok"
    assert attempts["count"] == 2
    assert sleeps == [60.0]


def test_retry_after_from_message_parses_minutes_and_seconds():
    assert retry_after_from_message("Please try again in 7m12s.") == 432.0


def test_call_with_retries_uses_longer_default_for_rate_limits(monkeypatch):
    sleeps = []
    attempts = {"count": 0}

    class RateLimit(Exception):
        status_code = 429

    def flaky_call():
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RateLimit("Rate limit reached.")
        return "ok"

    monkeypatch.setattr("truthfulqa_bench.providers.time.sleep", sleeps.append)

    assert call_with_retries(flaky_call) == "ok"
    assert sleeps == [60.0]


def test_call_with_retries_retries_connection_errors(monkeypatch):
    sleeps = []
    attempts = {"count": 0}

    class APIConnectionError(Exception):
        pass

    def flaky_call():
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise APIConnectionError("Connection error.")
        return "ok"

    monkeypatch.setattr("truthfulqa_bench.providers.time.sleep", sleeps.append)

    assert call_with_retries(flaky_call) == "ok"
    assert attempts["count"] == 2
    assert sleeps == [20.0]


def test_call_with_retries_runs_before_attempt_for_retries(monkeypatch):
    sleeps = []
    attempts = {"count": 0}
    before_attempts = {"count": 0}

    class RateLimit(Exception):
        status_code = 429

    def flaky_call():
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RateLimit("Rate limit reached.")
        return "ok"

    def before_attempt():
        before_attempts["count"] += 1

    monkeypatch.setattr("truthfulqa_bench.providers.time.sleep", sleeps.append)

    assert call_with_retries(flaky_call, before_attempt=before_attempt) == "ok"
    assert attempts["count"] == 2
    assert before_attempts["count"] == 2
    assert sleeps == [60.0]


def test_api_error_summary_includes_status_request_id_and_single_line_message():
    class RateLimit(Exception):
        status_code = 429
        request_id = "req_123"

    summary = api_error_summary(RateLimit("Rate limit reached.\nPlease try again in 60s."))

    assert summary == "RateLimit status=429 request_id=req_123 message=Rate limit reached. Please try again in 60s."


def test_openai_client_paces_requests(monkeypatch):
    sleeps = []
    times = iter([105.0, 126.0])
    monkeypatch.setattr("truthfulqa_bench.providers.monotonic", lambda: next(times))
    monkeypatch.setattr("truthfulqa_bench.providers.time.sleep", sleeps.append)
    OpenAIClient._last_request_started_at = 100.0

    OpenAIClient._pace_request()

    assert sleeps == [16.0]
    assert OpenAIClient._last_request_started_at == 126.0


def test_openai_request_interval_can_be_disabled_from_environment(monkeypatch):
    sleeps = []
    monkeypatch.setenv("TRUTHFULQA_OPENAI_MIN_REQUEST_INTERVAL_SECONDS", "0")
    monkeypatch.setattr("truthfulqa_bench.providers.time.sleep", sleeps.append)
    OpenAIClient._last_request_started_at = 100.0

    assert openai_min_request_interval_seconds() == 0.0
    OpenAIClient._pace_request()

    assert sleeps == []
    assert OpenAIClient._last_request_started_at == 100.0


def test_openai_max_output_tokens_can_be_set_from_environment(monkeypatch):
    monkeypatch.setenv("TRUTHFULQA_OPENAI_MAX_OUTPUT_TOKENS", "8192")

    assert openai_max_output_tokens() == 8192
