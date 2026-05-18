from __future__ import annotations

import os
import re
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic
from typing import Protocol

from .conditions import RunCondition
from .costs import estimate_cost
from .prompting import SYSTEM_PROMPT, BinaryPrompt, parse_choice
from .results import BenchmarkResult


class ProviderClient(Protocol):
    provider: str
    condition: RunCondition

    def complete_choice(self, prompt: BinaryPrompt, category: str) -> BenchmarkResult:
        ...


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int


MAX_API_ATTEMPTS = 20
DEFAULT_RETRY_SECONDS = 20.0
DEFAULT_RATE_LIMIT_RETRY_SECONDS = 60.0
DEFAULT_OPENAI_MIN_REQUEST_INTERVAL_SECONDS = 21.0
TRANSIENT_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


def require_api_key(provider: str) -> str:
    env_name = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
    }[provider]
    api_key = os.environ.get(env_name)
    if not api_key:
        raise RuntimeError(f"{env_name} is required before running paid {provider} benchmark calls.")
    return api_key


class AnthropicClient:
    provider = "anthropic"

    def __init__(self, condition: RunCondition, question_set_id: str, max_tokens: int = 16):
        require_api_key(self.provider)
        from anthropic import Anthropic

        self.condition = condition
        self.question_set_id = question_set_id
        self.max_tokens = max_tokens
        self._client = Anthropic()

    def complete_choice(self, prompt: BinaryPrompt, category: str) -> BenchmarkResult:
        start = monotonic()
        response = call_with_retries(
            lambda: self._client.messages.create(
                model=self.condition.model_id,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt.prompt}],
            )
        )
        latency = monotonic() - start
        raw_output = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
        usage = Usage(input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens)
        return _to_result(self.condition, self.question_set_id, prompt, category, raw_output, usage, latency)


class OpenAIClient:
    provider = "openai"
    _last_request_started_at = 0.0
    _pace_lock = threading.Lock()

    def __init__(self, condition: RunCondition, question_set_id: str, max_output_tokens: int | None = None):
        require_api_key(self.provider)
        from openai import OpenAI

        self.condition = condition
        self.question_set_id = question_set_id
        self.max_output_tokens = max_output_tokens or openai_max_output_tokens()
        self._client = OpenAI()

    def complete_choice(self, prompt: BinaryPrompt, category: str) -> BenchmarkResult:
        start = monotonic()
        request = {
            "model": self.condition.model_id,
            "max_output_tokens": self.max_output_tokens,
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt.prompt},
            ],
        }
        if self.condition.reasoning_effort is not None:
            request["reasoning"] = {"effort": self.condition.reasoning_effort}
        response = call_with_retries(
            lambda: self._client.responses.create(**request),
            before_attempt=self._pace_request,
        )
        latency = monotonic() - start
        raw_output = response.output_text
        usage = Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        return _to_result(self.condition, self.question_set_id, prompt, category, raw_output, usage, latency)

    @classmethod
    def _pace_request(cls) -> None:
        interval = openai_min_request_interval_seconds()
        if interval <= 0:
            return
        with cls._pace_lock:
            elapsed = monotonic() - cls._last_request_started_at
            if elapsed < interval:
                time.sleep(interval - elapsed)
            cls._last_request_started_at = monotonic()


def make_client(condition: RunCondition, question_set_id: str) -> ProviderClient:
    if condition.provider == "anthropic":
        return AnthropicClient(condition, question_set_id)
    if condition.provider == "openai":
        return OpenAIClient(condition, question_set_id)
    raise ValueError(f"Unsupported provider: {condition.provider}")


def openai_min_request_interval_seconds() -> float:
    raw = os.environ.get("TRUTHFULQA_OPENAI_MIN_REQUEST_INTERVAL_SECONDS")
    if raw is None:
        return DEFAULT_OPENAI_MIN_REQUEST_INTERVAL_SECONDS
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError("TRUTHFULQA_OPENAI_MIN_REQUEST_INTERVAL_SECONDS must be a number.") from exc
    if value < 0:
        raise RuntimeError("TRUTHFULQA_OPENAI_MIN_REQUEST_INTERVAL_SECONDS must be non-negative.")
    return value


def openai_max_output_tokens() -> int:
    raw = os.environ.get("TRUTHFULQA_OPENAI_MAX_OUTPUT_TOKENS")
    if raw is None:
        return 512
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError("TRUTHFULQA_OPENAI_MAX_OUTPUT_TOKENS must be an integer.") from exc
    if value <= 0:
        raise RuntimeError("TRUTHFULQA_OPENAI_MAX_OUTPUT_TOKENS must be positive.")
    return value


def validate_model_access(conditions: list[RunCondition]) -> None:
    by_provider: dict[str, set[str]] = {}
    for condition in conditions:
        by_provider.setdefault(condition.provider, set()).add(condition.model_id)

    if "openai" in by_provider:
        require_api_key("openai")
        from openai import OpenAI

        client = OpenAI()
        for model_id in sorted(by_provider["openai"]):
            client.models.retrieve(model_id)

    if "anthropic" in by_provider:
        require_api_key("anthropic")
        from anthropic import Anthropic

        client = Anthropic()
        models = getattr(client, "models", None)
        retrieve = getattr(models, "retrieve", None)
        if retrieve is None:
            raise RuntimeError(
                "Installed Anthropic SDK does not expose models.retrieve for preflight validation. "
                "Upgrade the anthropic package or rerun with --skip-preflight if you accept first-call validation."
            )
        for model_id in sorted(by_provider["anthropic"]):
            retrieve(model_id)


def call_with_retries(call: Callable[[], object], *, before_attempt: Callable[[], None] | None = None) -> object:
    for attempt in range(1, MAX_API_ATTEMPTS + 1):
        try:
            if before_attempt is not None:
                before_attempt()
            return call()
        except Exception as exc:
            if not is_retryable_api_error(exc) or attempt == MAX_API_ATTEMPTS:
                raise
            delay = retry_delay_seconds(exc, attempt)
            print(
                f"Retryable provider error on attempt {attempt}/{MAX_API_ATTEMPTS}: "
                f"{api_error_summary(exc)}. Sleeping {delay:.1f}s before retry.",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
    raise RuntimeError("unreachable retry loop state")


def api_error_summary(exc: Exception) -> str:
    details = [type(exc).__name__]
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        details.append(f"status={status_code}")
    request_id = getattr(exc, "request_id", None)
    if request_id:
        details.append(f"request_id={request_id}")
    message = str(exc).strip()
    if message:
        details.append(f"message={single_line(message)}")
    return " ".join(details)


def single_line(value: str) -> str:
    return " ".join(value.split())


def is_retryable_api_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in TRANSIENT_STATUS_CODES:
        return True
    name = type(exc).__name__.lower()
    if "connection" in name or "timeout" in name:
        return True
    return False


def retry_delay_seconds(exc: Exception, attempt: int) -> float:
    retry_after = retry_after_header(exc)
    if retry_after is not None:
        return retry_after
    message_delay = retry_after_from_message(str(exc))
    if message_delay is not None:
        if getattr(exc, "status_code", None) == 429:
            return max(message_delay, DEFAULT_RATE_LIMIT_RETRY_SECONDS)
        return message_delay
    if getattr(exc, "status_code", None) == 429:
        return DEFAULT_RATE_LIMIT_RETRY_SECONDS
    return min(DEFAULT_RETRY_SECONDS * attempt, 120.0)


def retry_after_header(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get("retry-after") or headers.get("Retry-After")
    if value is None:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        return None


def retry_after_from_message(message: str) -> float | None:
    match = re.search(r"try again in ([0-9]+(?:\.[0-9]+)?s)", message, flags=re.IGNORECASE)
    if match is not None:
        return parse_duration_seconds(match.group(1))
    match = re.search(
        r"try again in ((?:(?:[0-9]+(?:\.[0-9]+)?)h)?(?:(?:[0-9]+(?:\.[0-9]+)?)m)?(?:(?:[0-9]+(?:\.[0-9]+)?)s)?)",
        message,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    return parse_duration_seconds(match.group(1))


def parse_duration_seconds(value: str) -> float | None:
    total = 0.0
    matched = False
    for number, unit in re.findall(r"([0-9]+(?:\.[0-9]+)?)([hms])", value, flags=re.IGNORECASE):
        matched = True
        multiplier = {"h": 3600.0, "m": 60.0, "s": 1.0}[unit.lower()]
        total += float(number) * multiplier
    if not matched:
        return None
    return max(total, 0.0)


def _to_result(
    condition: RunCondition,
    question_set_id: str,
    prompt: BinaryPrompt,
    category: str,
    raw_output: str,
    usage: Usage,
    latency_seconds: float,
) -> BenchmarkResult:
    parsed = parse_choice(raw_output)
    return BenchmarkResult(
        condition_id=condition.condition_id,
        provider=condition.provider,
        model_id=condition.model_id,
        model_label=condition.model_label,
        reasoning_effort=condition.reasoning_effort,
        question_set_id=question_set_id,
        row_id=prompt.row_id,
        category=category,
        order=prompt.order,
        correct_choice=prompt.correct_choice,
        raw_output=raw_output,
        parsed_choice=parsed,
        is_correct=parsed == prompt.correct_choice,
        is_invalid=parsed is None,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        estimated_cost_usd=estimate_cost(condition.provider, condition.model_id, usage.input_tokens, usage.output_tokens),
        latency_seconds=latency_seconds,
    )
