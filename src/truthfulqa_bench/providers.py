from __future__ import annotations

import os
from dataclasses import dataclass
from time import monotonic
from typing import Protocol

from .costs import estimate_cost
from .prompting import SYSTEM_PROMPT, BinaryPrompt, parse_choice
from .results import BenchmarkResult


class ProviderClient(Protocol):
    provider: str
    model: str

    def complete_choice(self, prompt: BinaryPrompt, category: str) -> BenchmarkResult:
        ...


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int


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

    def __init__(self, model: str, max_tokens: int = 4):
        require_api_key(self.provider)
        from anthropic import Anthropic

        self.model = model
        self.max_tokens = max_tokens
        self._client = Anthropic()

    def complete_choice(self, prompt: BinaryPrompt, category: str) -> BenchmarkResult:
        start = monotonic()
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt.prompt}],
        )
        latency = monotonic() - start
        raw_output = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
        usage = Usage(input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens)
        return _to_result(self.provider, self.model, prompt, category, raw_output, usage, latency)


class OpenAIClient:
    provider = "openai"

    def __init__(self, model: str, max_output_tokens: int = 4):
        require_api_key(self.provider)
        from openai import OpenAI

        self.model = model
        self.max_output_tokens = max_output_tokens
        self._client = OpenAI()

    def complete_choice(self, prompt: BinaryPrompt, category: str) -> BenchmarkResult:
        start = monotonic()
        response = self._client.responses.create(
            model=self.model,
            temperature=0,
            max_output_tokens=self.max_output_tokens,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt.prompt},
            ],
        )
        latency = monotonic() - start
        raw_output = response.output_text
        usage = Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        return _to_result(self.provider, self.model, prompt, category, raw_output, usage, latency)


def make_client(provider: str, model: str) -> ProviderClient:
    if provider == "anthropic":
        return AnthropicClient(model)
    if provider == "openai":
        return OpenAIClient(model)
    raise ValueError(f"Unsupported provider: {provider}")


def _to_result(
    provider: str,
    model: str,
    prompt: BinaryPrompt,
    category: str,
    raw_output: str,
    usage: Usage,
    latency_seconds: float,
) -> BenchmarkResult:
    parsed = parse_choice(raw_output)
    return BenchmarkResult(
        provider=provider,
        model=model,
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
        estimated_cost_usd=estimate_cost(provider, model, usage.input_tokens, usage.output_tokens),
        latency_seconds=latency_seconds,
    )
