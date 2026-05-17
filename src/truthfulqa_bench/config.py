from __future__ import annotations

from dataclasses import dataclass


TRUTHFULQA_CSV_URL = "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv"
EXPECTED_ROW_COUNT = 790

DEFAULT_DATA_PATH = "data/TruthfulQA.csv"
DEFAULT_RESULTS_DIR = "results"
DEFAULT_PILOT_RESULTS = "results/pilot.jsonl"
DEFAULT_FULL_RESULTS = "results/full.jsonl"
DEFAULT_PILOT_PROJECTION = "results/pilot_projection.json"

DEFAULT_ANTHROPIC_MODELS = ("claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5")
DEFAULT_OPENAI_MODELS = ("gpt-5.5", "gpt-5.4", "gpt-5.4-mini")


@dataclass(frozen=True)
class Pricing:
    input_per_mtok: float
    output_per_mtok: float
    tokenizer_multiplier: float = 1.0

    def estimate(self, input_tokens: int, output_tokens: int) -> float:
        adjusted_input = input_tokens * self.tokenizer_multiplier
        return (adjusted_input / 1_000_000 * self.input_per_mtok) + (
            output_tokens / 1_000_000 * self.output_per_mtok
        )


PRICING_BY_PROVIDER_MODEL: dict[tuple[str, str], Pricing] = {
    ("anthropic", "claude-opus-4-7"): Pricing(5.0, 25.0),
    ("anthropic", "claude-sonnet-4-6"): Pricing(3.0, 15.0),
    ("anthropic", "claude-haiku-4-5"): Pricing(1.0, 5.0),
    ("openai", "gpt-5.5"): Pricing(5.0, 30.0),
    ("openai", "gpt-5.4"): Pricing(2.5, 15.0),
    ("openai", "gpt-5.4-mini"): Pricing(0.4, 1.6),
}


FALLBACK_PRICING_BY_PROVIDER: dict[str, Pricing] = {
    "anthropic": Pricing(5.0, 25.0),
    "openai": Pricing(5.0, 30.0),
}
