from __future__ import annotations

from dataclasses import dataclass


TRUTHFULQA_CSV_URL = "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv"
EXPECTED_ROW_COUNT = 790

DEFAULT_DATA_PATH = "data/TruthfulQA.csv"
DEFAULT_RESULTS_DIR = "results"
DEFAULT_PILOT_RESULTS = "results/pilot.jsonl"
DEFAULT_FULL_RESULTS = "results/full.jsonl"
DEFAULT_PILOT_PROJECTION = "results/pilot_projection.json"
DEFAULT_EXPERIMENT_MANIFEST = "results/experiment_manifest.json"

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
    ("openai", "gpt-5.5"): Pricing(5.0, 30.0),
}
