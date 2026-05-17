from __future__ import annotations

from .config import FALLBACK_PRICING_BY_PROVIDER, PRICING_BY_PROVIDER_MODEL, Pricing


def pricing_for(provider: str, model: str) -> Pricing:
    normalized_provider = provider.lower()
    return PRICING_BY_PROVIDER_MODEL.get(
        (normalized_provider, model),
        FALLBACK_PRICING_BY_PROVIDER[normalized_provider],
    )


def estimate_cost(provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
    return pricing_for(provider, model).estimate(input_tokens, output_tokens)
