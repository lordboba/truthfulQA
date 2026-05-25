from __future__ import annotations

from .config import PRICING_BY_PROVIDER_MODEL, Pricing


LOCAL_PRICING = Pricing(0.0, 0.0)


def pricing_for(provider: str, model: str) -> Pricing:
    normalized_provider = provider.lower()
    if normalized_provider == "local":
        return LOCAL_PRICING
    pricing = PRICING_BY_PROVIDER_MODEL.get((normalized_provider, model))
    if pricing is None:
        raise ValueError(f"No pricing configured for {normalized_provider}/{model}.")
    return pricing


def estimate_cost(provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
    return pricing_for(provider, model).estimate(input_tokens, output_tokens)
