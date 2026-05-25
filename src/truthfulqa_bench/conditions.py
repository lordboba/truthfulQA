from __future__ import annotations

from dataclasses import asdict, dataclass


ReasoningEffort = str | None


@dataclass(frozen=True)
class RunCondition:
    condition_id: str
    provider: str
    model_id: str
    model_label: str
    reasoning_effort: ReasoningEffort = None

    def to_json(self) -> dict[str, str | None]:
        return asdict(self)


def default_conditions(provider: str) -> list[RunCondition]:
    if provider == "openai":
        return [
            RunCondition(
                condition_id=f"openai_gpt-5.5_{effort}",
                provider="openai",
                model_id="gpt-5.5",
                model_label=f"GPT-5.5 ({effort})",
                reasoning_effort=effort,
            )
            for effort in ("high", "medium", "low")
        ]
    if provider == "anthropic":
        return [
            RunCondition(
                condition_id="anthropic_claude-opus-4-7",
                provider="anthropic",
                model_id="claude-opus-4-7",
                model_label="Claude Opus 4.7",
            ),
            RunCondition(
                condition_id="anthropic_claude-sonnet-4-6",
                provider="anthropic",
                model_id="claude-sonnet-4-6",
                model_label="Claude Sonnet 4.6",
            ),
        ]
    if provider == "local":
        raise ValueError(
            "Provider 'local' has no default models. Pass --models <id> [<id> ...] with the "
            "model identifiers loaded in your local server (e.g. LM Studio)."
        )
    raise ValueError(f"Unsupported provider: {provider}")


def model_conditions(provider: str, models: list[str] | None) -> list[RunCondition]:
    if not models:
        return default_conditions(provider)
    return [
        RunCondition(
            condition_id=condition_id_for(provider, model),
            provider=provider,
            model_id=model,
            model_label=model,
        )
        for model in models
    ]


def condition_id_for(provider: str, model: str) -> str:
    safe_model = model.replace("/", "_").replace(":", "_")
    return f"{provider}_{safe_model}"
