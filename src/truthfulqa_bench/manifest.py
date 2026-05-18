from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .conditions import RunCondition
from .config import PRICING_BY_PROVIDER_MODEL
from .dataset import TruthfulQARow
from .prompting import SYSTEM_PROMPT


NO_TOOLS_POLICY = (
    "No web search, file search, retrieval, tool definitions, repo URLs, or external context are provided. "
    "This prevents active lookup during API calls but cannot prevent public benchmark memorization."
)


@dataclass(frozen=True)
class ExperimentManifest:
    question_set_id: str
    dataset_path: str
    dataset_sha256: str
    row_ids: list[int]
    prompt_sha256: str
    conditions: list[dict[str, str | None]]
    pricing_snapshot: dict[str, dict[str, float]]
    anti_cheat_policy: str


def dataset_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_hash() -> str:
    return hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()


def question_set_id(*, dataset_sha256: str, rows: list[TruthfulQARow]) -> str:
    row_ids = ",".join(str(row.row_id) for row in rows)
    return hashlib.sha256(f"{dataset_sha256}:{row_ids}".encode("utf-8")).hexdigest()[:16]


def build_manifest(
    *,
    dataset_path: Path,
    rows: list[TruthfulQARow],
    conditions: list[RunCondition],
) -> ExperimentManifest:
    data_hash = dataset_hash(dataset_path)
    return ExperimentManifest(
        question_set_id=question_set_id(dataset_sha256=data_hash, rows=rows),
        dataset_path=str(dataset_path),
        dataset_sha256=data_hash,
        row_ids=[row.row_id for row in rows],
        prompt_sha256=prompt_hash(),
        conditions=[condition.to_json() for condition in conditions],
        pricing_snapshot=pricing_snapshot(conditions),
        anti_cheat_policy=NO_TOOLS_POLICY,
    )


def pricing_snapshot(conditions: list[RunCondition]) -> dict[str, dict[str, float]]:
    snapshot: dict[str, dict[str, float]] = {}
    for condition in conditions:
        pricing = PRICING_BY_PROVIDER_MODEL.get((condition.provider, condition.model_id))
        if pricing is None:
            raise ValueError(
                f"No pricing configured for {condition.provider}/{condition.model_id}. "
                "Refusing to run with unknown cost controls."
            )
        snapshot[condition.condition_id] = asdict(pricing)
    return snapshot


def write_manifest(path: Path, manifest: ExperimentManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
