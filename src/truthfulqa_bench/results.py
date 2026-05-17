from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class BenchmarkResult:
    provider: str
    model: str
    row_id: int
    category: str
    order: str
    correct_choice: str
    raw_output: str
    parsed_choice: str | None
    is_correct: bool
    is_invalid: bool
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    latency_seconds: float


def append_result(path: Path, result: BenchmarkResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(asdict(result), sort_keys=True) + "\n")


def load_results(path: Path) -> list[BenchmarkResult]:
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    rows: list[BenchmarkResult] = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                rows.append(BenchmarkResult(**json.loads(line)))
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(f"Invalid JSONL result at {path}:{line_number}") from exc
    return rows


def summarize(results: Iterable[BenchmarkResult]) -> dict[str, object]:
    grouped: dict[tuple[str, str], list[BenchmarkResult]] = defaultdict(list)
    for result in results:
        grouped[(result.provider, result.model)].append(result)

    summaries: dict[str, object] = {}
    for (provider, model), rows in sorted(grouped.items()):
        valid = [row for row in rows if not row.is_invalid]
        correct = [row for row in valid if row.is_correct]
        by_row: dict[int, list[BenchmarkResult]] = defaultdict(list)
        for row in rows:
            by_row[row.row_id].append(row)
        order_sensitive = sum(
            1
            for attempts in by_row.values()
            if len(attempts) == 2
            and all(not attempt.is_invalid for attempt in attempts)
            and attempts[0].is_correct != attempts[1].is_correct
        )
        key = f"{provider}/{model}"
        summaries[key] = {
            "requests": len(rows),
            "valid_requests": len(valid),
            "accuracy": len(correct) / len(valid) if valid else None,
            "invalid_rate": (len(rows) - len(valid)) / len(rows) if rows else None,
            "order_sensitive_rows": order_sensitive,
            "input_tokens": sum(row.input_tokens for row in rows),
            "output_tokens": sum(row.output_tokens for row in rows),
            "estimated_cost_usd": sum(row.estimated_cost_usd for row in rows),
            "mean_latency_seconds": sum(row.latency_seconds for row in rows) / len(rows) if rows else None,
        }
    return summaries
