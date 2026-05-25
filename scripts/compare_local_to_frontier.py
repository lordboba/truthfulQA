"""Compare a local-provider results file to the frontier numbers documented in
this repository (``EXPERIMENT_STATUS.md`` and ``paper/tables/main_results.tex``).

Frontier per-row JSONL files are gitignored, so the comparison is summary-only:
accuracy, Wilson 95% CI, invalid-call rate, order-sensitive question count,
estimated cost, and mean latency. The script also reports the local row count so
the reader can see whether the local conditions were evaluated on a subset.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from truthfulqa_bench.results import BenchmarkResult, load_results


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class FrontierStat:
    label: str
    valid_correct: int
    valid_attempts: int
    invalid_rate: float
    order_sensitive_rows: int
    estimated_cost_usd: float
    mean_latency_seconds: float


FRONTIER = [
    FrontierStat("Claude Opus 4.7", 1564, 1580, 0.0, 8, 1.29, 1.54),
    FrontierStat("GPT-5.5 medium", 1544, 1580, 0.0584, 10, 5.87, 9.93),
    FrontierStat("GPT-5.5 high", 1537, 1580, 0.1627, 7, 10.74, 11.22),
    FrontierStat("GPT-5.5 low", 1537, 1580, 0.0032, 5, 3.28, 8.91),
    FrontierStat("Claude Sonnet 4.6", 1513, 1580, 0.0, 27, 0.57, 1.30),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="Local provider results JSONL.")
    parser.add_argument("--out-json", type=Path, default=ROOT / "results" / "local_vs_frontier.json")
    parser.add_argument("--out-md", type=Path, default=ROOT / "results" / "local_vs_frontier.md")
    args = parser.parse_args()

    rows = load_results(args.results)
    if not rows:
        raise SystemExit(f"No results found in {args.results}.")
    local_stats = [stat for condition_id, stat in summarize_local(rows).items()]

    payload = {
        "local_results": str(args.results),
        "local_conditions": [stat_to_dict(stat) for stat in local_stats],
        "frontier_conditions": [frontier_to_dict(stat) for stat in FRONTIER],
        "notes": (
            "Frontier numbers are copied from EXPERIMENT_STATUS.md / paper/tables/main_results.tex. "
            "Per-row frontier JSONLs are gitignored in this repo, so no paired McNemar test is run."
        ),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    args.out_md.write_text(render_markdown(local_stats), encoding="utf-8")
    print(render_markdown(local_stats))
    return 0


@dataclass(frozen=True)
class LocalStat:
    condition_id: str
    label: str
    valid_correct: int
    valid_attempts: int
    total_attempts: int
    invalid_rate: float
    accuracy: float | None
    ci_low: float | None
    ci_high: float | None
    order_sensitive_rows: int
    estimated_cost_usd: float
    mean_latency_seconds: float
    unique_rows: int


def summarize_local(rows: list[BenchmarkResult]) -> dict[str, LocalStat]:
    grouped: dict[str, list[BenchmarkResult]] = defaultdict(list)
    for row in rows:
        grouped[row.condition_id].append(row)

    summaries: dict[str, LocalStat] = {}
    for condition_id, condition_rows in sorted(grouped.items()):
        valid = [row for row in condition_rows if not row.is_invalid]
        correct = sum(1 for row in valid if row.is_correct)
        invalid_rate = (len(condition_rows) - len(valid)) / len(condition_rows) if condition_rows else 0.0
        accuracy = correct / len(valid) if valid else None
        if accuracy is not None and len(valid) > 0:
            low, high = wilson_ci(correct, len(valid))
        else:
            low = high = None
        by_row: dict[int, list[BenchmarkResult]] = defaultdict(list)
        for row in valid:
            by_row[row.row_id].append(row)
        order_sensitive = sum(
            1
            for attempts in by_row.values()
            if len(attempts) == 2 and attempts[0].is_correct != attempts[1].is_correct
        )
        mean_latency = (
            sum(row.latency_seconds for row in condition_rows) / len(condition_rows) if condition_rows else 0.0
        )
        summaries[condition_id] = LocalStat(
            condition_id=condition_id,
            label=condition_rows[0].model_label,
            valid_correct=correct,
            valid_attempts=len(valid),
            total_attempts=len(condition_rows),
            invalid_rate=invalid_rate,
            accuracy=accuracy,
            ci_low=low,
            ci_high=high,
            order_sensitive_rows=order_sensitive,
            estimated_cost_usd=sum(row.estimated_cost_usd for row in condition_rows),
            mean_latency_seconds=mean_latency,
            unique_rows=len({row.row_id for row in condition_rows}),
        )
    return summaries


def wilson_ci(correct: int, n: int) -> tuple[float, float]:
    p = correct / n
    z = 1.959963984540054
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return center - margin, center + margin


def stat_to_dict(stat: LocalStat) -> dict[str, object]:
    return {
        "condition_id": stat.condition_id,
        "label": stat.label,
        "valid_correct": stat.valid_correct,
        "valid_attempts": stat.valid_attempts,
        "total_attempts": stat.total_attempts,
        "unique_rows": stat.unique_rows,
        "invalid_rate": stat.invalid_rate,
        "accuracy": stat.accuracy,
        "accuracy_ci_95": [stat.ci_low, stat.ci_high],
        "order_sensitive_rows": stat.order_sensitive_rows,
        "estimated_cost_usd": stat.estimated_cost_usd,
        "mean_latency_seconds": stat.mean_latency_seconds,
    }


def frontier_to_dict(stat: FrontierStat) -> dict[str, object]:
    accuracy = stat.valid_correct / stat.valid_attempts
    low, high = wilson_ci(stat.valid_correct, stat.valid_attempts)
    return {
        "label": stat.label,
        "valid_correct": stat.valid_correct,
        "valid_attempts": stat.valid_attempts,
        "accuracy": accuracy,
        "accuracy_ci_95": [low, high],
        "invalid_rate": stat.invalid_rate,
        "order_sensitive_rows": stat.order_sensitive_rows,
        "estimated_cost_usd": stat.estimated_cost_usd,
        "mean_latency_seconds": stat.mean_latency_seconds,
    }


FULL_BENCHMARK_QUESTIONS = 790


def render_markdown(local_stats: list[LocalStat]) -> str:
    lines = ["# Local vs. frontier TruthfulQA binary-choice accuracy", ""]
    lines.append(
        f"Frontier numbers are from this repository's full {FULL_BENCHMARK_QUESTIONS}-question / "
        "1,580-attempt controlled run (see `EXPERIMENT_STATUS.md`)."
    )
    lines.append(
        "Local conditions list per-condition coverage in the **Questions** column "
        f"(out of {FULL_BENCHMARK_QUESTIONS}); accuracies on partial coverage are not directly "
        "comparable to the frontier rows."
    )
    lines.append("")
    lines.append("| Condition | Questions | Accuracy | 95% CI | Correct/Valid | Invalid rate | Order-sensitive Qs | Cost (USD) | Mean latency (s) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for stat in local_stats:
        accuracy_str = pct(stat.accuracy) if stat.accuracy is not None else "—"
        ci_str = f"[{pct(stat.ci_low)}, {pct(stat.ci_high)}]" if stat.ci_low is not None else "—"
        coverage = f"{stat.unique_rows}/{FULL_BENCHMARK_QUESTIONS}"
        if stat.unique_rows < FULL_BENCHMARK_QUESTIONS:
            coverage += " ⚠️"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"**local: {stat.label}**",
                    coverage,
                    accuracy_str,
                    ci_str,
                    f"{stat.valid_correct}/{stat.valid_attempts}",
                    pct(stat.invalid_rate),
                    str(stat.order_sensitive_rows),
                    f"${stat.estimated_cost_usd:.2f}",
                    f"{stat.mean_latency_seconds:.2f}",
                ]
            )
            + " |"
        )
    for stat in FRONTIER:
        accuracy = stat.valid_correct / stat.valid_attempts
        low, high = wilson_ci(stat.valid_correct, stat.valid_attempts)
        lines.append(
            "| "
            + " | ".join(
                [
                    stat.label,
                    f"{FULL_BENCHMARK_QUESTIONS}/{FULL_BENCHMARK_QUESTIONS}",
                    pct(accuracy),
                    f"[{pct(low)}, {pct(high)}]",
                    f"{stat.valid_correct}/{stat.valid_attempts}",
                    pct(stat.invalid_rate),
                    str(stat.order_sensitive_rows),
                    f"${stat.estimated_cost_usd:.2f}",
                    f"{stat.mean_latency_seconds:.2f}",
                ]
            )
            + " |"
        )
    lines.append("")
    partial = [stat for stat in local_stats if stat.unique_rows < FULL_BENCHMARK_QUESTIONS]
    if partial:
        names = ", ".join(stat.label for stat in partial)
        lines.append(
            f"> ⚠️ Partial-coverage local conditions ({names}) have wider CIs and are not directly "
            "comparable to the 790-question frontier numbers; their accuracies will move once the "
            "rest of the rows are processed."
        )
    lines.append(
        "> Frontier numbers copied from `EXPERIMENT_STATUS.md` / `paper/tables/main_results.tex`. "
        "Per-row frontier JSONLs are gitignored, so paired McNemar tests against frontier conditions "
        "are not computed."
    )
    lines.append("")
    return "\n".join(lines)


def pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{100 * value:.2f}%"


if __name__ == "__main__":
    raise SystemExit(main())
