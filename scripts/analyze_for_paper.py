from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper" / "tables"


RESULT_PATHS = [
    ROOT / "results" / "anthropic_full.jsonl",
    ROOT / "results" / "openai_full.jsonl",
    ROOT / "results" / "local_full_openai_gpt-oss-20b.jsonl",
    ROOT / "results" / "local_full_google_gemma-4-e4b.jsonl",
    ROOT / "results" / "local_full_google_gemma-3-27b.jsonl",
    ROOT / "results" / "local_full_google_gemma-4-e2b.jsonl",
]


@dataclass(frozen=True)
class ConditionConfig:
    condition_id: str
    label: str
    group: str

    def __post_init__(self) -> None:
        if self.group not in {"cloud", "local"}:
            raise ValueError(f"Unsupported condition group for {self.condition_id}: {self.group}")


CONDITIONS = [
    ConditionConfig("anthropic_claude-opus-4-7", "Claude Opus 4.7", "cloud"),
    ConditionConfig("openai_gpt-5.5_medium", "GPT-5.5 medium", "cloud"),
    ConditionConfig("openai_gpt-5.5_high", "GPT-5.5 high", "cloud"),
    ConditionConfig("openai_gpt-5.5_low", "GPT-5.5 low", "cloud"),
    ConditionConfig("anthropic_claude-sonnet-4-6", "Claude Sonnet 4.6", "cloud"),
    ConditionConfig("local_openai_gpt-oss-20b", "openai/gpt-oss-20b (local)", "local"),
    ConditionConfig("local_google_gemma-4-e4b", "google/gemma-4-e4b (local)", "local"),
    ConditionConfig("local_google_gemma-3-27b", "google/gemma-3-27b (local)", "local"),
    ConditionConfig("local_google_gemma-4-e2b", "google/gemma-4-e2b (local)", "local"),
]


CLOUD_COMPARISONS = [
    ("anthropic_claude-opus-4-7", "openai_gpt-5.5_medium"),
    ("openai_gpt-5.5_medium", "openai_gpt-5.5_high"),
    ("openai_gpt-5.5_medium", "openai_gpt-5.5_low"),
    ("openai_gpt-5.5_medium", "anthropic_claude-sonnet-4-6"),
    ("anthropic_claude-opus-4-7", "anthropic_claude-sonnet-4-6"),
]


@dataclass(frozen=True)
class Row:
    condition_id: str
    category: str
    row_id: int
    order: str
    is_correct: bool
    is_invalid: bool
    estimated_cost_usd: float
    latency_seconds: float
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class CategorySummary:
    category: str
    correct: int
    attempts: int
    mean_accuracy: float


def load_rows(result_paths: Sequence[Path] = RESULT_PATHS) -> list[Row]:
    rows: list[Row] = []
    for path in result_paths:
        source_rows = load_source_rows(path)
        if not source_rows:
            raise ValueError(f"Result source has no rows: {path}")
        rows.extend(source_rows)
    return rows


def load_source_rows(path: Path) -> list[Row]:
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    rows: list[Row] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                rows.append(
                    Row(
                        condition_id=raw["condition_id"],
                        category=raw["category"],
                        row_id=int(raw["row_id"]),
                        order=raw["order"],
                        is_correct=bool(raw["is_correct"]),
                        is_invalid=bool(raw["is_invalid"]),
                        estimated_cost_usd=float(raw["estimated_cost_usd"]),
                        latency_seconds=float(raw["latency_seconds"]),
                        input_tokens=int(raw["input_tokens"]),
                        output_tokens=int(raw["output_tokens"]),
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid JSONL result at {path}:{line_number}") from exc
    return rows


def generate_analysis(
    *,
    result_paths: Sequence[Path] = RESULT_PATHS,
    out_dir: Path = OUT_DIR,
    conditions: Sequence[ConditionConfig] = CONDITIONS,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(result_paths)
    validate_rows(rows, conditions)

    condition_lookup = condition_by_id(conditions)
    valid = [row for row in rows if not row.is_invalid]

    write_main_results(out_dir, rows, conditions)
    write_paired_tests(out_dir, valid, conditions, condition_lookup)
    cloud_hardest = write_hardest_categories(out_dir, valid, conditions)
    write_local_hardest_categories(out_dir, valid, conditions, cloud_hardest)

    metadata = build_metadata(rows, valid, conditions, result_paths)
    (out_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def validate_rows(rows: Sequence[Row], conditions: Sequence[ConditionConfig]) -> None:
    if not rows:
        raise ValueError("No result rows loaded.")

    configured = condition_by_id(conditions)
    observed_conditions = {row.condition_id for row in rows}
    configured_conditions = set(configured)

    missing = sorted(configured_conditions - observed_conditions)
    if missing:
        raise ValueError(f"Missing configured condition(s): {', '.join(missing)}")

    unexpected = sorted(observed_conditions - configured_conditions)
    if unexpected:
        raise ValueError(f"Unexpected condition(s) in result files: {', '.join(unexpected)}")

    valid_keys: dict[tuple[str, int, str], Row] = {}
    valid_keys_by_condition: dict[str, set[tuple[int, str]]] = defaultdict(set)
    valid_rows_by_condition: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        if row.is_invalid:
            continue
        key = (row.condition_id, row.row_id, row.order)
        if key in valid_keys:
            raise ValueError(
                "Duplicate valid attempt key: "
                f"condition={row.condition_id}, row_id={row.row_id}, order={row.order}"
            )
        valid_keys[key] = row
        valid_keys_by_condition[row.condition_id].add((row.row_id, row.order))
        valid_rows_by_condition[row.condition_id].append(row)

    no_valid_rows = [condition.condition_id for condition in conditions if not valid_rows_by_condition[condition.condition_id]]
    if no_valid_rows:
        raise ValueError(f"Configured condition(s) have no valid rows: {', '.join(no_valid_rows)}")

    for left, right in comparison_pairs(conditions):
        matched = valid_keys_by_condition[left] & valid_keys_by_condition[right]
        if not matched:
            raise ValueError(f"No matched valid attempts for paired comparison: {left} vs. {right}")


def condition_by_id(conditions: Sequence[ConditionConfig]) -> dict[str, ConditionConfig]:
    lookup: dict[str, ConditionConfig] = {}
    for condition in conditions:
        if condition.condition_id in lookup:
            raise ValueError(f"Duplicate condition config: {condition.condition_id}")
        lookup[condition.condition_id] = condition
    return lookup


def write_main_results(out_dir: Path, rows: Sequence[Row], conditions: Sequence[ConditionConfig]) -> None:
    summary_rows: list[str] = []
    previous_group: str | None = None
    for condition in conditions:
        if previous_group is not None and condition.group != previous_group:
            summary_rows.append(r"\midrule")
        previous_group = condition.group

        all_rows = [row for row in rows if row.condition_id == condition.condition_id]
        condition_valid = [row for row in all_rows if not row.is_invalid]
        correct = sum(row.is_correct for row in condition_valid)
        n = len(condition_valid)
        low, high = ci95(correct, n)
        order_sensitive = order_sensitive_questions(condition_valid)
        invalid_rate = (len(all_rows) - n) / len(all_rows)
        mean_latency = sum(row.latency_seconds for row in all_rows) / len(all_rows)
        summary_rows.append(
            " & ".join(
                [
                    condition.label,
                    f"{correct}/{n}",
                    pct(correct / n),
                    f"[{pct(low)}, {pct(high)}]",
                    pct(invalid_rate),
                    str(order_sensitive),
                    money(sum(row.estimated_cost_usd for row in all_rows)),
                    f"{mean_latency:.2f}",
                ]
            )
            + r" \\"
        )

    write_table(
        out_dir / "main_results.tex",
        r"Model & Correct & Accuracy & 95\% CI & Invalid calls & Order-sensitive Qs & Cost & Latency (s) \\",
        summary_rows,
    )


def write_paired_tests(
    out_dir: Path,
    valid: Sequence[Row],
    conditions: Sequence[ConditionConfig],
    condition_lookup: dict[str, ConditionConfig],
) -> None:
    aligned: dict[tuple[int, str], dict[str, bool]] = defaultdict(dict)
    for row in valid:
        aligned[(row.row_id, row.order)][row.condition_id] = row.is_correct

    comparison_rows: list[str] = []
    previous_group: str | None = None
    for group, left, right in grouped_comparison_pairs(conditions):
        if previous_group is not None and group != previous_group:
            comparison_rows.append(r"\midrule")
        previous_group = group

        left_only = right_only = n = 0
        for outcomes in aligned.values():
            if left not in outcomes or right not in outcomes:
                continue
            n += 1
            if outcomes[left] and not outcomes[right]:
                left_only += 1
            elif outcomes[right] and not outcomes[left]:
                right_only += 1
        comparison_rows.append(
            " & ".join(
                [
                    comparison_label(condition_lookup[left], condition_lookup[right]),
                    str(n),
                    str(left_only),
                    str(right_only),
                    format_p_value(exact_mcnemar_p(left_only, right_only)),
                ]
            )
            + r" \\"
        )

    write_table(
        out_dir / "paired_tests.tex",
        r"Comparison & Matched attempts & Left only correct & Right only correct & Exact McNemar $p$ \\",
        comparison_rows,
    )


def write_hardest_categories(
    out_dir: Path,
    valid: Sequence[Row],
    conditions: Sequence[ConditionConfig],
    *,
    limit: int = 10,
) -> list[CategorySummary]:
    cloud_conditions = [condition for condition in conditions if condition.group == "cloud"]
    hardest = hardest_categories(valid, cloud_conditions, limit=limit)
    category_rows = [
        " & ".join([tex_escape(summary.category), f"{summary.correct}/{summary.attempts}", pct(summary.mean_accuracy)])
        + r" \\"
        for summary in hardest
    ]
    write_table(
        out_dir / "hardest_categories.tex",
        r"Category & Correct & Mean accuracy across conditions \\",
        category_rows,
    )
    return hardest


def write_local_hardest_categories(
    out_dir: Path,
    valid: Sequence[Row],
    conditions: Sequence[ConditionConfig],
    cloud_hardest: Sequence[CategorySummary],
) -> None:
    cloud_conditions = [condition for condition in conditions if condition.group == "cloud"]
    local_conditions = [condition for condition in conditions if condition.group == "local"]
    rows: list[str] = []
    for summary in cloud_hardest:
        cloud_rows = rows_for_conditions_and_category(valid, cloud_conditions, summary.category)
        local_rows = rows_for_conditions_and_category(valid, local_conditions, summary.category)
        if not local_rows:
            raise ValueError(f"No local valid rows for cloud-hardest category: {summary.category}")
        cloud_correct = sum(row.is_correct for row in cloud_rows)
        local_correct = sum(row.is_correct for row in local_rows)
        local_mean_accuracy = mean_category_accuracy(valid, local_conditions, summary.category)
        rows.append(
            " & ".join(
                [
                    tex_escape(summary.category),
                    f"{cloud_correct}/{len(cloud_rows)}",
                    pct(summary.mean_accuracy),
                    f"{local_correct}/{len(local_rows)}",
                    pct(local_mean_accuracy),
                ]
            )
            + r" \\"
        )

    write_table(
        out_dir / "hardest_categories_local.tex",
        r"Category & Cloud correct & Cloud mean acc. & Local correct & Local mean acc. \\",
        rows,
    )


def hardest_categories(
    valid: Sequence[Row],
    conditions: Sequence[ConditionConfig],
    *,
    limit: int,
) -> list[CategorySummary]:
    condition_ids = {condition.condition_id for condition in conditions}
    by_condition_category: dict[tuple[str, str], list[Row]] = defaultdict(list)
    for row in valid:
        if row.condition_id in condition_ids:
            by_condition_category[(row.condition_id, row.category)].append(row)

    categories = sorted({category for _, category in by_condition_category})
    summaries: list[CategorySummary] = []
    for category in categories:
        per_condition_scores: list[float] = []
        total_correct = 0
        total_attempts = 0
        for condition in conditions:
            category_rows = by_condition_category[(condition.condition_id, category)]
            if not category_rows:
                continue
            correct = sum(row.is_correct for row in category_rows)
            per_condition_scores.append(correct / len(category_rows))
            total_correct += correct
            total_attempts += len(category_rows)
        if per_condition_scores:
            summaries.append(
                CategorySummary(
                    category=category,
                    correct=total_correct,
                    attempts=total_attempts,
                    mean_accuracy=sum(per_condition_scores) / len(per_condition_scores),
                )
            )

    return sorted(summaries, key=lambda item: (item.mean_accuracy, item.category))[:limit]


def rows_for_conditions_and_category(
    rows: Sequence[Row],
    conditions: Sequence[ConditionConfig],
    category: str,
) -> list[Row]:
    condition_ids = {condition.condition_id for condition in conditions}
    return [row for row in rows if row.condition_id in condition_ids and row.category == category]


def mean_category_accuracy(rows: Sequence[Row], conditions: Sequence[ConditionConfig], category: str) -> float:
    scores: list[float] = []
    for condition in conditions:
        condition_rows = [
            row for row in rows if row.condition_id == condition.condition_id and row.category == category
        ]
        if condition_rows:
            scores.append(sum(row.is_correct for row in condition_rows) / len(condition_rows))
    if not scores:
        raise ValueError(f"No valid rows for category: {category}")
    return sum(scores) / len(scores)


def build_metadata(
    rows: Sequence[Row],
    valid: Sequence[Row],
    conditions: Sequence[ConditionConfig],
    result_paths: Sequence[Path],
) -> dict[str, object]:
    unique_attempts_by_condition = {
        condition.condition_id: len(
            {(row.row_id, row.order) for row in rows if row.condition_id == condition.condition_id}
        )
        for condition in conditions
    }
    cloud_conditions = [condition for condition in conditions if condition.group == "cloud"]
    local_conditions = [condition for condition in conditions if condition.group == "local"]

    return {
        "total_result_rows": len(rows),
        "expected_attempts": sum(unique_attempts_by_condition.values()),
        "valid_attempts": len(valid),
        "questions": len({row.row_id for row in rows}),
        "orders_per_question": len({row.order for row in rows}),
        "conditions": len(conditions),
        "cloud_conditions": len(cloud_conditions),
        "local_conditions": len(local_conditions),
        "condition_attempts": unique_attempts_by_condition,
        "configured_conditions": [asdict(condition) for condition in conditions],
        "result_sources": [display_path(path) for path in result_paths],
    }


def comparison_pairs(conditions: Sequence[ConditionConfig]) -> list[tuple[str, str]]:
    return [(left, right) for _, left, right in grouped_comparison_pairs(conditions)]


def grouped_comparison_pairs(conditions: Sequence[ConditionConfig]) -> list[tuple[str, str, str]]:
    configured_ids = {condition.condition_id for condition in conditions}
    local_condition_ids = [condition.condition_id for condition in conditions if condition.group == "local"]
    cloud_pairs = [
        ("cloud", left, right)
        for left, right in CLOUD_COMPARISONS
        if left in configured_ids and right in configured_ids
    ]
    local_pairs = [("local", left, right) for left, right in combinations(local_condition_ids, 2)]
    return cloud_pairs + local_pairs


def comparison_label(left: ConditionConfig, right: ConditionConfig) -> str:
    if left.group == "local" and right.group == "local":
        return f"{short_local_label(left)} vs. {short_local_label(right)} (local)"
    return f"{left.label} vs. {right.label}"


def short_local_label(condition: ConditionConfig) -> str:
    label = condition.label.removesuffix(" (local)")
    return label.rsplit("/", maxsplit=1)[-1]


def order_sensitive_questions(rows: Sequence[Row]) -> int:
    order_by_question: dict[int, list[Row]] = defaultdict(list)
    for row in rows:
        order_by_question[row.row_id].append(row)
    return sum(
        1
        for attempts in order_by_question.values()
        if len(attempts) == 2 and attempts[0].is_correct != attempts[1].is_correct
    )


def pct(value: float) -> str:
    return f"{100 * value:.2f}\\%"


def money(value: float) -> str:
    return f"\\${value:.2f}"


def ci95(k: int, n: int) -> tuple[float, float]:
    p = k / n
    z = 1.959963984540054
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return center - margin, center + margin


def tex_escape(text: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def write_table(path: Path, header: str, rows: Sequence[str]) -> None:
    body = "\n".join(rows)
    path.write_text(header + "\n\\midrule\n" + body + "\n\\bottomrule\n", encoding="utf-8")


def exact_mcnemar_p(a_only: int, b_only: int) -> float:
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, i) for i in range(min(a_only, b_only) + 1))
    return min(1.0, 2 * tail / (2**discordant))


def format_p_value(value: float) -> str:
    formatted = f"{value:.3g}"
    if "e" in formatted or "." not in formatted:
        return formatted

    fractional = formatted.split(".", maxsplit=1)[1]
    significant_digits = len(fractional.lstrip("0"))
    if significant_digits >= 3:
        return formatted
    return formatted + ("0" * (3 - significant_digits))


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    generate_analysis()


if __name__ == "__main__":
    main()
