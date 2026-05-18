from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATHS = [ROOT / "results" / "anthropic_full.jsonl", ROOT / "results" / "openai_full.jsonl"]
OUT_DIR = ROOT / "paper" / "tables"


LABELS = {
    "anthropic_claude-opus-4-7": "Claude Opus 4.7",
    "anthropic_claude-sonnet-4-6": "Claude Sonnet 4.6",
    "openai_gpt-5.5_high": "GPT-5.5 high",
    "openai_gpt-5.5_medium": "GPT-5.5 medium",
    "openai_gpt-5.5_low": "GPT-5.5 low",
}


ORDER = [
    "anthropic_claude-opus-4-7",
    "openai_gpt-5.5_medium",
    "openai_gpt-5.5_high",
    "openai_gpt-5.5_low",
    "anthropic_claude-sonnet-4-6",
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


def load_rows() -> list[Row]:
    rows: list[Row] = []
    for path in RESULT_PATHS:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
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
    return rows


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


def write_table(path: Path, header: str, rows: list[str]) -> None:
    body = "\n".join(rows)
    path.write_text(header + "\n\\midrule\n" + body + "\n\\bottomrule\n", encoding="utf-8")


def exact_mcnemar_p(a_only: int, b_only: int) -> float:
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, i) for i in range(min(a_only, b_only) + 1))
    return min(1.0, 2 * tail / (2**discordant))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    valid = [row for row in rows if not row.is_invalid]

    summary_rows: list[str] = []
    for condition in ORDER:
        all_rows = [row for row in rows if row.condition_id == condition]
        condition_valid = [row for row in all_rows if not row.is_invalid]
        correct = sum(row.is_correct for row in condition_valid)
        n = len(condition_valid)
        low, high = ci95(correct, n)
        order_by_question: dict[int, list[Row]] = defaultdict(list)
        for row in condition_valid:
            order_by_question[row.row_id].append(row)
        order_sensitive = sum(
            1
            for attempts in order_by_question.values()
            if len(attempts) == 2 and attempts[0].is_correct != attempts[1].is_correct
        )
        invalid_rate = (len(all_rows) - n) / len(all_rows)
        mean_latency = sum(row.latency_seconds for row in all_rows) / len(all_rows)
        summary_rows.append(
            " & ".join(
                [
                    LABELS[condition],
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
        OUT_DIR / "main_results.tex",
        r"Model & Correct & Accuracy & 95\% CI & Invalid calls & Order-sensitive Qs & Cost & Latency (s) \\",
        summary_rows,
    )

    aligned: dict[tuple[int, str], dict[str, bool]] = defaultdict(dict)
    for row in valid:
        aligned[(row.row_id, row.order)][row.condition_id] = row.is_correct
    comparisons = [
        ("anthropic_claude-opus-4-7", "openai_gpt-5.5_medium"),
        ("openai_gpt-5.5_medium", "openai_gpt-5.5_high"),
        ("openai_gpt-5.5_medium", "openai_gpt-5.5_low"),
        ("openai_gpt-5.5_medium", "anthropic_claude-sonnet-4-6"),
        ("anthropic_claude-opus-4-7", "anthropic_claude-sonnet-4-6"),
    ]
    comparison_rows: list[str] = []
    for left, right in comparisons:
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
                    f"{LABELS[left]} vs. {LABELS[right]}",
                    str(n),
                    str(left_only),
                    str(right_only),
                    f"{exact_mcnemar_p(left_only, right_only):.3g}",
                ]
            )
            + r" \\"
        )
    write_table(
        OUT_DIR / "paired_tests.tex",
        r"Comparison & Matched attempts & Left only correct & Right only correct & Exact McNemar $p$ \\",
        comparison_rows,
    )

    category_summary: dict[str, list[tuple[float, int, int]]] = defaultdict(list)
    for condition in ORDER:
        by_category: dict[str, list[Row]] = defaultdict(list)
        for row in valid:
            if row.condition_id == condition:
                by_category[row.category].append(row)
        for category, category_rows in by_category.items():
            correct = sum(row.is_correct for row in category_rows)
            n = len(category_rows)
            category_summary[category].append((correct / n, correct, n))

    hardest = sorted(
        (
            (sum(score for score, _, _ in stats) / len(stats), category, stats)
            for category, stats in category_summary.items()
        ),
        key=lambda item: item[0],
    )[:10]
    category_rows = []
    for mean_accuracy, category, stats in hardest:
        total_correct = sum(correct for _, correct, _ in stats)
        total_n = sum(n for _, _, n in stats)
        category_rows.append(
            " & ".join([tex_escape(category), f"{total_correct}/{total_n}", pct(mean_accuracy)])
            + r" \\"
        )
    write_table(
        OUT_DIR / "hardest_categories.tex",
        r"Category & Correct & Mean accuracy across conditions \\",
        category_rows,
    )

    metadata = {
        "total_api_rows": len(rows),
        "valid_attempts": len(valid),
        "questions": len({row.row_id for row in valid}),
        "orders_per_question": 2,
        "conditions": len(ORDER),
    }
    (OUT_DIR / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
