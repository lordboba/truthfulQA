from __future__ import annotations

from pathlib import Path

from .budget import BudgetProjection, load_projection, project_budget, write_projection
from .config import DEFAULT_PILOT_PROJECTION
from .dataset import TruthfulQARow
from .prompting import build_two_order_prompts
from .providers import make_client
from .results import BenchmarkResult, append_result, load_results


def run_benchmark(
    *,
    provider: str,
    models: list[str],
    rows: list[TruthfulQARow],
    output_path: Path,
) -> list[BenchmarkResult]:
    if output_path.exists():
        raise FileExistsError(f"Refusing to append to existing result file: {output_path}")

    results: list[BenchmarkResult] = []
    for model in models:
        client = make_client(provider, model)
        for row in rows:
            for prompt in build_two_order_prompts(row):
                result = client.complete_choice(prompt, row.category)
                append_result(output_path, result)
                results.append(result)
    return results


def run_pilot(
    *,
    provider: str,
    models: list[str],
    rows: list[TruthfulQARow],
    pilot_size: int,
    output_path: Path,
    projection_path: Path,
    budget_usd: float,
) -> BudgetProjection:
    pilot_rows = rows[:pilot_size]
    results = run_benchmark(provider=provider, models=models, rows=pilot_rows, output_path=output_path)
    projection = project_budget(
        provider=provider,
        pilot_results=results,
        pilot_rows=len(pilot_rows),
        total_rows=len(rows),
        budget_usd=budget_usd,
    )
    write_projection(projection_path, projection)
    return projection


def assert_full_run_allowed(
    *,
    projection_path: Path = Path(DEFAULT_PILOT_PROJECTION),
) -> BudgetProjection:
    projection = load_projection(projection_path)
    if not projection.passes:
        raise RuntimeError(
            "Full run blocked: pilot projected "
            f"${projection.projected_cost_usd:.4f}, exceeding "
            f"{projection.budget_gate_ratio:.0%} of ${projection.budget_usd:.2f}."
        )
    return projection


def load_existing_results(path: Path) -> list[BenchmarkResult]:
    return load_results(path)
