from __future__ import annotations

from pathlib import Path

from .budget import BudgetProjection, load_projection, project_budget, write_projection
from .config import DEFAULT_PILOT_PROJECTION
from .dataset import TruthfulQARow
from .prompting import BinaryPrompt, build_two_order_prompts
from .providers import make_client
from .results import BenchmarkResult, append_result, load_results

AttemptKey = tuple[str, int, str]


def run_benchmark(
    *,
    provider: str,
    models: list[str],
    rows: list[TruthfulQARow],
    output_path: Path,
    resume: bool = False,
) -> list[BenchmarkResult]:
    if output_path.exists() and not resume:
        raise FileExistsError(f"Refusing to append to existing result file: {output_path}")

    expected_attempts = build_expected_attempts(models, rows)
    results = (
        load_resume_results(
            output_path=output_path,
            provider=provider,
            expected_attempts=expected_attempts,
        )
        if resume and output_path.exists()
        else []
    )
    completed = {attempt_key(result) for result in results}

    for model in models:
        pending_prompts = [
            (row, prompt)
            for row in rows
            for prompt in build_two_order_prompts(row)
            if (model, row.row_id, prompt.order) not in completed
        ]
        if not pending_prompts:
            continue

        client = make_client(provider, model)
        for row, prompt in pending_prompts:
            result = client.complete_choice(prompt, row.category)
            append_result(output_path, result)
            results.append(result)
            completed.add(attempt_key(result))
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
    resume: bool = False,
) -> BudgetProjection:
    pilot_rows = rows[:pilot_size]
    results = run_benchmark(
        provider=provider,
        models=models,
        rows=pilot_rows,
        output_path=output_path,
        resume=resume,
    )
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


def build_expected_attempts(
    models: list[str],
    rows: list[TruthfulQARow],
) -> dict[AttemptKey, tuple[TruthfulQARow, BinaryPrompt]]:
    expected: dict[AttemptKey, tuple[TruthfulQARow, BinaryPrompt]] = {}
    for model in models:
        for row in rows:
            for prompt in build_two_order_prompts(row):
                expected[(model, row.row_id, prompt.order)] = (row, prompt)
    return expected


def load_resume_results(
    *,
    output_path: Path,
    provider: str,
    expected_attempts: dict[AttemptKey, tuple[TruthfulQARow, BinaryPrompt]],
) -> list[BenchmarkResult]:
    results = load_results(output_path)
    seen: set[AttemptKey] = set()
    for result in results:
        key = attempt_key(result)
        if key in seen:
            raise ValueError(
                "Cannot resume from results with duplicate attempt "
                f"for model={result.model}, row_id={result.row_id}, order={result.order}."
            )
        seen.add(key)

        expected = expected_attempts.get(key)
        if expected is None:
            raise ValueError(
                "Cannot resume from results that do not match this run: "
                f"model={result.model}, row_id={result.row_id}, order={result.order}."
            )
        row, prompt = expected
        validate_resume_result(result=result, provider=provider, row=row, prompt=prompt)
    return results


def validate_resume_result(
    *,
    result: BenchmarkResult,
    provider: str,
    row: TruthfulQARow,
    prompt: BinaryPrompt,
) -> None:
    if result.provider != provider:
        raise ValueError(
            f"Cannot resume {provider} run from {result.provider} result "
            f"for model={result.model}, row_id={result.row_id}, order={result.order}."
        )
    if result.category != row.category:
        raise ValueError(
            "Cannot resume from result with mismatched category "
            f"for row_id={result.row_id}: expected {row.category!r}, found {result.category!r}."
        )
    if result.correct_choice != prompt.correct_choice:
        raise ValueError(
            "Cannot resume from result with mismatched correct choice "
            f"for row_id={result.row_id}, order={result.order}: "
            f"expected {prompt.correct_choice!r}, found {result.correct_choice!r}."
        )


def attempt_key(result: BenchmarkResult) -> AttemptKey:
    return (result.model, result.row_id, result.order)
