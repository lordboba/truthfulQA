from __future__ import annotations

import threading
from collections.abc import Iterable
from contextlib import contextmanager
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Iterator

from collections import Counter

from .budget import BudgetProjection, load_projection, project_budget, write_projection
from .conditions import RunCondition
from .config import DEFAULT_LOCAL_MAX_INVALID_RETRIES_PER_ATTEMPT, DEFAULT_PILOT_PROJECTION
from .dataset import TruthfulQARow
from .prompting import BinaryPrompt, build_two_order_prompts
from .providers import make_client
from .results import BenchmarkResult, append_result, load_results

AttemptKey = tuple[str, int, str]
PendingAttempt = tuple[TruthfulQARow, BinaryPrompt, RunCondition]
DEFAULT_REQUEST_COST_RESERVE_USD = 1.0


def run_benchmark(
    *,
    provider: str,
    conditions: list[RunCondition],
    rows: list[TruthfulQARow],
    question_set_id: str,
    output_path: Path,
    budget_usd: float | None = None,
    request_cost_reserve_usd: float = DEFAULT_REQUEST_COST_RESERVE_USD,
    resume: bool = False,
    max_workers: int = 1,
) -> list[BenchmarkResult]:
    with acquire_run_lock(output_path):
        return run_benchmark_locked(
            provider=provider,
            conditions=conditions,
            rows=rows,
            question_set_id=question_set_id,
            output_path=output_path,
            budget_usd=budget_usd,
            request_cost_reserve_usd=request_cost_reserve_usd,
            resume=resume,
            max_workers=max_workers,
        )


def run_benchmark_locked(
    *,
    provider: str,
    conditions: list[RunCondition],
    rows: list[TruthfulQARow],
    question_set_id: str,
    output_path: Path,
    budget_usd: float | None = None,
    request_cost_reserve_usd: float = DEFAULT_REQUEST_COST_RESERVE_USD,
    resume: bool = False,
    max_workers: int = 1,
) -> list[BenchmarkResult]:
    if max_workers <= 0:
        raise ValueError("max_workers must be positive.")
    if output_path.exists() and not resume:
        raise FileExistsError(f"Refusing to append to existing result file: {output_path}")

    validate_conditions(provider=provider, conditions=conditions)
    expected_attempts = build_expected_attempts(conditions, rows)
    results = (
        load_resume_results(
            output_path=output_path,
            provider=provider,
            question_set_id=question_set_id,
            expected_attempts=expected_attempts,
        )
        if resume and output_path.exists()
        else []
    )
    completed = {attempt_key(result) for result in results if not result.is_invalid}
    invalid_counts: Counter[AttemptKey] = Counter(
        attempt_key(result) for result in results if result.is_invalid
    )
    max_invalid_retries = max_invalid_retries_for(provider)
    spent = sum(result.estimated_cost_usd for result in results if result.provider == provider)

    while True:
        pending = list(
            iter_pending_attempts(
                rows=rows,
                conditions=conditions,
                completed=completed,
                invalid_counts=invalid_counts,
                max_invalid_retries=max_invalid_retries,
            )
        )
        if not pending:
            return results
        if max_workers == 1:
            spent = run_benchmark_sequential(
                provider=provider,
                question_set_id=question_set_id,
                pending=pending,
                output_path=output_path,
                budget_usd=budget_usd,
                request_cost_reserve_usd=request_cost_reserve_usd,
                results=results,
                completed=completed,
                invalid_counts=invalid_counts,
                initial_spent=spent,
            )
        else:
            spent = run_benchmark_parallel(
                provider=provider,
                question_set_id=question_set_id,
                pending=pending,
                output_path=output_path,
                budget_usd=budget_usd,
                request_cost_reserve_usd=request_cost_reserve_usd,
                results=results,
                completed=completed,
                invalid_counts=invalid_counts,
                initial_spent=spent,
                max_workers=max_workers,
            )


def run_benchmark_sequential(
    *,
    provider: str,
    question_set_id: str,
    pending: Iterable[PendingAttempt],
    output_path: Path,
    budget_usd: float | None,
    request_cost_reserve_usd: float,
    results: list[BenchmarkResult],
    completed: set[AttemptKey],
    invalid_counts: Counter[AttemptKey],
    initial_spent: float,
) -> float:
    spent = initial_spent
    clients = {}
    try:
        for row, prompt, condition in pending:
            assert_budget_allows_request(
                provider=provider,
                spent=spent,
                budget_usd=budget_usd,
                request_cost_reserve_usd=request_cost_reserve_usd,
                in_flight_requests=0,
            )
            client = clients.get(condition.condition_id)
            if client is None:
                client = make_client(condition, question_set_id)
                clients[condition.condition_id] = client
            result = client.complete_choice(prompt, row.category)
            append_completed_result(
                provider=provider,
                output_path=output_path,
                results=results,
                completed=completed,
                invalid_counts=invalid_counts,
                result=result,
                budget_usd=budget_usd,
            )
            spent += result.estimated_cost_usd
    finally:
        clients.clear()
    return spent


@contextmanager
def acquire_run_lock(output_path: Path) -> Iterator[None]:
    import fcntl

    lock_path = output_path.with_name(f"{output_path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another benchmark process is already writing {output_path}.") from exc
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def run_benchmark_parallel(
    *,
    provider: str,
    question_set_id: str,
    pending: Iterable[PendingAttempt],
    output_path: Path,
    budget_usd: float | None,
    request_cost_reserve_usd: float,
    results: list[BenchmarkResult],
    completed: set[AttemptKey],
    invalid_counts: Counter[AttemptKey],
    initial_spent: float,
    max_workers: int,
) -> float:
    spent = initial_spent
    pending_iter = iter(pending)
    in_flight: set[Future[BenchmarkResult]] = set()
    thread_state = threading.local()

    def complete(row: TruthfulQARow, prompt: BinaryPrompt, condition: RunCondition) -> BenchmarkResult:
        clients = getattr(thread_state, "clients", None)
        if clients is None:
            clients = {}
            thread_state.clients = clients
        client = clients.get(condition.condition_id)
        if client is None:
            client = make_client(condition, question_set_id)
            clients[condition.condition_id] = client
        return client.complete_choice(prompt, row.category)

    def submit_next(executor: ThreadPoolExecutor) -> bool:
        nonlocal spent
        try:
            row, prompt, condition = next(pending_iter)
        except StopIteration:
            return False
        assert_budget_allows_request(
            provider=provider,
            spent=spent,
            budget_usd=budget_usd,
            request_cost_reserve_usd=request_cost_reserve_usd,
            in_flight_requests=len(in_flight),
        )
        in_flight.add(executor.submit(complete, row, prompt, condition))
        return True

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while len(in_flight) < max_workers and submit_next(executor):
            pass
        while in_flight:
            done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                result = future.result()
                append_completed_result(
                    provider=provider,
                    output_path=output_path,
                    results=results,
                    completed=completed,
                    invalid_counts=invalid_counts,
                    result=result,
                    budget_usd=budget_usd,
                )
                spent += result.estimated_cost_usd
            while len(in_flight) < max_workers and submit_next(executor):
                pass
    return spent


def assert_budget_allows_request(
    *,
    provider: str,
    spent: float,
    budget_usd: float | None,
    request_cost_reserve_usd: float,
    in_flight_requests: int,
) -> None:
    if provider == "local" or budget_usd is None:
        return
    reserved = request_cost_reserve_usd * (in_flight_requests + 1)
    if spent + reserved > budget_usd:
        raise RuntimeError(
            f"Stopping before scheduling another {provider} request because estimated spend "
            f"${spent:.4f} plus ${reserved:.2f} reserve would exceed "
            f"the ${budget_usd:.2f} budget."
        )


def append_completed_result(
    *,
    provider: str,
    output_path: Path,
    results: list[BenchmarkResult],
    completed: set[AttemptKey],
    invalid_counts: Counter[AttemptKey],
    result: BenchmarkResult,
    budget_usd: float | None,
) -> None:
    append_result(output_path, result)
    results.append(result)
    key = attempt_key(result)
    if result.is_invalid:
        invalid_counts[key] += 1
    else:
        completed.add(key)
    if provider == "local" or budget_usd is None:
        return
    spent = sum(row.estimated_cost_usd for row in results if row.provider == provider)
    if spent > budget_usd:
        raise RuntimeError(
            f"{provider} budget exceeded after a completed API call: estimated spend "
            f"${spent:.4f} is above ${budget_usd:.2f}. Increase request reserve before rerunning."
        )


def iter_pending_attempts(
    *,
    rows: list[TruthfulQARow],
    conditions: list[RunCondition],
    completed: set[AttemptKey],
    invalid_counts: Counter[AttemptKey] | None = None,
    max_invalid_retries: int | None = None,
):
    for row in rows:
        for prompt in build_two_order_prompts(row):
            for condition in conditions:
                key = (condition.condition_id, row.row_id, prompt.order)
                if key in completed:
                    continue
                if (
                    max_invalid_retries is not None
                    and invalid_counts is not None
                    and invalid_counts.get(key, 0) >= max_invalid_retries
                ):
                    continue
                yield row, prompt, condition


def max_invalid_retries_for(provider: str) -> int | None:
    if provider == "local":
        return DEFAULT_LOCAL_MAX_INVALID_RETRIES_PER_ATTEMPT
    return None


def run_pilot(
    *,
    provider: str,
    conditions: list[RunCondition],
    rows: list[TruthfulQARow],
    question_set_id: str,
    pilot_size: int,
    output_path: Path,
    projection_path: Path,
    budget_usd: float,
    resume: bool = False,
    max_workers: int = 1,
) -> BudgetProjection:
    pilot_rows = rows[:pilot_size]
    results = run_benchmark(
        provider=provider,
        conditions=conditions,
        rows=pilot_rows,
        question_set_id=question_set_id,
        output_path=output_path,
        budget_usd=budget_usd,
        resume=resume,
        max_workers=max_workers,
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
    conditions: list[RunCondition],
    rows: list[TruthfulQARow],
) -> dict[AttemptKey, tuple[TruthfulQARow, BinaryPrompt]]:
    expected: dict[AttemptKey, tuple[TruthfulQARow, BinaryPrompt]] = {}
    for condition in conditions:
        for row in rows:
            for prompt in build_two_order_prompts(row):
                expected[(condition.condition_id, row.row_id, prompt.order)] = (row, prompt)
    return expected


def load_resume_results(
    *,
    output_path: Path,
    provider: str,
    question_set_id: str,
    expected_attempts: dict[AttemptKey, tuple[TruthfulQARow, BinaryPrompt]],
) -> list[BenchmarkResult]:
    results = load_results(output_path)
    valid_seen: set[AttemptKey] = set()
    for result in results:
        key = attempt_key(result)
        if not result.is_invalid and key in valid_seen:
            raise ValueError(
                "Cannot resume from results with duplicate valid attempt "
                f"for condition={result.condition_id}, row_id={result.row_id}, order={result.order}."
            )
        if not result.is_invalid:
            valid_seen.add(key)

        expected = expected_attempts.get(key)
        if expected is None:
            raise ValueError(
                "Cannot resume from results that do not match this run: "
                f"condition={result.condition_id}, row_id={result.row_id}, order={result.order}."
            )
        row, prompt = expected
        validate_resume_result(
            result=result,
            provider=provider,
            question_set_id=question_set_id,
            row=row,
            prompt=prompt,
        )
    return results


def validate_resume_result(
    *,
    result: BenchmarkResult,
    provider: str,
    question_set_id: str,
    row: TruthfulQARow,
    prompt: BinaryPrompt,
) -> None:
    if result.provider != provider:
        raise ValueError(
            f"Cannot resume {provider} run from {result.provider} result "
            f"for condition={result.condition_id}, row_id={result.row_id}, order={result.order}."
        )
    if result.question_set_id != question_set_id:
        raise ValueError(
            "Cannot resume from result with mismatched question set "
            f"for row_id={result.row_id}: expected {question_set_id!r}, found {result.question_set_id!r}."
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
    return (result.condition_id, result.row_id, result.order)


def validate_conditions(*, provider: str, conditions: list[RunCondition]) -> None:
    if not conditions:
        raise ValueError("At least one run condition is required.")
    seen: set[str] = set()
    for condition in conditions:
        if condition.provider != provider:
            raise ValueError(
                f"Condition {condition.condition_id} uses provider {condition.provider}, "
                f"but run provider is {provider}."
            )
        if condition.condition_id in seen:
            raise ValueError(f"Duplicate condition_id: {condition.condition_id}")
        seen.add(condition.condition_id)
