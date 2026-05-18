import pytest

from truthfulqa_bench.dataset import TruthfulQARow
from truthfulqa_bench.results import BenchmarkResult, append_result, load_results
from truthfulqa_bench.runner import acquire_run_lock, run_benchmark
from truthfulqa_bench.conditions import RunCondition


ROWS = [
    TruthfulQARow(
        row_id=0,
        category="Health",
        question="Which answer is true?",
        best_answer="The true answer.",
        best_incorrect_answer="The false answer.",
    ),
    TruthfulQARow(
        row_id=1,
        category="Science",
        question="Which science answer is true?",
        best_answer="The scientific answer.",
        best_incorrect_answer="The unscientific answer.",
    ),
]

CONDITION = RunCondition(
    condition_id="openai_gpt-5.5_medium",
    provider="openai",
    model_id="gpt-5.5",
    model_label="GPT-5.5 (medium)",
    reasoning_effort="medium",
)
QUESTION_SET_ID = "test-set"


def result_for(
    *,
    row_id: int,
    category: str,
    order: str,
    correct_choice: str,
    condition: RunCondition = CONDITION,
    question_set_id: str = QUESTION_SET_ID,
) -> BenchmarkResult:
    return BenchmarkResult(
        condition_id=condition.condition_id,
        provider="openai",
        model_id=condition.model_id,
        model_label=condition.model_label,
        reasoning_effort=condition.reasoning_effort,
        question_set_id=question_set_id,
        row_id=row_id,
        category=category,
        order=order,
        correct_choice=correct_choice,
        raw_output=correct_choice,
        parsed_choice=correct_choice,
        is_correct=True,
        is_invalid=False,
        input_tokens=10,
        output_tokens=1,
        estimated_cost_usd=0.001,
        latency_seconds=0.1,
    )


class RecordingClient:
    provider = "openai"

    def __init__(self, condition: RunCondition):
        self.condition = condition

    def complete_choice(self, prompt, category: str) -> BenchmarkResult:
        return result_for(
            condition=self.condition,
            row_id=prompt.row_id,
            category=category,
            order=prompt.order,
            correct_choice=prompt.correct_choice,
        )


class InvalidThenValidClient:
    provider = "openai"

    def __init__(self, condition: RunCondition):
        self.condition = condition
        self.calls_by_attempt = {}

    def complete_choice(self, prompt, category: str) -> BenchmarkResult:
        key = (prompt.row_id, prompt.order)
        calls = self.calls_by_attempt.get(key, 0)
        self.calls_by_attempt[key] = calls + 1
        if calls == 0:
            return BenchmarkResult(
                condition_id=self.condition.condition_id,
                provider="openai",
                model_id=self.condition.model_id,
                model_label=self.condition.model_label,
                reasoning_effort=self.condition.reasoning_effort,
                question_set_id=QUESTION_SET_ID,
                row_id=prompt.row_id,
                category=category,
                order=prompt.order,
                correct_choice=prompt.correct_choice,
                raw_output="",
                parsed_choice=None,
                is_correct=False,
                is_invalid=True,
                input_tokens=10,
                output_tokens=128,
                estimated_cost_usd=0.004,
                latency_seconds=0.1,
            )
        return result_for(
            condition=self.condition,
            row_id=prompt.row_id,
            category=category,
            order=prompt.order,
            correct_choice=prompt.correct_choice,
        )


def test_run_benchmark_resumes_and_skips_completed_attempts(tmp_path, monkeypatch):
    output_path = tmp_path / "full.jsonl"
    append_result(output_path, result_for(row_id=0, category="Health", order="correct_first", correct_choice="A"))

    monkeypatch.setattr("truthfulqa_bench.runner.make_client", lambda condition, question_set_id: RecordingClient(condition))

    results = run_benchmark(
        provider="openai",
        conditions=[CONDITION],
        rows=ROWS,
        question_set_id=QUESTION_SET_ID,
        output_path=output_path,
        resume=True,
    )

    persisted = load_results(output_path)
    keys = {(result.condition_id, result.row_id, result.order) for result in persisted}
    assert len(results) == 4
    assert len(persisted) == 4
    assert keys == {
        ("openai_gpt-5.5_medium", 0, "correct_first"),
        ("openai_gpt-5.5_medium", 0, "incorrect_first"),
        ("openai_gpt-5.5_medium", 1, "correct_first"),
        ("openai_gpt-5.5_medium", 1, "incorrect_first"),
    }


def test_run_benchmark_parallel_resumes_without_duplicate_attempts(tmp_path, monkeypatch):
    output_path = tmp_path / "full.jsonl"
    append_result(output_path, result_for(row_id=0, category="Health", order="correct_first", correct_choice="A"))

    monkeypatch.setattr("truthfulqa_bench.runner.make_client", lambda condition, question_set_id: RecordingClient(condition))

    results = run_benchmark(
        provider="openai",
        conditions=[CONDITION],
        rows=ROWS,
        question_set_id=QUESTION_SET_ID,
        output_path=output_path,
        resume=True,
        max_workers=3,
    )

    persisted = load_results(output_path)
    keys = [(result.condition_id, result.row_id, result.order) for result in persisted]
    assert len(results) == 4
    assert len(persisted) == 4
    assert len(keys) == len(set(keys))
    assert set(keys) == {
        ("openai_gpt-5.5_medium", 0, "correct_first"),
        ("openai_gpt-5.5_medium", 0, "incorrect_first"),
        ("openai_gpt-5.5_medium", 1, "correct_first"),
        ("openai_gpt-5.5_medium", 1, "incorrect_first"),
    }


def test_run_benchmark_resume_does_not_create_client_when_complete(tmp_path, monkeypatch):
    output_path = tmp_path / "full.jsonl"
    for row in ROWS:
        append_result(
            output_path,
            result_for(row_id=row.row_id, category=row.category, order="correct_first", correct_choice="A"),
        )
        append_result(
            output_path,
            result_for(row_id=row.row_id, category=row.category, order="incorrect_first", correct_choice="B"),
        )

    def fail_client_creation(condition: RunCondition, question_set_id: str):
        raise AssertionError("client should not be created for a complete resumed run")

    monkeypatch.setattr("truthfulqa_bench.runner.make_client", fail_client_creation)

    results = run_benchmark(
        provider="openai",
        conditions=[CONDITION],
        rows=ROWS,
        question_set_id=QUESTION_SET_ID,
        output_path=output_path,
        resume=True,
    )

    assert len(results) == 4


def test_run_benchmark_resume_rejects_duplicate_valid_attempts(tmp_path):
    output_path = tmp_path / "full.jsonl"
    existing = result_for(row_id=0, category="Health", order="correct_first", correct_choice="A")
    append_result(output_path, existing)
    append_result(output_path, existing)

    with pytest.raises(ValueError, match="duplicate valid attempt"):
        run_benchmark(
            provider="openai",
            conditions=[CONDITION],
            rows=ROWS,
            question_set_id=QUESTION_SET_ID,
            output_path=output_path,
            resume=True,
        )


def test_run_benchmark_rejects_concurrent_writer_for_same_output(tmp_path):
    output_path = tmp_path / "full.jsonl"

    with acquire_run_lock(output_path):
        with pytest.raises(RuntimeError, match="already writing"):
            run_benchmark(
                provider="openai",
                conditions=[CONDITION],
                rows=ROWS,
                question_set_id=QUESTION_SET_ID,
                output_path=output_path,
            )


def test_run_benchmark_resume_rejects_mismatched_dataset_metadata(tmp_path):
    output_path = tmp_path / "full.jsonl"
    append_result(output_path, result_for(row_id=0, category="Wrong", order="correct_first", correct_choice="A"))

    with pytest.raises(ValueError, match="mismatched category"):
        run_benchmark(
            provider="openai",
            conditions=[CONDITION],
            rows=ROWS,
            question_set_id=QUESTION_SET_ID,
            output_path=output_path,
            resume=True,
        )


def test_run_benchmark_distinguishes_reasoning_conditions(tmp_path, monkeypatch):
    output_path = tmp_path / "full.jsonl"
    high = RunCondition("openai_gpt-5.5_high", "openai", "gpt-5.5", "GPT-5.5 (high)", "high")
    low = RunCondition("openai_gpt-5.5_low", "openai", "gpt-5.5", "GPT-5.5 (low)", "low")

    monkeypatch.setattr("truthfulqa_bench.runner.make_client", lambda condition, question_set_id: RecordingClient(condition))

    results = run_benchmark(
        provider="openai",
        conditions=[high, low],
        rows=ROWS[:1],
        question_set_id=QUESTION_SET_ID,
        output_path=output_path,
    )

    keys = {(result.condition_id, result.row_id, result.order) for result in results}
    assert keys == {
        ("openai_gpt-5.5_high", 0, "correct_first"),
        ("openai_gpt-5.5_high", 0, "incorrect_first"),
        ("openai_gpt-5.5_low", 0, "correct_first"),
        ("openai_gpt-5.5_low", 0, "incorrect_first"),
    }


def test_run_benchmark_interleaves_conditions_by_row_and_order(tmp_path, monkeypatch):
    output_path = tmp_path / "full.jsonl"
    high = RunCondition("openai_gpt-5.5_high", "openai", "gpt-5.5", "GPT-5.5 (high)", "high")
    low = RunCondition("openai_gpt-5.5_low", "openai", "gpt-5.5", "GPT-5.5 (low)", "low")

    monkeypatch.setattr("truthfulqa_bench.runner.make_client", lambda condition, question_set_id: RecordingClient(condition))

    results = run_benchmark(
        provider="openai",
        conditions=[high, low],
        rows=ROWS[:1],
        question_set_id=QUESTION_SET_ID,
        output_path=output_path,
    )

    assert [(result.condition_id, result.order) for result in results] == [
        ("openai_gpt-5.5_high", "correct_first"),
        ("openai_gpt-5.5_low", "correct_first"),
        ("openai_gpt-5.5_high", "incorrect_first"),
        ("openai_gpt-5.5_low", "incorrect_first"),
    ]


def test_run_benchmark_resume_retries_invalid_attempts(tmp_path, monkeypatch):
    output_path = tmp_path / "full.jsonl"
    append_result(
        output_path,
        BenchmarkResult(
            condition_id=CONDITION.condition_id,
            provider="openai",
            model_id=CONDITION.model_id,
            model_label=CONDITION.model_label,
            reasoning_effort=CONDITION.reasoning_effort,
            question_set_id=QUESTION_SET_ID,
            row_id=0,
            category="Health",
            order="correct_first",
            correct_choice="A",
            raw_output="",
            parsed_choice=None,
            is_correct=False,
            is_invalid=True,
            input_tokens=10,
            output_tokens=128,
            estimated_cost_usd=0.004,
            latency_seconds=0.1,
        ),
    )
    monkeypatch.setattr("truthfulqa_bench.runner.make_client", lambda condition, question_set_id: RecordingClient(condition))

    results = run_benchmark(
        provider="openai",
        conditions=[CONDITION],
        rows=ROWS[:1],
        question_set_id=QUESTION_SET_ID,
        output_path=output_path,
        resume=True,
    )

    assert len(results) == 3
    assert sum(result.is_invalid for result in results) == 1
    assert sum(
        result.row_id == 0 and result.order == "correct_first" and not result.is_invalid for result in results
    ) == 1

    resumed = run_benchmark(
        provider="openai",
        conditions=[CONDITION],
        rows=ROWS[:1],
        question_set_id=QUESTION_SET_ID,
        output_path=output_path,
        resume=True,
    )
    assert len(resumed) == 3


def test_run_benchmark_retries_invalid_attempts_before_returning(tmp_path, monkeypatch):
    output_path = tmp_path / "full.jsonl"
    client = InvalidThenValidClient(CONDITION)

    monkeypatch.setattr("truthfulqa_bench.runner.make_client", lambda condition, question_set_id: client)

    results = run_benchmark(
        provider="openai",
        conditions=[CONDITION],
        rows=ROWS[:1],
        question_set_id=QUESTION_SET_ID,
        output_path=output_path,
    )

    keys = {(result.condition_id, result.row_id, result.order) for result in results if not result.is_invalid}
    assert keys == {
        ("openai_gpt-5.5_medium", 0, "correct_first"),
        ("openai_gpt-5.5_medium", 0, "incorrect_first"),
    }
    assert sum(result.is_invalid for result in results) == 2
    assert len(results) == 4


def test_run_benchmark_stops_before_request_when_budget_reserve_would_exceed(tmp_path, monkeypatch):
    output_path = tmp_path / "full.jsonl"

    def fail_client_creation(condition: RunCondition, question_set_id: str):
        raise AssertionError("client should not be created when budget reserve blocks the first request")

    monkeypatch.setattr("truthfulqa_bench.runner.make_client", fail_client_creation)

    with pytest.raises(RuntimeError, match="reserve would exceed"):
        run_benchmark(
            provider="openai",
            conditions=[CONDITION],
            rows=ROWS[:1],
            question_set_id=QUESTION_SET_ID,
            output_path=output_path,
            budget_usd=0.50,
        )


def test_run_benchmark_parallel_reserves_budget_for_in_flight_requests(tmp_path, monkeypatch):
    output_path = tmp_path / "full.jsonl"

    monkeypatch.setattr("truthfulqa_bench.runner.make_client", lambda condition, question_set_id: RecordingClient(condition))

    with pytest.raises(RuntimeError, match="reserve would exceed"):
        run_benchmark(
            provider="openai",
            conditions=[CONDITION],
            rows=ROWS,
            question_set_id=QUESTION_SET_ID,
            output_path=output_path,
            budget_usd=2.50,
            max_workers=3,
        )
