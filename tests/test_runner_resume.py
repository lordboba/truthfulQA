import pytest

from truthfulqa_bench.dataset import TruthfulQARow
from truthfulqa_bench.results import BenchmarkResult, append_result, load_results
from truthfulqa_bench.runner import run_benchmark


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


def result_for(*, row_id: int, category: str, order: str, correct_choice: str, model: str = "gpt-5.5") -> BenchmarkResult:
    return BenchmarkResult(
        provider="openai",
        model=model,
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

    def __init__(self, model: str):
        self.model = model

    def complete_choice(self, prompt, category: str) -> BenchmarkResult:
        return result_for(
            model=self.model,
            row_id=prompt.row_id,
            category=category,
            order=prompt.order,
            correct_choice=prompt.correct_choice,
        )


def test_run_benchmark_resumes_and_skips_completed_attempts(tmp_path, monkeypatch):
    output_path = tmp_path / "full.jsonl"
    append_result(output_path, result_for(row_id=0, category="Health", order="correct_first", correct_choice="A"))

    monkeypatch.setattr("truthfulqa_bench.runner.make_client", lambda provider, model: RecordingClient(model))

    results = run_benchmark(
        provider="openai",
        models=["gpt-5.5"],
        rows=ROWS,
        output_path=output_path,
        resume=True,
    )

    persisted = load_results(output_path)
    keys = {(result.model, result.row_id, result.order) for result in persisted}
    assert len(results) == 4
    assert len(persisted) == 4
    assert keys == {
        ("gpt-5.5", 0, "correct_first"),
        ("gpt-5.5", 0, "incorrect_first"),
        ("gpt-5.5", 1, "correct_first"),
        ("gpt-5.5", 1, "incorrect_first"),
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

    def fail_client_creation(provider: str, model: str):
        raise AssertionError("client should not be created for a complete resumed run")

    monkeypatch.setattr("truthfulqa_bench.runner.make_client", fail_client_creation)

    results = run_benchmark(
        provider="openai",
        models=["gpt-5.5"],
        rows=ROWS,
        output_path=output_path,
        resume=True,
    )

    assert len(results) == 4


def test_run_benchmark_resume_rejects_duplicate_attempts(tmp_path):
    output_path = tmp_path / "full.jsonl"
    existing = result_for(row_id=0, category="Health", order="correct_first", correct_choice="A")
    append_result(output_path, existing)
    append_result(output_path, existing)

    with pytest.raises(ValueError, match="duplicate attempt"):
        run_benchmark(
            provider="openai",
            models=["gpt-5.5"],
            rows=ROWS,
            output_path=output_path,
            resume=True,
        )


def test_run_benchmark_resume_rejects_mismatched_dataset_metadata(tmp_path):
    output_path = tmp_path / "full.jsonl"
    append_result(output_path, result_for(row_id=0, category="Wrong", order="correct_first", correct_choice="A"))

    with pytest.raises(ValueError, match="mismatched category"):
        run_benchmark(
            provider="openai",
            models=["gpt-5.5"],
            rows=ROWS,
            output_path=output_path,
            resume=True,
        )
