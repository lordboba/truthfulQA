from truthfulqa_bench.budget import project_budget
from truthfulqa_bench.results import BenchmarkResult


def result(cost: float) -> BenchmarkResult:
    return BenchmarkResult(
        condition_id="anthropic_claude-opus-4-7",
        provider="anthropic",
        model_id="claude-opus-4-7",
        model_label="Claude Opus 4.7",
        reasoning_effort=None,
        question_set_id="test-set",
        row_id=0,
        category="Misconceptions",
        order="correct_first",
        correct_choice="A",
        raw_output="A",
        parsed_choice="A",
        is_correct=True,
        is_invalid=False,
        input_tokens=10,
        output_tokens=1,
        estimated_cost_usd=cost,
        latency_seconds=0.1,
    )


def test_budget_projection_passes_under_gate():
    projection = project_budget(
        provider="anthropic",
        pilot_results=[result(0.01)],
        pilot_rows=20,
        total_rows=790,
        budget_usd=100,
    )

    assert projection.projected_cost_usd == 0.395
    assert projection.passes


def test_budget_projection_fails_over_gate():
    projection = project_budget(
        provider="anthropic",
        pilot_results=[result(3.0)],
        pilot_rows=20,
        total_rows=790,
        budget_usd=100,
    )

    assert not projection.passes
