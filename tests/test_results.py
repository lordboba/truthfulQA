from truthfulqa_bench.results import BenchmarkResult, summarize


def make_result(row_id: int, order: str, parsed: str | None, correct: str) -> BenchmarkResult:
    return BenchmarkResult(
        condition_id="openai_gpt-5.5_medium",
        provider="openai",
        model_id="gpt-5.5",
        model_label="GPT-5.5 (medium)",
        reasoning_effort="medium",
        question_set_id="test-set",
        row_id=row_id,
        category="Health",
        order=order,
        correct_choice=correct,
        raw_output=parsed or "Answer: A",
        parsed_choice=parsed,
        is_correct=parsed == correct,
        is_invalid=parsed is None,
        input_tokens=100,
        output_tokens=1,
        estimated_cost_usd=0.001,
        latency_seconds=0.2,
    )


def test_summarize_reports_accuracy_invalids_and_cost():
    summary = summarize(
        [
            make_result(0, "correct_first", "A", "A"),
            make_result(0, "incorrect_first", "B", "B"),
            make_result(1, "correct_first", None, "A"),
        ]
    )

    model_summary = summary["openai_gpt-5.5_medium"]
    assert model_summary["requests"] == 3
    assert model_summary["valid_requests"] == 2
    assert model_summary["accuracy"] == 1.0
    assert model_summary["invalid_rate"] == 1 / 3
    assert model_summary["estimated_cost_usd"] == 0.003


def test_summarize_order_sensitivity_uses_correctness_not_literal_letter():
    summary = summarize(
        [
            make_result(0, "correct_first", "A", "A"),
            make_result(0, "incorrect_first", "B", "B"),
            make_result(1, "correct_first", "A", "A"),
            make_result(1, "incorrect_first", "A", "B"),
        ]
    )

    assert summary["openai_gpt-5.5_medium"]["order_sensitive_rows"] == 1
