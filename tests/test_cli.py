import csv
import json

from truthfulqa_bench.cli import main
from truthfulqa_bench.dataset import TruthfulQARow
from truthfulqa_bench.results import BenchmarkResult, append_result


def write_dataset(path, rows):
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=["Category", "Question", "Best Answer", "Best Incorrect Answer"],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_experiment_dry_run_writes_manifest_and_request_count(tmp_path, capsys, monkeypatch):
    dataset = tmp_path / "TruthfulQA.csv"
    manifest = tmp_path / "manifest.json"
    write_dataset(
        dataset,
        [
            {
                "Category": "Health",
                "Question": "Which is true?",
                "Best Answer": "True.",
                "Best Incorrect Answer": "False.",
            }
        ],
    )
    monkeypatch.setattr(
        "truthfulqa_bench.cli.load_dataset",
        lambda path: [
            TruthfulQARow(
                row_id=0,
                category="Health",
                question="Which is true?",
                best_answer="True.",
                best_incorrect_answer="False.",
            )
        ],
    )

    exit_code = main(
        [
            "experiment",
            "--provider",
            "openai",
            "--dataset",
            str(dataset),
            "--manifest",
            str(manifest),
            "--dry-run",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    manifest_json = json.loads(manifest.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert output["planned_requests"] == 6
    assert [condition["reasoning_effort"] for condition in output["conditions"]] == ["high", "medium", "low"]
    assert output["question_set_id"] == manifest_json["question_set_id"]
    assert "cannot prevent public benchmark memorization" in output["anti_cheat_policy"]


def test_status_reports_expected_and_actual_requests(tmp_path, capsys, monkeypatch):
    dataset = tmp_path / "TruthfulQA.csv"
    results = tmp_path / "results.jsonl"
    row = TruthfulQARow(
        row_id=0,
        category="Health",
        question="Which is true?",
        best_answer="True.",
        best_incorrect_answer="False.",
    )
    write_dataset(
        dataset,
        [
            {
                "Category": row.category,
                "Question": row.question,
                "Best Answer": row.best_answer,
                "Best Incorrect Answer": row.best_incorrect_answer,
            }
        ],
    )
    monkeypatch.setattr("truthfulqa_bench.cli.load_dataset", lambda path: [row])
    append_result(
        results,
        BenchmarkResult(
            condition_id="anthropic_claude-opus-4-7",
            provider="anthropic",
            model_id="claude-opus-4-7",
            model_label="Claude Opus 4.7",
            reasoning_effort=None,
            question_set_id="test-set",
            row_id=0,
            category="Health",
            order="correct_first",
            correct_choice="A",
            raw_output="A",
            parsed_choice="A",
            is_correct=True,
            is_invalid=False,
            input_tokens=10,
            output_tokens=1,
            estimated_cost_usd=0.001,
            latency_seconds=0.1,
        ),
    )

    exit_code = main(
        [
            "status",
            "--provider",
            "anthropic",
            "--dataset",
            str(dataset),
            "--results",
            str(results),
        ]
    )

    status = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert status["expected_requests"] == 4
    assert status["actual_requests"] == 1
    assert status["conditions"]["anthropic_claude-opus-4-7"]["complete"] is False
    assert status["conditions"]["anthropic_claude-opus-4-7"]["valid_attempts"] == 1
    assert status["conditions"]["anthropic_claude-opus-4-7"]["missing_valid_attempts"] == 1
    assert status["conditions"]["anthropic_claude-sonnet-4-6"]["actual_requests"] == 0


def test_status_completion_uses_valid_attempts_not_total_retry_rows(tmp_path, capsys, monkeypatch):
    dataset = tmp_path / "TruthfulQA.csv"
    results = tmp_path / "results.jsonl"
    row = TruthfulQARow(
        row_id=0,
        category="Health",
        question="Which is true?",
        best_answer="True.",
        best_incorrect_answer="False.",
    )
    write_dataset(
        dataset,
        [
            {
                "Category": row.category,
                "Question": row.question,
                "Best Answer": row.best_answer,
                "Best Incorrect Answer": row.best_incorrect_answer,
            }
        ],
    )
    monkeypatch.setattr("truthfulqa_bench.cli.load_dataset", lambda path: [row])
    for order, correct_choice in [("correct_first", "A"), ("incorrect_first", "B")]:
        append_result(
            results,
            BenchmarkResult(
                condition_id="openai_gpt-5.5_high",
                provider="openai",
                model_id="gpt-5.5",
                model_label="GPT-5.5 (high)",
                reasoning_effort="high",
                question_set_id="test-set",
                row_id=0,
                category="Health",
                order=order,
                correct_choice=correct_choice,
                raw_output="",
                parsed_choice=None,
                is_correct=False,
                is_invalid=True,
                input_tokens=10,
                output_tokens=512,
                estimated_cost_usd=0.01,
                latency_seconds=0.1,
            ),
        )
        append_result(
            results,
            BenchmarkResult(
                condition_id="openai_gpt-5.5_high",
                provider="openai",
                model_id="gpt-5.5",
                model_label="GPT-5.5 (high)",
                reasoning_effort="high",
                question_set_id="test-set",
                row_id=0,
                category="Health",
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
            ),
        )

    exit_code = main(
        [
            "status",
            "--provider",
            "openai",
            "--dataset",
            str(dataset),
            "--results",
            str(results),
        ]
    )

    status = json.loads(capsys.readouterr().out)
    condition = status["conditions"]["openai_gpt-5.5_high"]
    assert exit_code == 0
    assert condition["actual_requests"] == 4
    assert condition["expected_requests"] == 2
    assert condition["valid_attempts"] == 2
    assert condition["missing_valid_attempts"] == 0
    assert condition["complete"] is True
