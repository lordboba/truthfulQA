from __future__ import annotations

import json
import importlib.util
import sys
from dataclasses import asdict
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("analyze_for_paper", ROOT / "scripts" / "analyze_for_paper.py")
assert SPEC is not None
assert SPEC.loader is not None
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


def write_result(path: Path, **overrides: object) -> None:
    row = {
        "condition_id": "cloud_a",
        "provider": "test",
        "model_id": "model",
        "model_label": "Model",
        "reasoning_effort": None,
        "question_set_id": "fixture",
        "row_id": 0,
        "category": "Category 00",
        "order": "correct_first",
        "correct_choice": "A",
        "raw_output": "A",
        "parsed_choice": "A",
        "is_correct": True,
        "is_invalid": False,
        "input_tokens": 10,
        "output_tokens": 1,
        "estimated_cost_usd": 0.01,
        "latency_seconds": 0.5,
    }
    row.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def fixture_conditions() -> list[analysis.ConditionConfig]:
    return [
        analysis.ConditionConfig("cloud_a", "Cloud A", "cloud"),
        analysis.ConditionConfig("cloud_b", "Cloud B", "cloud"),
        analysis.ConditionConfig("local_a", "Local A", "local"),
        analysis.ConditionConfig("local_b", "Local B", "local"),
    ]


def write_fixture_sources(tmp_path: Path) -> list[Path]:
    paths = {
        "cloud_a": tmp_path / "cloud_a.jsonl",
        "cloud_b": tmp_path / "cloud_b.jsonl",
        "local_a": tmp_path / "local_a.jsonl",
        "local_b": tmp_path / "local_b.jsonl",
    }
    for row_id in range(11):
        category = f"Category {row_id:02d}"
        for order in ("correct_first", "incorrect_first"):
            for condition_id, path in paths.items():
                is_cloud = condition_id.startswith("cloud")
                if is_cloud:
                    is_correct = row_id == 10 or (1 <= row_id <= 9 and order == "correct_first")
                elif condition_id == "local_a":
                    is_correct = row_id % 2 == 0
                else:
                    is_correct = row_id % 3 == 0
                write_result(
                    path,
                    condition_id=condition_id,
                    provider="local" if condition_id.startswith("local") else "test",
                    model_id=condition_id,
                    model_label=condition_id,
                    row_id=row_id,
                    category=category,
                    order=order,
                    is_correct=is_correct,
                    estimated_cost_usd=0.0 if condition_id.startswith("local") else 0.01,
                )
    return list(paths.values())


def test_generate_analysis_writes_cloud_and_local_tables(tmp_path: Path) -> None:
    out_dir = tmp_path / "tables"

    metadata = analysis.generate_analysis(
        result_paths=write_fixture_sources(tmp_path),
        out_dir=out_dir,
        conditions=fixture_conditions(),
    )

    main_results = (out_dir / "main_results.tex").read_text(encoding="utf-8")
    assert "Cloud A" in main_results
    assert "Local A" in main_results
    assert main_results.count(r"\midrule") >= 2

    paired_tests = (out_dir / "paired_tests.tex").read_text(encoding="utf-8")
    assert "Local A vs. Local B" in paired_tests

    cloud_hardest = (out_dir / "hardest_categories.tex").read_text(encoding="utf-8")
    local_hardest = (out_dir / "hardest_categories_local.tex").read_text(encoding="utf-8")
    assert "Category 00" in cloud_hardest
    assert "Category 00" in local_hardest
    assert "Category 10" not in cloud_hardest
    assert "Category 10" not in local_hardest

    assert metadata["conditions"] == 4
    assert metadata["cloud_conditions"] == 2
    assert metadata["local_conditions"] == 2


def test_default_condition_metadata_tracks_all_nine_configured_conditions(tmp_path: Path) -> None:
    source = tmp_path / "all_conditions.jsonl"
    for condition in analysis.CONDITIONS:
        for order in ("correct_first", "incorrect_first"):
            write_result(
                source,
                condition_id=condition.condition_id,
                provider=condition.group,
                model_id=condition.condition_id,
                model_label=condition.label,
                row_id=0,
                order=order,
                is_correct=True,
            )

    metadata = analysis.generate_analysis(result_paths=[source], out_dir=tmp_path / "tables")

    assert metadata["conditions"] == 9
    assert metadata["cloud_conditions"] == 5
    assert metadata["local_conditions"] == 4


def test_condition_config_serializes_for_metadata() -> None:
    config = analysis.ConditionConfig("local_example", "Local Example", "local")

    assert asdict(config) == {
        "condition_id": "local_example",
        "label": "Local Example",
        "group": "local",
    }


def test_format_p_value_preserves_significant_trailing_zeroes() -> None:
    assert analysis.format_p_value(0.00830) == "0.00830"
    assert analysis.format_p_value(0.460) == "0.460"
    assert analysis.format_p_value(4.71e-5) == "4.71e-05"


def test_generate_analysis_rejects_missing_configured_conditions(tmp_path: Path) -> None:
    source = tmp_path / "cloud_only.jsonl"
    for order in ("correct_first", "incorrect_first"):
        write_result(source, condition_id="cloud_a", order=order)

    with pytest.raises(ValueError, match="Missing configured condition"):
        analysis.generate_analysis(
            result_paths=[source],
            out_dir=tmp_path / "tables",
            conditions=fixture_conditions(),
        )


def test_generate_analysis_rejects_duplicate_valid_attempts(tmp_path: Path) -> None:
    source = tmp_path / "duplicates.jsonl"
    for condition in fixture_conditions():
        for order in ("correct_first", "incorrect_first"):
            write_result(source, condition_id=condition.condition_id, order=order)
    write_result(source, condition_id="cloud_a", order="correct_first")

    with pytest.raises(ValueError, match="Duplicate valid attempt key"):
        analysis.generate_analysis(
            result_paths=[source],
            out_dir=tmp_path / "tables",
            conditions=fixture_conditions(),
        )
