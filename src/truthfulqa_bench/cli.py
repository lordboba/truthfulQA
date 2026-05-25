from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .conditions import RunCondition, model_conditions
from .config import (
    DEFAULT_DATA_PATH,
    DEFAULT_EXPERIMENT_MANIFEST,
    DEFAULT_FULL_RESULTS,
    DEFAULT_PILOT_PROJECTION,
    DEFAULT_PILOT_RESULTS,
    LOCAL_BASE_URL_ENV,
)
from .dataset import TruthfulQARow, download_dataset, load_dataset
from .manifest import build_manifest, write_manifest
from .prompting import build_two_order_prompts
from .providers import validate_model_access
from .results import load_results, summarize
from .runner import assert_full_run_allowed, run_benchmark, run_pilot


PROVIDER_CHOICES = ("anthropic", "openai", "local")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="truthfulqa-bench")
    subparsers = parser.add_subparsers(required=True)

    prepare = subparsers.add_parser("prepare", help="Download and validate TruthfulQA.csv.")
    prepare.add_argument("--dataset", type=Path, default=Path(DEFAULT_DATA_PATH))
    prepare.set_defaults(func=cmd_prepare)

    pilot = subparsers.add_parser("pilot", help="Run a paid pilot and write a budget projection.")
    add_run_args(pilot, default_output=DEFAULT_PILOT_RESULTS)
    pilot.add_argument("--pilot-size", type=int, default=20)
    pilot.add_argument("--projection", type=Path, default=Path(DEFAULT_PILOT_PROJECTION))
    pilot.set_defaults(func=cmd_pilot)

    run = subparsers.add_parser("run", help="Run the full paid benchmark after pilot budget validation.")
    add_run_args(run, default_output=DEFAULT_FULL_RESULTS)
    run.add_argument("--projection", type=Path, default=Path(DEFAULT_PILOT_PROJECTION))
    run.set_defaults(func=cmd_run)

    experiment = subparsers.add_parser("experiment", help="Write a manifest and optionally run the controlled benchmark.")
    add_run_args(experiment, default_output=DEFAULT_FULL_RESULTS)
    experiment.add_argument("--manifest", type=Path, default=Path(DEFAULT_EXPERIMENT_MANIFEST))
    experiment.add_argument("--dry-run", action="store_true", help="Print the controlled plan without paid API calls.")
    experiment.set_defaults(func=cmd_experiment)

    report = subparsers.add_parser("report", help="Summarize a JSONL results file.")
    report.add_argument("--results", type=Path, required=True)
    report.set_defaults(func=cmd_report)

    status = subparsers.add_parser("status", help="Show controlled-run progress for a JSONL results file.")
    status.add_argument("--provider", choices=PROVIDER_CHOICES, required=True)
    status.add_argument("--results", type=Path, required=True)
    status.add_argument("--dataset", type=Path, default=Path(DEFAULT_DATA_PATH))
    status.add_argument("--models", nargs="+")
    status.add_argument("--limit", type=int, help="Restrict status comparison to the first N dataset rows.")
    status.set_defaults(func=cmd_status)

    return parser


def add_run_args(parser: argparse.ArgumentParser, *, default_output: str) -> None:
    parser.add_argument("--provider", choices=PROVIDER_CHOICES, required=True)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--dataset", type=Path, default=Path(DEFAULT_DATA_PATH))
    parser.add_argument("--output", type=Path, default=Path(default_output))
    parser.add_argument("--openai-budget", type=float, default=100.0)
    parser.add_argument("--anthropic-budget", type=float, default=100.0)
    parser.add_argument("--resume", action="store_true", help="Continue an interrupted run from an existing JSONL file.")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip provider model access validation.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of concurrent provider requests to run from this process.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Use only the first N dataset rows (useful for piloting local runs).",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        help="Override the local provider's OpenAI-compatible base URL (e.g. http://localhost:1234/v1).",
    )


def cmd_prepare(args: argparse.Namespace) -> int:
    download_dataset(args.dataset)
    rows = load_dataset(args.dataset)
    print(f"Downloaded and validated {len(rows)} TruthfulQA rows at {args.dataset}.")
    return 0


def cmd_pilot(args: argparse.Namespace) -> int:
    apply_local_base_url(args)
    rows = limit_rows(load_dataset(args.dataset), getattr(args, "limit", None))
    if args.pilot_size <= 0 or args.pilot_size > len(rows):
        raise SystemExit(f"--pilot-size must be between 1 and {len(rows)}.")
    conditions = resolve_conditions(args.provider, args.models)
    manifest = build_manifest(dataset_path=args.dataset, rows=rows[: args.pilot_size], conditions=conditions)
    if not args.skip_preflight:
        validate_model_access(conditions, local_base_url_override=getattr(args, "base_url", None))
    projection = run_pilot(
        provider=args.provider,
        conditions=conditions,
        rows=rows,
        question_set_id=manifest.question_set_id,
        pilot_size=args.pilot_size,
        output_path=args.output,
        projection_path=args.projection,
        budget_usd=budget_for(args),
        resume=args.resume,
        max_workers=args.max_workers,
    )
    print_projection(projection)
    if not projection.passes:
        raise SystemExit(1)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    apply_local_base_url(args)
    rows = limit_rows(load_dataset(args.dataset), getattr(args, "limit", None))
    if args.provider != "local":
        projection = assert_full_run_allowed(projection_path=args.projection)
        if projection.provider != args.provider:
            raise SystemExit(
                f"Pilot projection was for {projection.provider}, but requested full run for {args.provider}."
            )
    conditions = resolve_conditions(args.provider, args.models)
    manifest = build_manifest(dataset_path=args.dataset, rows=rows, conditions=conditions)
    if not args.skip_preflight:
        validate_model_access(conditions, local_base_url_override=getattr(args, "base_url", None))
    results = run_benchmark(
        provider=args.provider,
        conditions=conditions,
        rows=rows,
        question_set_id=manifest.question_set_id,
        output_path=args.output,
        budget_usd=budget_for(args),
        resume=args.resume,
        max_workers=args.max_workers,
    )
    print(json.dumps(summarize(results), indent=2, sort_keys=True))
    return 0


def cmd_experiment(args: argparse.Namespace) -> int:
    apply_local_base_url(args)
    rows = limit_rows(load_dataset(args.dataset), getattr(args, "limit", None))
    conditions = resolve_conditions(args.provider, args.models)
    manifest = build_manifest(dataset_path=args.dataset, rows=rows, conditions=conditions)
    write_manifest(args.manifest, manifest)
    planned_requests = len(rows) * len(conditions) * 2
    if args.dry_run:
        print(
            json.dumps(
                {
                    "provider": args.provider,
                    "question_set_id": manifest.question_set_id,
                    "rows": len(rows),
                    "conditions": [condition.to_json() for condition in conditions],
                    "planned_requests": planned_requests,
                    "manifest": str(args.manifest),
                    "anti_cheat_policy": manifest.anti_cheat_policy,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if not args.skip_preflight:
        validate_model_access(conditions, local_base_url_override=getattr(args, "base_url", None))
    results = run_benchmark(
        provider=args.provider,
        conditions=conditions,
        rows=rows,
        question_set_id=manifest.question_set_id,
        output_path=args.output,
        budget_usd=budget_for(args),
        resume=args.resume,
        max_workers=args.max_workers,
    )
    print(json.dumps(summarize(results), indent=2, sort_keys=True))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    results = load_results(args.results)
    print(json.dumps(summarize(results), indent=2, sort_keys=True))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    rows = limit_rows(load_dataset(args.dataset), getattr(args, "limit", None))
    conditions = resolve_conditions(args.provider, args.models)
    results = load_results(args.results)
    expected_by_condition = len(rows) * 2
    summary = summarize(results)
    valid_attempts_by_condition = {}
    for condition in conditions:
        expected_keys = {
            (row.row_id, prompt.order)
            for row in rows
            for prompt in build_two_order_prompts(row)
        }
        valid_keys = {
            (result.row_id, result.order)
            for result in results
            if result.condition_id == condition.condition_id and not result.is_invalid
        }
        valid_attempts_by_condition[condition.condition_id] = {
            "expected": len(expected_keys),
            "valid": len(valid_keys & expected_keys),
            "missing": len(expected_keys - valid_keys),
        }
    status = {
        "provider": args.provider,
        "results": str(args.results),
        "rows": len(rows),
        "expected_requests": expected_by_condition * len(conditions),
        "actual_requests": len(results),
        "estimated_cost_usd": sum(row.estimated_cost_usd for row in results),
        "conditions": {},
    }
    for condition in conditions:
        condition_summary = summary.get(condition.condition_id)
        actual = int(condition_summary["requests"]) if condition_summary else 0
        valid_attempts = valid_attempts_by_condition[condition.condition_id]
        status["conditions"][condition.condition_id] = {
            "model_id": condition.model_id,
            "reasoning_effort": condition.reasoning_effort,
            "expected_requests": expected_by_condition,
            "actual_requests": actual,
            "valid_attempts": valid_attempts["valid"],
            "missing_valid_attempts": valid_attempts["missing"],
            "complete": valid_attempts["missing"] == 0,
            "summary": condition_summary,
        }
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


def resolve_conditions(provider: str, models: list[str] | None) -> list[RunCondition]:
    return model_conditions(provider, models)


def apply_local_base_url(args: argparse.Namespace) -> None:
    base_url = getattr(args, "base_url", None)
    if base_url and getattr(args, "provider", None) == "local":
        os.environ[LOCAL_BASE_URL_ENV] = base_url


def limit_rows(rows: list[TruthfulQARow], limit: int | None) -> list[TruthfulQARow]:
    if limit is None:
        return rows
    if limit <= 0 or limit > len(rows):
        raise SystemExit(f"--limit must be between 1 and {len(rows)}.")
    return rows[:limit]


def budget_for(args: argparse.Namespace) -> float | None:
    if args.provider == "local":
        return None
    return args.anthropic_budget if args.provider == "anthropic" else args.openai_budget


def print_projection(projection) -> None:
    status = "passes" if projection.passes else "fails"
    print(
        f"Pilot projection {status}: observed ${projection.observed_cost_usd:.4f}; "
        f"projected full run ${projection.projected_cost_usd:.4f}; "
        f"budget gate ${projection.budget_usd * projection.budget_gate_ratio:.2f}."
    )


if __name__ == "__main__":
    raise SystemExit(main())
