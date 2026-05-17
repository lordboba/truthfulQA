from __future__ import annotations

import argparse
import json
from pathlib import Path

from .budget import project_budget, write_projection
from .config import (
    DEFAULT_ANTHROPIC_MODELS,
    DEFAULT_DATA_PATH,
    DEFAULT_FULL_RESULTS,
    DEFAULT_OPENAI_MODELS,
    DEFAULT_PILOT_PROJECTION,
    DEFAULT_PILOT_RESULTS,
    EXPECTED_ROW_COUNT,
)
from .dataset import download_dataset, load_dataset
from .results import load_results, summarize
from .runner import assert_full_run_allowed, run_benchmark, run_pilot


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

    report = subparsers.add_parser("report", help="Summarize a JSONL results file.")
    report.add_argument("--results", type=Path, required=True)
    report.set_defaults(func=cmd_report)

    return parser


def add_run_args(parser: argparse.ArgumentParser, *, default_output: str) -> None:
    parser.add_argument("--provider", choices=("anthropic", "openai"), required=True)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--dataset", type=Path, default=Path(DEFAULT_DATA_PATH))
    parser.add_argument("--output", type=Path, default=Path(default_output))
    parser.add_argument("--openai-budget", type=float, default=100.0)
    parser.add_argument("--anthropic-budget", type=float, default=100.0)


def cmd_prepare(args: argparse.Namespace) -> int:
    download_dataset(args.dataset)
    rows = load_dataset(args.dataset)
    print(f"Downloaded and validated {len(rows)} TruthfulQA rows at {args.dataset}.")
    return 0


def cmd_pilot(args: argparse.Namespace) -> int:
    rows = load_dataset(args.dataset)
    if args.pilot_size <= 0 or args.pilot_size > len(rows):
        raise SystemExit(f"--pilot-size must be between 1 and {len(rows)}.")
    models = resolve_models(args.provider, args.models)
    projection = run_pilot(
        provider=args.provider,
        models=models,
        rows=rows,
        pilot_size=args.pilot_size,
        output_path=args.output,
        projection_path=args.projection,
        budget_usd=budget_for(args),
    )
    print_projection(projection)
    if not projection.passes:
        raise SystemExit(1)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    rows = load_dataset(args.dataset)
    projection = assert_full_run_allowed(projection_path=args.projection)
    if projection.provider != args.provider:
        raise SystemExit(
            f"Pilot projection was for {projection.provider}, but requested full run for {args.provider}."
        )
    models = resolve_models(args.provider, args.models)
    results = run_benchmark(provider=args.provider, models=models, rows=rows, output_path=args.output)
    print(json.dumps(summarize(results), indent=2, sort_keys=True))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    results = load_results(args.results)
    print(json.dumps(summarize(results), indent=2, sort_keys=True))
    return 0


def resolve_models(provider: str, models: list[str] | None) -> list[str]:
    if models:
        return models
    if provider == "anthropic":
        return list(DEFAULT_ANTHROPIC_MODELS)
    if provider == "openai":
        return list(DEFAULT_OPENAI_MODELS)
    raise ValueError(f"Unsupported provider: {provider}")


def budget_for(args: argparse.Namespace) -> float:
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
