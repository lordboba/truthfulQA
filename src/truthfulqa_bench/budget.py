from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .results import BenchmarkResult


@dataclass(frozen=True)
class BudgetProjection:
    provider: str
    pilot_rows: int
    total_rows: int
    observed_requests: int
    projected_requests: int
    observed_cost_usd: float
    projected_cost_usd: float
    budget_usd: float
    budget_gate_ratio: float

    @property
    def passes(self) -> bool:
        return self.projected_cost_usd <= self.budget_usd * self.budget_gate_ratio


def project_budget(
    provider: str,
    pilot_results: list[BenchmarkResult],
    pilot_rows: int,
    total_rows: int,
    budget_usd: float,
    budget_gate_ratio: float = 0.8,
) -> BudgetProjection:
    if pilot_rows <= 0:
        raise ValueError("pilot_rows must be positive.")
    observed_cost = sum(row.estimated_cost_usd for row in pilot_results if row.provider == provider)
    observed_requests = sum(1 for row in pilot_results if row.provider == provider)
    projected_cost = observed_cost * (total_rows / pilot_rows)
    projected_requests = int(observed_requests * (total_rows / pilot_rows))
    return BudgetProjection(
        provider=provider,
        pilot_rows=pilot_rows,
        total_rows=total_rows,
        observed_requests=observed_requests,
        projected_requests=projected_requests,
        observed_cost_usd=observed_cost,
        projected_cost_usd=projected_cost,
        budget_usd=budget_usd,
        budget_gate_ratio=budget_gate_ratio,
    )


def write_projection(path: Path, projection: BudgetProjection) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(projection), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_projection(path: Path) -> BudgetProjection:
    if not path.exists():
        raise FileNotFoundError(f"Pilot projection not found: {path}. Run `truthfulqa-bench pilot` first.")
    return BudgetProjection(**json.loads(path.read_text(encoding="utf-8")))
