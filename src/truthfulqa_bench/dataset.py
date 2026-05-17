from __future__ import annotations

import csv
import ssl
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import certifi

from .config import EXPECTED_ROW_COUNT, TRUTHFULQA_CSV_URL


@dataclass(frozen=True)
class TruthfulQARow:
    row_id: int
    category: str
    question: str
    best_answer: str
    best_incorrect_answer: str


REQUIRED_COLUMNS = {
    "Category",
    "Question",
    "Best Answer",
    "Best Incorrect Answer",
}


def download_dataset(path: Path, url: str = TRUTHFULQA_CSV_URL) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(url, timeout=30, context=ssl_context) as response:
        content = response.read()
    path.write_bytes(content)


def load_dataset(path: Path, expected_rows: int = EXPECTED_ROW_COUNT) -> list[TruthfulQARow]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Run `truthfulqa-bench prepare` first.")

    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError("Dataset CSV has no header row.")
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"Dataset CSV is missing required columns: {missing_list}")

        rows = [
            TruthfulQARow(
                row_id=index,
                category=raw["Category"],
                question=raw["Question"],
                best_answer=raw["Best Answer"],
                best_incorrect_answer=raw["Best Incorrect Answer"],
            )
            for index, raw in enumerate(reader)
        ]

    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} TruthfulQA rows, found {len(rows)}.")
    return rows
