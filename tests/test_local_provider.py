import csv
import json
import sys
from types import SimpleNamespace

import pytest

from truthfulqa_bench.cli import main
from truthfulqa_bench.conditions import RunCondition, model_conditions
from truthfulqa_bench.costs import pricing_for
from truthfulqa_bench.dataset import TruthfulQARow
from truthfulqa_bench.prompting import BinaryPrompt
from truthfulqa_bench.providers import LocalClient, validate_local_models
from truthfulqa_bench.runner import (
    assert_budget_allows_request,
    iter_pending_attempts,
    max_invalid_retries_for,
)


class FakeChatCompletions:
    def __init__(self):
        self.request = None

    def create(self, **request):
        self.request = request
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="A"))],
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=1),
        )


class FakeChat:
    def __init__(self):
        self.completions = FakeChatCompletions()


class FakeModels:
    def __init__(self, model_ids):
        self._listing = SimpleNamespace(data=[SimpleNamespace(id=mid) for mid in model_ids])

    def list(self):
        return self._listing


class FakeLocalOpenAI:
    latest = None

    def __init__(self, base_url=None, api_key=None, timeout=None):
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.chat = FakeChat()
        self.models = FakeModels(["llama-3.1-8b-instruct"])
        FakeLocalOpenAI.latest = self


def test_local_client_uses_chat_completions(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeLocalOpenAI))
    condition = RunCondition(
        condition_id="local_llama-3.1-8b-instruct",
        provider="local",
        model_id="llama-3.1-8b-instruct",
        model_label="llama-3.1-8b-instruct",
    )

    client = LocalClient(condition=condition, question_set_id="test-set", base_url="http://localhost:1234/v1")
    result = client.complete_choice(
        BinaryPrompt(row_id=0, order="correct_first", prompt="Q?\n\nA. yes\n\nB. no", correct_choice="A"),
        "Health",
    )

    request = FakeLocalOpenAI.latest.chat.completions.request
    assert request["model"] == "llama-3.1-8b-instruct"
    assert request["temperature"] == 0
    assert request["max_tokens"] == 1024
    assert "reasoning" not in request
    assert result.condition_id == "local_llama-3.1-8b-instruct"
    assert result.provider == "local"
    assert result.is_correct is True
    assert result.estimated_cost_usd == 0.0
    assert result.input_tokens == 12
    assert result.output_tokens == 1


def test_local_pricing_is_free():
    assert pricing_for("local", "anything").input_per_mtok == 0.0
    assert pricing_for("local", "anything").output_per_mtok == 0.0


def test_validate_local_models_rejects_missing_model(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeLocalOpenAI))
    with pytest.raises(RuntimeError, match="did not advertise the requested model"):
        validate_local_models(["missing-model"], base_url="http://localhost:1234/v1")


def test_validate_local_models_clear_error_when_unreachable(monkeypatch):
    class ExplodingOpenAI:
        def __init__(self, **_):
            self.models = SimpleNamespace(list=self._raise)

        def _raise(self):
            raise ConnectionError("Connection refused")

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=ExplodingOpenAI))
    with pytest.raises(RuntimeError, match="Could not reach local model server"):
        validate_local_models(["whatever"], base_url="http://localhost:9999/v1")


def test_local_provider_bypasses_budget():
    assert_budget_allows_request(
        provider="local",
        spent=10_000.0,
        budget_usd=1.0,
        request_cost_reserve_usd=1.0,
        in_flight_requests=0,
    )


def test_model_conditions_for_local_requires_models():
    with pytest.raises(ValueError, match="no default models"):
        model_conditions("local", None)


def test_local_condition_id_sanitizes_slashes_and_colons():
    conditions = model_conditions("local", ["meta-llama/Llama-3.1:8b"])
    assert conditions[0].condition_id == "local_meta-llama_Llama-3.1_8b"


def test_max_invalid_retries_caps_local_but_not_openai():
    assert max_invalid_retries_for("local") == 3
    assert max_invalid_retries_for("openai") is None
    assert max_invalid_retries_for("anthropic") is None


def test_iter_pending_skips_attempts_past_retry_cap():
    from collections import Counter

    row = TruthfulQARow(
        row_id=0,
        category="Health",
        question="Q?",
        best_answer="True.",
        best_incorrect_answer="False.",
    )
    condition = RunCondition(
        condition_id="local_x",
        provider="local",
        model_id="x",
        model_label="x",
    )
    invalid_counts = Counter(
        {
            ("local_x", 0, "correct_first"): 3,
            ("local_x", 0, "incorrect_first"): 0,
        }
    )

    pending = list(
        iter_pending_attempts(
            rows=[row],
            conditions=[condition],
            completed=set(),
            invalid_counts=invalid_counts,
            max_invalid_retries=3,
        )
    )

    assert [(r.row_id, p.order) for r, p, _ in pending] == [(0, "incorrect_first")]


def write_dataset(path, rows):
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=["Category", "Question", "Best Answer", "Best Incorrect Answer"],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_cli_local_dry_run_applies_limit_and_base_url(tmp_path, capsys, monkeypatch):
    dataset = tmp_path / "TruthfulQA.csv"
    manifest = tmp_path / "manifest.json"
    rows = [
        TruthfulQARow(row_id=i, category="Health", question=f"Q{i}", best_answer="A", best_incorrect_answer="B")
        for i in range(5)
    ]
    write_dataset(
        dataset,
        [{"Category": r.category, "Question": r.question, "Best Answer": r.best_answer, "Best Incorrect Answer": r.best_incorrect_answer} for r in rows],
    )
    monkeypatch.setattr("truthfulqa_bench.cli.load_dataset", lambda path: rows)
    monkeypatch.delenv("TRUTHFULQA_LOCAL_BASE_URL", raising=False)

    exit_code = main(
        [
            "experiment",
            "--provider",
            "local",
            "--models",
            "llama-3.1-8b-instruct",
            "--dataset",
            str(dataset),
            "--manifest",
            str(manifest),
            "--limit",
            "3",
            "--base-url",
            "http://127.0.0.1:9999/v1",
            "--dry-run",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["rows"] == 3
    assert output["planned_requests"] == 6
    assert output["conditions"][0]["condition_id"] == "local_llama-3.1-8b-instruct"
    import os

    assert os.environ.get("TRUTHFULQA_LOCAL_BASE_URL") == "http://127.0.0.1:9999/v1"
