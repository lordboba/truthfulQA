# TruthfulQA Benchmark Runner

Local benchmark runner for the current TruthfulQA binary multiple-choice setup.

The runner evaluates each TruthfulQA row as a forced binary choice between `Best Answer` and `Best Incorrect Answer`. It runs both answer orders so results can expose position bias. It does not implement the legacy GPT-3/logprob MC1/MC2 flow.

## Setup

```bash
cd /Users/tylerxiao/Documents/truthfulQA
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

Add API keys only when you are ready to run paid calls:

```bash
cp .env.example .env
# edit .env with OPENAI_API_KEY and/or ANTHROPIC_API_KEY
set -a; source .env; set +a
```

## Safe Offline Commands

These do not call model APIs:

```bash
truthfulqa-bench prepare
truthfulqa-bench report --results results/pilot.jsonl
pytest
```

## Paid Benchmark Flow

Run a small pilot first:

```bash
truthfulqa-bench pilot \
  --provider anthropic \
  --models claude-opus-4-7 claude-sonnet-4-6 \
  --pilot-size 20 \
  --anthropic-budget 100
```

Then run the full benchmark only after the pilot projection passes the budget gate:

```bash
truthfulqa-bench run \
  --provider anthropic \
  --models claude-opus-4-7 claude-sonnet-4-6 \
  --anthropic-budget 100
```

If a paid pilot or full run is interrupted, rerun the same command with `--resume`.
Resume mode validates the existing JSONL file against the requested provider,
models, dataset rows, answer order, and correct-choice metadata before appending
only the missing prompts:

```bash
truthfulqa-bench run \
  --provider anthropic \
  --models claude-opus-4-7 claude-sonnet-4-6 \
  --anthropic-budget 100 \
  --resume
```

OpenAI model IDs are intentionally configurable because latest aliases change:

```bash
truthfulqa-bench pilot \
  --provider openai \
  --models gpt-5.5 gpt-5.4 gpt-5.4-mini \
  --pilot-size 20 \
  --openai-budget 100
```

## Outputs

Results are JSON Lines under `results/` by default. Each line records provider, model, row id, answer order, raw output, parsed output, correctness, token usage, cost estimate, latency, and invalid-output status.

The report command summarizes accuracy, invalid-output rate, order sensitivity, token usage, and estimated cost:

```bash
truthfulqa-bench report --results results/full.jsonl
```

## Budget Policy

- Default pilot size: 20 questions.
- Default provider budget: `$100`.
- Full runs require pilot-projected cost below 80% of the relevant provider cap.
- Resume mode continues an interrupted JSONL result file only after validating it
  matches the requested run.
- API keys are required before any paid command starts.
- Missing keys abort before API clients are created.
