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
truthfulqa-bench experiment --provider openai --dry-run
truthfulqa-bench experiment --provider anthropic --dry-run
truthfulqa-bench report --results results/pilot.jsonl
pytest
```

## Paid Benchmark Flow

Run a small pilot first:

```bash
truthfulqa-bench pilot \
  --provider anthropic \
  --pilot-size 20 \
  --anthropic-budget 100
```

Then run the full benchmark only after the pilot projection passes the budget gate:

```bash
truthfulqa-bench run \
  --provider anthropic \
  --anthropic-budget 100
```

If a paid pilot or full run is interrupted, rerun the same command with `--resume`.
Resume mode validates the existing JSONL file against the requested provider,
models, dataset rows, answer order, and correct-choice metadata before appending
only the missing prompts:

```bash
truthfulqa-bench run \
  --provider anthropic \
  --anthropic-budget 100 \
  --resume
```

Default Anthropic conditions are `claude-opus-4-7` and `claude-sonnet-4-6`.
Default OpenAI conditions are `gpt-5.5` at `high`, `medium`, and `low`
reasoning effort. Those OpenAI conditions share a model ID but have distinct
condition IDs, so resume and summaries do not collide.

```bash
truthfulqa-bench pilot \
  --provider openai \
  --pilot-size 20 \
  --openai-budget 100
```

The controlled experiment command writes a manifest with the dataset hash, row
IDs, prompt hash, conditions, pricing snapshot, and anti-cheating policy before
running paid calls:

```bash
truthfulqa-bench experiment \
  --provider openai \
  --openai-budget 100
```

Pass `--dry-run` to inspect the exact question set and request count without API
calls. Paid commands validate model access before running unless
`--skip-preflight` is provided.

For the full OpenAI run, use the checked-in launch script so interruptions
resume safely:

```bash
scripts/run_openai_full.sh
```

The launch script defaults to one locked writer process, `OPENAI_MAX_WORKERS=24`,
`TRUTHFULQA_OPENAI_MIN_REQUEST_INTERVAL_SECONDS=0.25`, and
`TRUTHFULQA_OPENAI_MAX_OUTPUT_TOKENS=8192`. Override those environment variables
if the account limit or output budget changes. The result schema, prompt
controls, and resume validation are unchanged by concurrency.

## Outputs

Results are JSON Lines under `results/` by default. Each line records condition
ID, provider, model ID, model label, reasoning effort, question set ID, row ID,
answer order, raw output, parsed output, correctness, token usage, cost
estimate, latency, and invalid-output status.

The report command summarizes accuracy, invalid-output rate, order sensitivity, token usage, and estimated cost:

```bash
truthfulqa-bench report --results results/full.jsonl
```

The status command compares a result file against the expected controlled run:

```bash
truthfulqa-bench status --provider anthropic --results results/anthropic_full.jsonl
```

## Budget Policy

- Default pilot size: 20 questions.
- Default provider budget: `$100`.
- Full runs require pilot-projected cost below 80% of the relevant provider cap.
- Runtime budget checks reserve `$1` before scheduling each request. Exact
  billed tokens are only known after the API response, so this is a conservative
  pre-call gate rather than a claim of perfect billing control.
- Resume mode continues an interrupted JSONL result file only after validating it
  matches the requested run condition, question set, row metadata, answer order,
  and correct-choice metadata.
- API keys are required before any paid command starts.
- Missing keys abort before API clients are created.

## Anti-Cheating Boundary

The runner does not provide web search, file search, retrieval, tools, repo URLs,
or external context to model calls. This prevents active lookup during the
benchmark. It cannot prevent a frontier model from having memorized public
TruthfulQA content during training, so reports should describe that limitation
honestly.
