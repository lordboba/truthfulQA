# Controlled TruthfulQA Experiment Status

Last updated: 2026-05-18 00:04 PDT.

## Implemented Controls

- Same-question control is implemented with a frozen `question_set_id`, dataset hash, row IDs, prompt hash, and per-condition attempt keys.
- Paid scheduling is row/order/condition interleaved, so partial OpenAI progress fills comparable reasoning-effort attempts for the same early questions before moving far ahead.
- OpenAI execution now uses a single locked writer process with `--max-workers 24`; JSONL appends still happen from the main process after provider calls complete.
- OpenAI defaults are `gpt-5.5` at `high`, `medium`, and `low` reasoning effort.
- Anthropic defaults are `claude-opus-4-7` and `claude-sonnet-4-6`.
- Model preflight passed for both providers using the local `.env` keys.
- No web search, file search, retrieval, tools, repo URLs, or external context are sent to model calls.
- Public benchmark memorization is not preventable by API-call controls and remains a stated limitation.

## Completed Runs

Anthropic full run completed successfully:

- Results: `results/anthropic_full.jsonl`
- Manifest: `results/anthropic_manifest.json`
- Status: `results/anthropic_full_status.json`
- Report: `results/anthropic_full_report.json`
- Requests: `3160 / 3160`
- Estimated spend: `$1.8564`
- `claude-opus-4-7`: `1580 / 1580`, accuracy `0.9898734177215189`, invalid rate `0.0`
- `claude-sonnet-4-6`: `1580 / 1580`, accuracy `0.9575949367088608`, invalid rate `0.0`

OpenAI full run completed successfully:

- Results: `results/openai_full.jsonl`
- Manifest: `results/openai_manifest.json`
- Status: `results/openai_full_status.json`
- Report: `results/openai_full_report.json`
- Expected valid attempts: `4740 / 4740`
- Total API call rows, including invalid retry history: `5150`
- Estimated spend: `$19.884795`
- `gpt-5.5` high: `1580 / 1580` valid attempts, `1887` total call rows, accuracy `0.9727848101265822`, invalid call rate `0.16269210386857447`, cost `$10.74021`
- `gpt-5.5` medium: `1580 / 1580` valid attempts, `1678` total call rows, accuracy `0.9772151898734177`, invalid call rate `0.058402860548271755`, cost `$5.8694049999999995`
- `gpt-5.5` low: `1580 / 1580` valid attempts, `1585` total call rows, accuracy `0.9727848101265822`, invalid call rate `0.0031545741324921135`, cost `$3.2751799999999998`

## OpenAI Run Notes

OpenAI model access and request shape are validated. After the account moved to
Tier 5, the run was restarted with controlled in-process concurrency:

- Current script defaults: `OPENAI_MAX_WORKERS=24`, `TRUTHFULQA_OPENAI_MIN_REQUEST_INTERVAL_SECONDS=0.25`, and `TRUTHFULQA_OPENAI_MAX_OUTPUT_TOKENS=8192`
- Full controlled pass size: `4740` requests
- The detached `truthfulqa-openai` screen session exited after completion.
- Live log: `results/openai_full.log`
- OpenAI output cap was raised from `512` to `2048`, then to `8192`, because two high/medium reasoning attempts repeatedly consumed the smaller output budgets without emitting visible `A`/`B` answers.
- Two earlier empty high-reasoning rows were archived to `results/openai_full_invalid_archived.jsonl` and replaced through resume.
- Transient API connection errors and rate limits are retried with visible stderr logging that includes status, request id, and provider message.
- 429 retry pacing now runs before every OpenAI API attempt, including retry attempts.
- OpenAI provider delay parsing now honors minute-plus-second messages such as `7m12s`.
- A stale orphaned sequential OpenAI worker was stopped during the Tier 5 restart. One duplicate valid row from the overlap was archived to `results/openai_full_duplicate_archived.jsonl`; the live result file now has no duplicate valid attempts.
- Resume validation now rejects duplicate valid attempts while allowing invalid retry history, so invalid outputs remain auditable without making the run non-resumable.
- Final integrity check: `4740` valid attempt keys, `0` missing valid attempts, `0` extra attempt keys, `0` duplicate valid attempts, `410` invalid retry rows, `106` retry-history keys.
- Valid probe: `results/openai_probe_128.jsonl`
- Valid probe requests: `6 / 6`
- Valid probe estimated spend: `$0.0054`
- Valid probe projected full spend: about `$4.27`

The old `results/openai_pilot_invalid_16tok.jsonl` file is intentionally not usable for analysis. It was produced before increasing OpenAI `max_output_tokens` from `16` to `128`; high-reasoning calls consumed the output budget without visible `A`/`B` answers.

## Rerun Command

If the run needs to be resumed or repeated intentionally, run:

```bash
scripts/run_openai_full.sh
```

The script already uses `--resume`, so interrupted runs continue from
`results/openai_full.jsonl` after validating the expected condition and question
set metadata.
