#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f .env ]]; then
  echo ".env is required with OPENAI_API_KEY before running paid OpenAI calls." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

export TRUTHFULQA_OPENAI_MIN_REQUEST_INTERVAL_SECONDS="${TRUTHFULQA_OPENAI_MIN_REQUEST_INTERVAL_SECONDS:-0.25}"
export TRUTHFULQA_OPENAI_MAX_OUTPUT_TOKENS="${TRUTHFULQA_OPENAI_MAX_OUTPUT_TOKENS:-8192}"
OPENAI_MAX_WORKERS="${OPENAI_MAX_WORKERS:-24}"

truthfulqa-bench experiment \
  --provider openai \
  --output results/openai_full.jsonl \
  --manifest results/openai_manifest.json \
  --openai-budget 100 \
  --max-workers "$OPENAI_MAX_WORKERS" \
  --resume
