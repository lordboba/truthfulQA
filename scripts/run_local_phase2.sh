#!/usr/bin/env bash
# Phase 2 of the local benchmark suite:
#   1. Finish gpt-oss-20b (resume).
#   2. Run gemma-3-27b fresh.
#   3. Emit a mid-suite local_vs_frontier.md checkpoint (gemma-4-e2b done,
#      gemma-4-e4b partial, gpt-oss-20b done, gemma-3-27b done).
#   4. Resume gemma-4-e4b to completion.
#   5. Emit the final local_vs_frontier.md update.
#
# gemma-4-31b is intentionally skipped.
set -e
set -o pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

DATA=data/TruthfulQA.csv
RESULTS=results

generate_combined() {
  local out="$RESULTS/local_full_combined.jsonl"
  : > "$out"
  for jsonl in \
    "$RESULTS/local_full_google_gemma-4-e2b.jsonl" \
    "$RESULTS/local_full_google_gemma-4-e4b.jsonl" \
    "$RESULTS/local_full_openai_gpt-oss-20b.jsonl" \
    "$RESULTS/local_full_google_gemma-3-27b.jsonl"; do
    if [[ -s "$jsonl" ]]; then
      cat "$jsonl" >> "$out"
    fi
  done
}

update_markdown() {
  local label=$1
  generate_combined
  python scripts/compare_local_to_frontier.py --results "$RESULTS/local_full_combined.jsonl" > /dev/null
  echo "=== Markdown checkpoint (${label}) written to $RESULTS/local_vs_frontier.md ==="
}

echo "=== Unloading any pre-existing models ==="
lms unload --all || true

echo "=== Phase 2/run a: openai/gpt-oss-20b (resume) ==="
date
truthfulqa-bench experiment \
  --provider local \
  --models openai/gpt-oss-20b \
  --dataset "$DATA" \
  --output "$RESULTS/local_full_openai_gpt-oss-20b.jsonl" \
  --manifest "$RESULTS/local_full_openai_gpt-oss-20b_manifest.json" \
  --resume

echo "=== Unloading openai/gpt-oss-20b ==="
lms unload --all || true

echo "=== Phase 2/run b: google/gemma-3-27b (fresh) ==="
date
truthfulqa-bench experiment \
  --provider local \
  --models google/gemma-3-27b \
  --dataset "$DATA" \
  --output "$RESULTS/local_full_google_gemma-3-27b.jsonl" \
  --manifest "$RESULTS/local_full_google_gemma-3-27b_manifest.json"

echo "=== Unloading google/gemma-3-27b ==="
lms unload --all || true

update_markdown "mid"

echo "=== Phase 2/run c: google/gemma-4-e4b (resume to completion) ==="
date
truthfulqa-bench experiment \
  --provider local \
  --models google/gemma-4-e4b \
  --dataset "$DATA" \
  --output "$RESULTS/local_full_google_gemma-4-e4b.jsonl" \
  --manifest "$RESULTS/local_full_google_gemma-4-e4b_manifest.json" \
  --resume

echo "=== Unloading google/gemma-4-e4b ==="
lms unload --all || true

update_markdown "final"

echo "=== Done ==="
date
wc -l "$RESULTS"/local_full_*.jsonl
