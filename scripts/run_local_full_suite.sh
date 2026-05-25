#!/usr/bin/env bash
# Run the local-provider full 1,580-attempt TruthfulQA benchmark across a
# sequence of LM Studio models, unloading each model between phases so only
# one model occupies memory at a time.
set -e
set -o pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

MODELS=(
  "google/gemma-4-e2b"
  "google/gemma-4-e4b"
  "openai/gpt-oss-20b"
  "google/gemma-3-27b"
)

# These models may already have partial JSONLs from earlier interrupted runs.
RESUME_MODELS=(
  "openai/gpt-oss-20b"
)

is_resume() {
  local needle=$1
  for m in "${RESUME_MODELS[@]}"; do
    [[ "$m" == "$needle" ]] && return 0
  done
  return 1
}

slug() {
  echo "$1" | tr '/' '_'
}

echo "=== Unloading any pre-existing models ==="
lms unload --all || true

total=${#MODELS[@]}
i=0
for model in "${MODELS[@]}"; do
  i=$((i + 1))
  slug_name=$(slug "$model")
  output="results/local_full_${slug_name}.jsonl"
  manifest="results/local_full_${slug_name}_manifest.json"
  resume_flag=""
  if is_resume "$model"; then
    resume_flag="--resume"
  fi
  echo "=== Full run ${i}/${total}: ${model} ${resume_flag} ==="
  date
  truthfulqa-bench experiment \
    --provider local \
    --models "$model" \
    --dataset data/TruthfulQA.csv \
    --output "$output" \
    --manifest "$manifest" \
    $resume_flag
  echo "=== Unloading ${model} ==="
  lms unload --all || true
done

echo "=== Concatenating ${total}-model results ==="
cat \
  results/local_full_google_gemma-4-e2b.jsonl \
  results/local_full_google_gemma-4-e4b.jsonl \
  results/local_full_openai_gpt-oss-20b.jsonl \
  results/local_full_google_gemma-3-27b.jsonl \
  > results/local_full_combined.jsonl

echo "=== Generating comparison ==="
python scripts/compare_local_to_frontier.py --results results/local_full_combined.jsonl > /dev/null
echo "Comparison written to results/local_vs_frontier.md and results/local_vs_frontier.json."

echo "=== Done ==="
date
wc -l results/local_full_*.jsonl
