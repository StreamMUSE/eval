#!/usr/bin/env bash
set -e

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <prompt_dir> [additional evaluate_accompaniment_metrics.py args...]" >&2
  exit 1
fi

PROMPT_DIR=$1
shift

python scripts/evaluate_accompaniment_metrics.py \
  --generated-dir "${PROMPT_DIR}/generated" \
  --groundtruth-dir "${PROMPT_DIR}/gt" \
  --polydis-root icm-deep-music-generation \
  --melody-track-names Guitar \
  --frechet-music-distance \
  --auto-phrase-analysis \
  "$@"
