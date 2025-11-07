#!/usr/bin/env bash
#
# Run `evaluate_accompaniment_metrics.py` across multiple prompt folders and
# collect their textual summaries into a single CSV file. You can supply run
# directories directly on the command line or list them once in a configuration
# file (one `label:/path` pair per line).

set -euo pipefail

usage() {
  cat <<'EOF_USAGE' >&2
Usage: batch_evaluate_accompaniment_metrics.sh [config_file] [options] [label:/path ...] [-- extra evaluator args]

Options:
  --output PATH              Destination CSV path (default: music_quality_results_batch.csv)
  --generated-subdir NAME    Subdirectory containing generated MIDI (default: generated)
  --groundtruth-subdir NAME  Subdirectory containing ground-truth MIDI (default: gt)
  -h, --help                 Show this message

Configuration file format:
  Each non-empty, non-comment line must be `label:/absolute/or/relative/path`.
  The script looks for `promptXX` directories inside each provided path.
EOF_USAGE
}

CONFIG_FILE=""
OUTPUT_CSV="music_quality_results_batch.csv"
GENERATED_SUBDIR="generated"
GROUNDTRUTH_SUBDIR="gt"
DEFAULT_POLYDIS_ROOT="icm-deep-music-generation"

declare -a LABELS=()
declare -a PATHS=()
declare -a EXTRA_ARGS=()

if [[ $# -eq 0 ]]; then
  usage
  exit 1
fi

while [[ $# -gt 0 ]]; do
  arg=$1
  case "$arg" in
    -h|--help)
      usage
      exit 0
      ;;
    --output)
      if [[ $# -lt 2 ]]; then
        echo "Error: --output requires a value" >&2
        exit 1
      fi
      OUTPUT_CSV=$2
      shift 2
      ;;
    --generated-subdir)
      if [[ $# -lt 2 ]]; then
        echo "Error: --generated-subdir requires a value" >&2
        exit 1
      fi
      GENERATED_SUBDIR=$2
      shift 2
      ;;
    --groundtruth-subdir)
      if [[ $# -lt 2 ]]; then
        echo "Error: --groundtruth-subdir requires a value" >&2
        exit 1
      fi
      GROUNDTRUTH_SUBDIR=$2
      shift 2
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    *=*)
      label="${arg%%:*}"
      path="${arg#*:}"
      if [[ -z "$label" || -z "$path" ]]; then
        echo "Error: invalid label:path pair '$arg'" >&2
        exit 1
      fi
      if [[ ! -d "$path" ]]; then
        echo "Error: directory not found for label '$label': $path" >&2
        exit 1
      fi
      LABELS+=("$label")
      PATHS+=("$path")
      shift
      ;;
    *)
      if [[ -z "$CONFIG_FILE" && -f "$arg" ]]; then
        CONFIG_FILE=$arg
        shift
      else
        echo "Error: unrecognised argument '$arg'" >&2
        exit 1
      fi
      ;;
  esac
done

if [[ -n "$CONFIG_FILE" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" =~ ^# ]] && continue
    if [[ "$line" != *:* ]]; then
      echo "Warning: skipping malformed config line '$line'" >&2
      continue
    fi
    label="${line%%:*}"
    path="${line#*:}"
    label="${label## }"; label="${label%% }"
    path="${path## }"; path="${path%% }"
    if [[ -z "$label" || -z "$path" ]]; then
      echo "Warning: skipping malformed config line '$line'" >&2
      continue
    fi
    if [[ ! -d "$path" ]]; then
      echo "Warning: directory not found for label '$label': $path" >&2
      continue
    fi
    LABELS+=("$label")
    PATHS+=("$path")
  done <"$CONFIG_FILE"
fi

if [[ ${#LABELS[@]} -eq 0 ]]; then
  echo "Error: no label:path pairs provided (via config or CLI)." >&2
  exit 1
fi

tmp_dir=$(mktemp -d)
trap 'rm -rf "$tmp_dir"' EXIT

metadata_file="$tmp_dir/metadata.tsv"
labels_file="$tmp_dir/labels.txt"
printf "%s\n" "${LABELS[@]}" > "$labels_file"
> "$metadata_file"

for idx in "${!LABELS[@]}"; do
  label="${LABELS[$idx]}"
  base_path="${PATHS[$idx]}"

  shopt -s nullglob
  prompt_dirs=("$base_path"/prompt*)
  shopt -u nullglob

  if [[ ${#prompt_dirs[@]} -eq 0 ]]; then
    echo "Warning: no prompt directories found under $base_path" >&2
    continue
  fi

  for prompt_path in "${prompt_dirs[@]}"; do
    [[ -d "$prompt_path" ]] || continue
    prompt_name=$(basename "$prompt_path")
    if [[ ! $prompt_name =~ ^prompt([0-9]+)$ ]]; then
      echo "Warning: skipping directory (name does not match promptXX): $prompt_path" >&2
      continue
    fi
    prompt_len="${BASH_REMATCH[1]}"

    generated_dir="$prompt_path/$GENERATED_SUBDIR"
    groundtruth_dir="$prompt_path/$GROUNDTRUTH_SUBDIR"
    if [[ ! -d "$generated_dir" ]]; then
      echo "Warning: generated directory missing ($generated_dir); skipping" >&2
      continue
    fi
    if [[ ! -d "$groundtruth_dir" ]]; then
      echo "Warning: ground-truth directory missing ($groundtruth_dir); skipping" >&2
      continue
    fi

    echo "Evaluating $label @ prompt $prompt_len" >&2
    output_path="$tmp_dir/${prompt_len}__${label}.txt"

    cmd=(
      python scripts/evaluate_accompaniment_metrics.py
      --generated-dir "$generated_dir"
      --groundtruth-dir "$groundtruth_dir"
    )

    has_polydis=false
    has_frechet=false
    has_auto_phrase=false
    has_melody_names=false
    for arg in "${EXTRA_ARGS[@]}"; do
      if [[ "$arg" == "--polydis-root" ]]; then
        has_polydis=true
      elif [[ "$arg" == "--frechet-music-distance" ]]; then
        has_frechet=true
      elif [[ "$arg" == "--auto-phrase-analysis" ]]; then
        has_auto_phrase=true
      elif [[ "$arg" == "--melody-track-names" ]]; then
        has_melody_names=true
      fi
    done

    if ! $has_polydis; then
      cmd+=(--polydis-root "$DEFAULT_POLYDIS_ROOT")
    fi

    if ! $has_frechet; then
      cmd+=(--frechet-music-distance)
    fi

    if ! $has_auto_phrase; then
      cmd+=(--auto-phrase-analysis)
    fi

    if ! $has_melody_names; then
      cmd+=(--melody-track-names Guitar)
    fi

    cmd+=("${EXTRA_ARGS[@]}")

    "${cmd[@]}" >"$output_path"

    printf "%s\t%s\t%s\n" "$prompt_len" "$label" "$output_path" >> "$metadata_file"
  done
done

if [[ ! -s "$metadata_file" ]]; then
  echo "No evaluations completed; aborting." >&2
  exit 1
fi

python - "$metadata_file" "$labels_file" "$OUTPUT_CSV" <<'PY'
import csv
import sys
from pathlib import Path

metadata_path = Path(sys.argv[1])
labels_path = Path(sys.argv[2])
output_csv = Path(sys.argv[3])

labels = [line.strip() for line in labels_path.read_text().splitlines() if line.strip()]

records = {}
prompts = set()

with metadata_path.open() as handle:
    for line in handle:
        prompt, label, path = line.rstrip('\n').split('\t')
        prompts.add(prompt)
        text = Path(path).read_text().strip()
        records[(prompt, label)] = text

try:
    sorted_prompts = sorted(prompts, key=lambda v: int(v))
except ValueError:
    sorted_prompts = sorted(prompts)

output_csv.parent.mkdir(parents=True, exist_ok=True)
with output_csv.open('w', newline='') as handle:
    writer = csv.writer(handle)
    writer.writerow([''] + labels)
    for prompt in sorted_prompts:
        row = [prompt]
        for label in labels:
            row.append(records.get((prompt, label), ''))
        writer.writerow(row)
PY

echo "Saved aggregated results to $OUTPUT_CSV" >&2
