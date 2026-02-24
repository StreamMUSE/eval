# eval_toolkit — StreamMUSE Evaluation Toolkit

A suite of **modular, dependency-free** utilities for processing, analyzing, and exporting evaluation metrics from StreamMUSE experiments.
*(提供一套模块化、零第三方依赖的通用工具，用于处理、分析和导出 StreamMUSE 实验的评测指标。)*

## Structure (模块结构)

| Module | Description |
|---|---|
| `path_utils.py` | Metric type registry (`RESULT`, `NLL`, `EXP_RAW`) and path resolution helpers (`get_path`, `get_keys_from_dir`) |
| `json_parser.py` | Extract values from RESULT / NLL / EXP_RAW JSON files; normalizes them into flat lists ready for stats |
| `stats.py` | `compute_stats(values)` — returns count, mean, std, variance, min, max, p25/p50/p75, IQR. stdlib only, no numpy |
| `csv_exporter.py` | CLI tool: reads a result directory, aggregates all types, writes a clean CSV |

## Usage (使用方法)

### As a library (代码中调用)

```python
from eval_toolkit.path_utils import get_path
from eval_toolkit.json_parser import parse_by_type
from eval_toolkit.stats import compute_stats

key = "interval4_gen5_prompt_128_gen_576"

# 1. Resolve path for a metric type
p = get_path(key, "nll", base_dir="records")

# 2. Parse the JSON into a flat list of floats
items = parse_by_type(key, "nll", p)

# 3. Compute descriptive statistics
stats = compute_stats(items)
print(f"Weighted NLL mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
```

### As a CLI (命令行批量导出 CSV)

```bash
uv run python -m eval_toolkit.csv_exporter \
  --base_dir result/results-experiments2-local \
  --out reports/summary.csv \
  --types pitch_jsd,onset_jsd,nll,hit_rate \
  --stats mean,stdev_samp
```

Use `--dry-run` to preview headers without writing.

## Supported Metric Types

| Group | Types |
|---|---|
| `RESULT` | `pitch_jsd`, `onset_jsd`, `duration_jsd`, `consonant_ratio`, `unsupported_ratio`, `prompt_generated_txt_mean_distance`, `frechet_music_distance`, `chord_accuracy` |
| `NLL` | `nll` (avg per-file), `nll_weighted` (token-weighted) |
| `EXP_RAW` | `hit_rate`, `hit_rate_weighted`, `backup_level` |
