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
| `system_trace_v2.py` | Strict matched-system evaluator for frame deadlines and decision-availability spans |

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

## Matched System Metrics (schema v2)

This evaluator only accepts `system_trace.jsonl` schema v2. Each JSONL record
must declare `record_type` as either:

- `frame_deadline`: `tick`, `nominal_tick_time_s`, `deadline_time_s`
- `availability_span`: `start_tick`, `end_tick_exclusive`,
  `availability_time_s`

It deliberately rejects schema v1 and never treats MIDI notes or sparse
`note_on` events as evidence that a frame decision was delivered.

From the repository root:

```bash
PYTHONPATH=src python -m eval_toolkit.system_trace_v2 \
  --session-dir /path/to/session_a \
  --session-dir /path/to/session_b \
  --output-dir results/matched_system \
  --per-frame

# Or recursively discover sessions.
PYTHONPATH=src python -m eval_toolkit.system_trace_v2 \
  --root /path/to/experiment_root \
  --output-dir results/matched_system
```

The default observation tick is `session_config.prompt_length_ticks`; use
`--observation-tick` to override it. The window is
`[observation_tick, end_tick)`. Without `--end-tick`, the evaluator uses the
end of the continuous deadline sequence and rejects gaps.

For frame \(f\), let \(a_f\) be the earliest availability time among all spans
covering its tick and \(d_f\) its deadline. Then:

\[
\mathrm{ISR}_f = \frac{\#\{f: a_f \le d_f\}}{\#\{f\}}, \qquad
\mathrm{DeliveryRate} = \frac{\#\{f: a_f\ \mathrm{exists}\}}{\#\{f\}}.
\]

Valid REST decisions count as delivered frames because availability spans are
decision-level records. For delivered frames,
`staleness_ms = max(0, a_f - d_f) * 1000`; missing frames are reported
separately and are not inserted as infinity. `TTFP_ms` is the non-negative
difference between the earliest post-observation availability and the nominal
time of the observation tick; it is null when no frame is available.

Outputs are `per_session.csv`, `per_session.json`, `summary.json`, and optional
`per_frame.csv`. In `summary.json`, formal system comparisons must read the
`groups` object, which separates sessions by `(condition, continuation_mode)`.
Its stable keys have the form
`condition=<value>__continuation_mode=<value>`. The `overall` object mixes all
systems and is retained for audit only; it must not be used for paper tables.

All currently reported `mean`, `p50`, and `p95` values are descriptive
statistics. They are not confidence intervals. `summary.json` explicitly
records that 95% bootstrap confidence intervals are not implemented.

## Supported Metric Types

| Group | Types |
|---|---|
| `RESULT` | `pitch_jsd`, `onset_jsd`, `duration_jsd`, `consonant_ratio`, `unsupported_ratio`, `prompt_generated_txt_mean_distance`, `frechet_music_distance`, `chord_accuracy` |
| `NLL` | `nll` (avg per-file), `nll_weighted` (token-weighted) |
| `EXP_RAW` | `hit_rate`, `hit_rate_weighted`, `backup_level` |
