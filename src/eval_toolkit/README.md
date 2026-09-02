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

Within each group, paper tables must read `table_metrics`:

- `ttfp_p50_ms` and `ttfp_p95_ms` are percentiles of session-level TTFP.
- `isr_f` and `delivery_rate` are recomputed from all frame counts in the group.
- `staleness_p50_ms` and `staleness_p95_ms` are computed directly from all
  delivered-frame `staleness_ms` values in the group, not from per-session
  percentiles. Missing frames are excluded from staleness and reported in
  `missing_frames`.

The adjacent `metrics` object has `metrics_scope=per_session_descriptive` and
is retained for session-distribution auditing; its staleness summaries must not
be copied into the paper's `Stale_50/95` columns.

In legacy `--root` / `--session-dir` mode, all reported `mean`, `p50`, and
`p95` values are descriptive statistics. They are not confidence intervals.

### Formal matched manifest and confidence intervals

Formal comparison uses a CSV manifest with these required columns:

```text
piece_id,seed,system_id,session_dir,run_status,melody_input_sha256,failure_reason
```

The manifest must contain the complete Cartesian grid of 40 pieces, 3 seeds,
and every declared system. All rows for one `piece_id` must have the same
SHA-256 melody hash. `run_status` is `complete`, `failed`, or `missing`.
Complete rows require a session directory and are strictly evaluated as schema
v2. Failed and missing trials remain in `manifest_audit.csv` and require a
failure reason; either status, or an invalid complete session, blocks primary
confidence intervals rather than silently dropping the trial. In that case,
the CLI writes the audit and descriptive outputs, then returns a non-zero exit
status.

Manifest mode requires one fixed evaluation window for every trial:

```bash
PYTHONPATH=src python -m eval_toolkit.system_trace_v2 \
  --manifest /path/to/manifest.csv \
  --observation-tick 32 \
  --end-tick 128 \
  --bootstrap-replicates 10000 \
  --bootstrap-seed 0 \
  --output-dir results/matched_system \
  --per-frame
```

`--manifest` is mutually exclusive with `--root` and `--session-dir`.
Manifest-relative session paths are resolved relative to the manifest file.
In `summary.json`, formal table rows are keyed directly by `system_id` under
`groups`. Each group contains its `table_metrics` point estimate and
`bootstrap_ci` entries with the 2.5th/97.5th percentile interval and the number
of valid replicates. `paired_system_differences` reports matched differences
as `first system - second system`.

The bootstrap samples `piece_id` clusters with replacement. A sampled piece
retains all of its seeds, sessions, and frames, and every system uses the same
piece draw in a replicate. TTFP is recomputed from session-level values; ISR
and delivery rate are recomputed from pooled frame counts; staleness p50/p95
are recomputed directly from delivered frames. Missing frames are never
inserted as infinite staleness. These intervals are available only in valid
manifest mode; the legacy mode remains descriptive-only.

### Prepare matched post-join music-quality inputs

`prepare_matched_music_eval.py` converts realtime `combined.mid` exports and an
optional future offline manifest into exactly matched post-join MIDI pairs. It
uses the fixed 120 BPM window `[8, 32)` beats (`[4, 16)` seconds), clips notes
that cross either boundary, and shifts the prepared window to `[0, 12]`
seconds. The realtime manifest accepts the evaluator's `run_status` and
`melody_input_sha256` columns; the strict aliases `status` and `hash` are also
accepted, but supplying both names for one field is rejected. Hashes are file
SHA-256 values and are checked against `melody_midi_sha256`; canonical event
hashes are not interchangeable. Cohort Melody and GT file hashes are both
verified. The cohort Melody MIDI must contain exactly one non-empty named
`Melody` instrument, no other non-empty music instruments, and a non-empty
post-join window. Its cropped Melody events must exactly match the cropped
Melody events in the cohort GT.

```bash
PYTHONPATH=src python -m eval_toolkit.prepare_matched_music_eval \
  --cohort-manifest /path/to/cohort_manifest.json \
  --realtime-manifest /path/to/eval_manifest.csv \
  --offline-manifest /path/to/offline_trials.json \
  --output-dir results/matched_music
```

The default matched grid is 40 pieces by seeds `0,1,2`; smoke tests can use
`--expected-piece-count` and `--expected-seeds`. Failed, missing, malformed,
or unmatched trials remain in `audit.csv`, block publication of the paired
directories, and return a non-zero status. In a blocked batch all actual
published-target path fields are null/blank; no audit row points at a staging
file that was never published.

Each system has two explicit collections:

- `all_trials/` retains every complete trial, including legal REST-only or
  empty-accompaniment results. `prepared_manifest.json` reports
  `generated_acc_note_count`, `valid_output`, and the all-trial denominator for
  Valid Output Rate.
- `valid_only/` contains only basename-matched trials with at least one
  generated accompaniment note. Existing music metrics run on this explicitly
  conditional subset (`music_metrics_scope=conditional_on_valid_output`). Its
  `generated/` and `groundtruth/` directories are created even when the subset
  is empty, so a zero-valid-output system remains an explicit empty set.

Generated MIDI keeps Melody and Accompaniment content. Metric-ready GT is
accompaniment-only and its sole instrument is named `Piano`, matching the
legacy evaluator contract; the full cohort GT remains referenced in the audit.
For offline rows, `source_gt_midi` and `source_gt_sha256` identify the supplied
post-join GT file, while `cohort_full_gt_midi` and `cohort_full_gt_sha256`
identify the independently verified cohort source; the two hashes are never
interchanged.
Offline rows must declare already-postjoin generated/GT MIDI. Their timing is
validated but never shifted a second time, and their system scope is recorded
as `music_quality_only`; this tool never produces offline system metrics.

## Supported Metric Types

| Group | Types |
|---|---|
| `RESULT` | `pitch_jsd`, `onset_jsd`, `duration_jsd`, `consonant_ratio`, `unsupported_ratio`, `prompt_generated_txt_mean_distance`, `frechet_music_distance`, `chord_accuracy` |
| `NLL` | `nll` (avg per-file), `nll_weighted` (token-weighted) |
| `EXP_RAW` | `hit_rate`, `hit_rate_weighted`, `backup_level` |
