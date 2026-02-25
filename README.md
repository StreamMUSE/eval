# Evaluation Toolkit

Utilities in this directory evaluate melody-conditioned accompaniment runs and
summarise the results for plotting. All examples assume you are running from
the repository root (`StreamMUSE/`).

## Contents
- `evaluate_accompaniment_metrics.py` – score a single run by comparing
  generated MIDI files against ground-truth accompaniment.
- `batch_evaluate_accompaniment_metrics.sh` – loop the evaluator over many
  prompt folders and collate summaries into a CSV. It automatically discovers
  `promptXX` subdirectories and forwards evaluator flags (details below).
- `batch_runs.conf` – example configuration consumed by the batch script.
- `plot_music_quality.py` – produce the standard music-quality plots from the
  aggregated CSV output.
- `run_prompt_polydis_eval.sh` – helper for quickly scoring a single `promptXX`
  directory with PolyDis, FMD, auto-phrase analysis, and guitar-track removal
  enabled out of the box.

## The eval script
1. Gather two directories that share MIDI basenames:
   - `GEN_DIR`: generated outputs (melody + accompaniment). Melody tracks are
     removed automatically unless you pass `--keep-melody`.
   - `GT_DIR`: ground-truth accompaniment.
2. Launch the evaluator:
   ```bash
   python eval/evaluate_accompaniment_metrics.py \
       --generated-dir "$GEN_DIR" \
       --groundtruth-dir "$GT_DIR" \
       --output-json results/single_run_metrics.json \
       --frechet-music-distance \
       --polydis-root /path/to/icm-deep-music-generation
   ```

Key switches:
- `--frechet-music-distance` enables FMD (add `--frechet-cache-dir` to reuse
  embeddings). Omit the flag to save time.
- `--polydis-root` points at the PolyDis repo (clone
  `https://github.com/ZZWaang/icm-deep-music-generation.git` and use that path)
  when latent texture metrics are desired. Skip it if PolyDis is unnecessary or
  already cached.
- `--auto-phrase-analysis` segments the MIDI into fixed-bar windows for
  rhythm-density/voice-number checks when no validation phrases are supplied.

The script prints per-run statistics to stdout and, if `--output-json` is set,
writes detailed per-piece results for later inspection.

## Quick PolyDis Evaluation for a Single Prompt Folder
`run_prompt_polydis_eval.sh` wraps the evaluator for one prompt directory:

```bash
bash eval/run_prompt_polydis_eval.sh /exp/aria_eval/prompt75
```

The script expands to:
- `--generated-dir <prompt>/generated`
- `--groundtruth-dir <prompt>/gt`
- `--polydis-root icm-deep-music-generation` (override with your checkout)
- `--melody-track-names Guitar`
- `--frechet-music-distance`
- `--auto-phrase-analysis`

Any extra arguments (e.g. different cache paths, instrument filters) are passed
through to `evaluate_accompaniment_metrics.py`.

## Evaluating Many Prompt Folders
Use `batch_evaluate_accompaniment_metrics.sh` to iterate across prompt
directories (e.g. `prompt75`, `prompt150`, ...). The script expects each run to
contain `promptXX/<generated-subdir>` and `promptXX/<groundtruth-subdir>`.

### Using a Config File
Add lines of the form `label:/abs/path/to/run` to a text file (see
`batch_runs.conf`). Then run:
```bash
bash eval/batch_evaluate_accompaniment_metrics.sh eval/batch_runs.conf \
    --output results/music_quality_results_batch.csv \
    --generated-subdir generated \
    --groundtruth-subdir gt \
    -- --frechet-music-distance \
       --polydis-root /path/to/icm-deep-music-generation
```
Arguments after `--` are forwarded to `evaluate_accompaniment_metrics.py` for
every prompt. The script writes a CSV with one row per model/prompt pair and
stores the raw textual summaries alongside it.

**Defaults the batch script adds automatically (unless you pass your own):**
- `--polydis-root` pointing to `/mnt/weka/home/jianshu.she/xy/eval_/icm-deep-music-generation`.
- `--frechet-music-distance`.
- `--auto-phrase-analysis`.
- `--melody-track-names Guitar` (treat guitar-labeled tracks as melody and drop them).
Use the arguments after `--` to override any of these behaviours (e.g. specify a
different PolyDis checkout or disable FMD).

### Supplying Runs Inline
Alternatively, pass `label:/path` pairs on the command line:
```bash
bash eval/batch_evaluate_accompaniment_metrics.sh \
    aria:/exp/aria_eval dropout:/exp/dropout_eval \
    --output results/music_quality_results_batch.csv
```

## Plotting Music Quality Charts
Feed the aggregated CSV into `plot_music_quality.py` to recreate the JSD/FMD
dashboard:
```bash
python eval/plot_music_quality.py results/music_quality_results_batch.csv \
    --jsd-output images/music_quality_jsd_boxplots.png \
    --fmd-output images/music_quality_fmd.png \
    --polydis-output images/music_quality_polydis_texture.png \
    --harmonicity-output images/music_quality_harmonicity.png
```

The plotting script expects the CSV structure produced by the batch evaluator
(columns named by model labels and prompt lengths). Adjust the output paths as
needed; they will be created if absent.


## Tips
- Heavy metrics (PolyDis, FMD) are optional—skip them when iterating quickly,
  then rerun with caching enabled for final reports.
- Clone the PolyDis repository once:
  ```bash
  git clone https://github.com/ZZWaang/icm-deep-music-generation.git ~/poly_dis
  ```
  Then pass `--polydis-root ~/poly_dis` (or set that path in the batch script) whenever
  you require texture metrics.
- Keep `results/` and `images/` directories tidy by pointing `--output` and the
  plotting destinations to unique filenames per experiment.
- For ad-hoc experiments, run the single evaluator first, then manually append
  rows to a CSV before plotting.

---

## NLL Evaluation (NLL 负对数似然评测)

NLL (Negative Log-Likelihood) measures how well the model predicts the accompaniment given a melody. A **lower NLL is better**.

### ⚠️ Path Convention (路径约定) — IMPORTANT

`add_nll_to_summary.py` automatically discovers NLL results by looking for files named `experiments*.json` inside a `nll_runs/` subdirectory of your results folder. You **must** follow this naming convention when saving NLL output from StreamMUSE-1:

```
results-<experiment-name>/          ← your experiment results folder
├── <music quality JSON files>      ← produced by evaluate_accompaniment_metrics.py
├── final-sys-results/              ← produced by compute_final_system_metric.py
└── nll_runs/                       ← ⬅ create this manually
    ├── experiments_interval1_gen3.json    ← NLL output (name must start with "experiments")
    ├── experiments_interval2_gen5.json
    └── ...
```

The filename `experiments_<anything>.json` will be matched to CSV rows by parsing the `interval` and `gen_frame` numbers out of the filename automatically.

---

### Step 1 — Compute raw NLL (in StreamMUSE repo)

Run on the machine with the GPU and the trained model checkpoint. See [`src/nll_compute/README.md`](src/nll_compute/README.md) for the full workflow:

```bash
# Run from f:\repos\StreamMUSE-1
# Save output directly into the convention path inside eval's results folder
uv run python -m nll_compute.runners.run_cal_nll \
  --midi_dir /path/to/generated/interval_2_gen_frame_5/generated \
  --ckpt_path /path/to/model.ckpt \
  --save_json_path /path/to/eval/results-exp1/nll_runs/experiments_interval2_gen5.json \
  --window 384 --offset 128
```

*(Note: `nll_compute` is a top-level package in `StreamMUSE-1`, so no `src.` prefix needed there.)*

This writes a raw JSON like `{ "001.mid": { "avg_nll": 2.31, "total_nll": ..., "total_tokens": ... }, ... }`.

### Step 2 — Aggregate music quality + system metrics (in this repo)

```bash
# Run from f:\repos\eval
uv run summarize_metrics.py results-exp1/
```

This produces `results-exp1/<folder-name> music quality summary.csv` with all musical and system metrics.

### Step 3 — Merge NLL into the summary table

```bash
# Run from f:\repos\eval
uv run add_nll_to_summary.py results-exp1/ -o final_experiment_results.csv
```

This reads `results-exp1/nll_runs/experiments*.json`, matches each file to the correct CSV row by `interval`/`gen_frame` in the filename, and appends two new columns:
- `ave_nll` — unweighted mean of per-file `avg_nll`
- `weighted_ave_nll` — token-count-weighted mean (more reliable)

### Step 4 — Plot heatmap across parameter grid (optional)

If you swept across `interval` × `gen_frame` combinations:

```bash
uv run python -m src.nll_compute.runners.run_heatmap \
  --input-dir results-exp1/nll_runs \
  --out reports/heatmap_exp1.png \
  --csv-out reports/heatmap_exp1.csv \
  --value-mode weighted_avg --annotate
```

> **Where to look:** Merge logic → [`add_nll_to_summary.py`](add_nll_to_summary.py)  
> Aggregation logic → [`src/nll_compute/aggregate.py`](src/nll_compute/aggregate.py)  
> Heatmap logic → [`src/nll_compute/plot_nll_heatmap.py`](src/nll_compute/plot_nll_heatmap.py)  
> Shared stats/parsing tools → [`src/eval_toolkit/`](src/eval_toolkit/README.md)

