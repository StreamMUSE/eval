# Accompaniment Evaluation Script

Utilities for benchmarking generated accompaniment MIDI against ground‑truth accompaniments. The CLI computes distributional divergences, latent distances (PolyDis), harmonic alignment, and phrase‑level continuity metrics, then presents the results in grouped summaries and machine‑readable JSON.

---

## Requirements

- Python 3.9+
- Python packages (see `requirements.txt`): `numpy`, `pretty_midi`, `mir_eval`, `torch`, `scipy`, `tqdm` *(optional)*
- For PolyDis latent metrics, clone the [`icm-deep-music-generation`](https://github.com/YatingMusic/icm-deep-music-generation) repo (or vendor the `poly_dis` package) and place the checkpoint at `poly_dis/model_param/polydis-v1.pt`.

Install dependencies:

```bash
pip install -r requirements.txt
```

CPU-only environments work, though PyTorch may emit CUDA warnings.

---

## Expected Inputs

- `--generated-dir`: MIDI files containing melody+accompaniment. Melody tracks can be filtered using name/program/index flags.
- `--groundtruth-dir`: MIDI files with ground-truth accompaniments (paired to generated files by filename stem).
- Optional validation phrases (`--validation-phrases`): JSONL with events for custom RD/VN checks.
- Optional automatic phrase analysis (`--auto-phrase-analysis`): segments each MIDI into fixed-length bar windows (default 4 bars) for RD/VN continuity.
- PolyDis metrics require `--polydis-root` pointing to the PolyDis codebase.

### PolyDis Setup

This repository does **not** ship the PolyDis sources or weights. To enable latent-distance metrics:

```bash
cd <workspace>
git clone https://github.com/YatingMusic/icm-deep-music-generation.git
```

After cloning, open `poly_dis/model_param/gdrive_link.txt` for the official download URL, retrieve `polydis-v1.pt`, and place it under `poly_dis/model_param/`, replacing the placeholder link file if present. When running the evaluator, point `--polydis-root` to the clone:

```bash
python scripts/evaluate_accompaniment_metrics.py \
  ... \
  --polydis-root /path/to/icm-deep-music-generation
```

### Directory Layout

```
<workspace>/
  generated/
    001.mid        # melody + accompaniment tracks
    002.mid
    ...
  groundtruth/
    001.mid        # accompaniment-only tracks
    002.mid
    ...
```

File stems must match between `generated/` and `groundtruth/` (e.g., `001.mid` in both). Inside each generated MIDI, label the melody track(s) clearly (e.g., instrument name `melody`) and the accompaniment track(s) as `accompaniment`. The script treats melody tracks as excluded from accompaniment analysis unless you pass `--keep-melody`.

---

## Usage Examples

### Basic comparison

```bash
python scripts/evaluate_accompaniment_metrics.py \
  --generated-dir path/to/generated_midis \
  --groundtruth-dir path/to/groundtruth_midis
```

### With PolyDis and automatic phrase analysis

```bash
python scripts/evaluate_accompaniment_metrics.py \
  --generated-dir tmp_eval/generated \
  --groundtruth-dir tmp_eval/groundtruth \
  --auto-phrase-analysis --phrase-bars 4 \
  --phrase-rhythm-resolution 0.01 \
  --polydis-root /path/to/icm-deep-music-generation \
  --output-json tmp_eval/results.json
```

### Using hand-crafted phrase windows

```bash
python scripts/evaluate_accompaniment_metrics.py \
  --generated-dir ... \
  --groundtruth-dir ... \
  --validation-phrases path/to/phrases.jsonl \
  --phrase-window-seconds 2.0 \
  --phrase-rhythm-resolution 0.01
```

Common melody filters:

- `--melody-track-names flute vocal`
- `--melody-programs 0 4`
- `--melody-track-indices 0`
- Add `--keep-melody` to retain melody tracks during analysis.

---

## Outputs

### Console Summary (Grouped)

```
Pairs evaluated: 12
Accompaniment vs Ground Truth:
  Pitch JSD: 0.0203
  Onset JSD: 0.0028
  Duration JSD: 0.0145
  PolyDis txt_mean: 6.6490
  PolyDis chd_mean: 4.7192
  PolyDis segments: 431

Accompaniment Inter-Track Continuity:
  Random: RD diff=1.2145 (±0.9542), VN diff=0.6921 (±0.5710), n=1000
  Same Song: RD diff=0.8473 (±0.8025), VN diff=0.5419 (±0.4882), n=8724
  Adjacent: RD diff=0.4620 (±0.5087), VN diff=0.3984 (±0.4026), n=398
  Auto phrase (gen vs gt): RD diff=0.3117 (±0.3528), VN diff=0.5289 (±0.2941), n=512

Accompaniment <-> Melody:
  Harmonic consonant ratio: 0.9185
  Harmonic dissonant ratio: 0.0584
  Harmonic unsupported ratio: 0.0231
  Chord accuracy: 0.7429
```

### JSON (`--output-json`)

Structured for downstream analysis:

```json
{
  "meta": {"pairs": 12},
  "summary": {
    "accompaniment_vs_groundtruth": { ... },
    "inter_track_continuity": { ... },
    "melody_relationship": { ... }
  },
  "details": [
    {
      "piece": "001",
      "pitch_jsd": 0.0187,
      "polydis": {"segments": 42, "txt_mean_distance": 5.21, ...},
      "auto_phrase_metrics": { ... },
      "harmonicity": { ... }
    },
    "..."
  ]
}
```

---

## Metrics at a Glance

| Group                       | Metrics                                                                         |
|-----------------------------|----------------------------------------------------------------------------------|
| Accompaniment vs Ground Truth | Pitch/Onset/Duration JS divergence; PolyDis latent distance (txt/chord).        |
| Inter-Track Continuity      | RD/VN differences for Random, Same Song, Adjacent phrase pairs + auto vs GT.    |
| Accompaniment ↔ Melody      | Harmonic consonant/dissonant/unsupported ratios; chord accuracy (if available). |

Lower JS divergence and PolyDis scores indicate closer matches. For RD/VN, expect Adjacent < Same Song < Random; large jumps flag discontinuities.

---

## Comparing Model Variants

1. Run the script for each configuration (e.g., dropout/no-dropout, prompt lengths, alternate melodies).
2. Collect the JSON summaries (`summary.accompaniment_vs_groundtruth`, etc.).
3. Load them into a notebook or DataFrame to build side-by-side tables or visualizations.

---

## Troubleshooting

- *PolyDis warning*: ensure `polydis-v1.pt` exists and `--polydis-root` points at the codebase.
- *Missing chords/harmonicity*: if either melody or accompaniment notes are absent, the harmonic metrics fall back to `n/a`.
- *pretty_midi errors*: some Linux distros need `libasound2-dev` or similar system libraries.
- *CUDA messages*: harmless on CPU-only machines; PyTorch falls back automatically.

---

## Repository Layout (Suggested)

```
scripts/
  evaluate_accompaniment_metrics.py
  README.md
requirements.txt
poly_dis/            # optional vendored dependency
tmp_eval/            # optional example data
```

Share this script alongside dependencies, PolyDis assets, and example data so teammates can reproduce the evaluation quickly.
