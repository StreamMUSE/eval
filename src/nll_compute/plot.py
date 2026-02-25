"""Heatmap logic for NLL values."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Tuple, Any, Optional

import math
import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns

    _HAS_SEABORN = True
except Exception:
    _HAS_SEABORN = False


FILENAME_RE_INTERVAL = re.compile(r"interval[_-]?(\d+)")
FILENAME_RE_GENFRAME = re.compile(r"gen[_-]?frame[_-]?(\d+)")


def extract_interval_gen_frame(name: str) -> Optional[Tuple[int, int]]:
    """Return (interval, gen_frame) or None if not found."""
    m1 = FILENAME_RE_INTERVAL.search(name)
    m2 = FILENAME_RE_GENFRAME.search(name)
    if not m1 or not m2:
        return None
    try:
        return int(m1.group(1)), int(m2.group(1))
    except Exception:
        return None


def compute_scalar_from_nll_json(obj: Dict[str, Any], mode: str = "avg_mean") -> Optional[float]:
    """Compute scalar from a loaded .nll.json object."""
    if "summary" in obj and isinstance(obj["summary"], dict):
        summary = obj["summary"]
        if mode == "avg_mean":
            try:
                return summary["per_file_stats"]["avg_nll"]["mean"]
            except Exception:
                pass
        if mode == "weighted_total":
            return summary.get("weighted_avg_nll")
        if mode == "weighted_avg":
            return summary.get("weighted_avg_nll")

    vals = []
    weights = []
    total_nll_sum = 0.0
    total_tokens = 0
    for k, v in obj.items():
        if not isinstance(v, dict):
            continue
        if "error" in v:
            continue
        try:
            avg = float(v.get("avg_nll"))
            toks = int(v.get("total_tokens", 0))
            total = float(v.get("total_nll", avg * toks if toks else avg))
        except Exception:
            continue
        vals.append(avg)
        weights.append(toks)
        total_nll_sum += total
        total_tokens += toks

    if not vals:
        return None

    if mode == "avg_mean":
        return float(sum(vals) / len(vals))
    if mode == "weighted_avg":
        if sum(weights) == 0:
            return float(sum(vals) / len(vals))
        return float(sum(a * w for a, w in zip(vals, weights)) / sum(weights))
    if mode == "weighted_total":
        if total_tokens == 0:
            return float(sum(vals) / len(vals))
        return float(total_nll_sum / total_tokens)
    return None


def build_pivot_table(dirpath: Path, mode: str = "avg_mean"):
    records: Dict[Tuple[int, int], float] = {}
    files = sorted(dirpath.glob("*.nll.json"))
    for p in files:
        name = p.name
        if name in ("combined_nll.json", "summary_nll.json"):
            continue
        ex = extract_interval_gen_frame(name)
        if ex is None:
            ex = extract_interval_gen_frame(str(p.parent.name))
        if ex is None:
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        val = compute_scalar_from_nll_json(obj, mode=mode)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            continue
        records[ex] = val

    if not records:
        return None, None, None

    intervals = sorted({k[0] for k in records.keys()})
    gen_frames = sorted({k[1] for k in records.keys()})

    mat = np.full((len(intervals), len(gen_frames)), np.nan, dtype=float)
    idx_i = {v: i for i, v in enumerate(intervals)}
    idx_g = {v: i for i, v in enumerate(gen_frames)}
    for (itv, gf), val in records.items():
        mat[idx_i[itv], idx_g[gf]] = val

    return intervals, gen_frames, mat


def plot_heatmap(intervals, gen_frames, mat, outpath: Path, cmap: str = "viridis", annotate: bool = False):
    plt.figure(figsize=(max(6, len(gen_frames) * 0.5), max(4, len(intervals) * 0.5)))
    if _HAS_SEABORN:
        ax = sns.heatmap(
            mat,
            xticklabels=gen_frames,
            yticklabels=intervals,
            cmap=cmap,
            annot=annotate,
            fmt=".3f",
            cbar_kws={"label": "avg_nll"},
        )
    else:
        im = plt.imshow(mat, aspect="auto", cmap=cmap)
        plt.colorbar(im, label="avg_nll")
        plt.xticks(range(len(gen_frames)), gen_frames, rotation=45)
        plt.yticks(range(len(intervals)), intervals)
        if annotate:
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    v = mat[i, j]
                    if not math.isnan(v):
                        plt.text(j, i, f"{v:.3f}", ha="center", va="center", color="w", fontsize=7)

    plt.xlabel("gen_frame")
    plt.ylabel("interval")
    plt.title("NLL heatmap")
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()
