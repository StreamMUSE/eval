#!/usr/bin/env python3
"""
从 JSON（如 results/*.json）读取 per-piece harmonicity（consonant/dissonant/unsupported ratios）
并按 model 分组绘制 box/violin。用法示例：
  python plot_harmonicity.py results/*.json -o harmonicity.png
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRIC_KEYS = {
    "consonant_ratio": "CR",
    "dissonant_ratio": "DR",
    "unsupported_ratio": "UR",
}
METRIC_ORDER = ["CR", "DR", "UR"]
METRIC_COLORS = {"CR": "#4C72B0", "DR": "#55A868", "UR": "#C44E52"}
BOX_ALPHA = 0.75

def _natural_key(s: str):
    parts = re.split(r"(\d+)", str(s))
    return tuple(int(p) if p.isdigit() else p.lower() for p in parts)


def load_harmonicity_from_json(path: Path, model_name: str | None = None) -> tuple[pd.DataFrame, Dict[str, dict]]:
    """从单个文件读取 per-piece 样本与 summary（兼容 summary.harmonicity 或 details[*].harmonicity）。"""
    def get_by_path(obj, path: str):
        cur = obj
        for part in path.split("."):
            if not isinstance(cur, dict):
                return None
            cur = cur.get(part)
            if cur is None:
                return None
        return cur

    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception as e:
        logging.warning("Failed to read JSON %s: %s", path, e)
        return pd.DataFrame(columns=["model", "metric", "value"]), {}

    model = model_name or path.stem
    rows: List[Dict[str, object]] = []
    summaries: Dict[str, dict] = {}

    # summary.harmonicity -> summaries
    if isinstance(obj, dict):
        summary = obj.get("summary") or {}
        harm = summary.get("harmonicity") if isinstance(summary, dict) else None
        if isinstance(harm, dict):
            for key, label in METRIC_KEYS.items():
                stat = harm.get(key)
                # stat may be scalar (mean) or dict (q1/median/q3/min/max)
                if isinstance(stat, dict):
                    try:
                        q1 = stat.get("q1", stat.get("q25"))
                        med = stat.get("median", stat.get("med"), stat.get("q2"))
                        q3 = stat.get("q3", stat.get("q75"))
                        mn = stat.get("min")
                        mx = stat.get("max")
                        cnt = int(stat.get("count", stat.get("n", 0)) or 0)
                        if None not in (q1, med, q3, mn, mx):
                            summaries[label] = {"q1": float(q1), "med": float(med), "q3": float(q3), "min": float(mn), "max": float(mx), "count": cnt}
                    except Exception:
                        continue
                else:
                    # scalar summary value -> store as med (fallback)
                    try:
                        if stat is not None:
                            summaries[label] = {"med": float(stat)}
                    except Exception:
                        continue

    # details -> per-piece harmonicity entries
    details = obj.get("details") if isinstance(obj, dict) else None
    if not details:
        details = obj.get("results") if isinstance(obj, dict) else None

    if isinstance(details, list):
        for entry in details:
            if not isinstance(entry, dict):
                continue
            # prefer entry.harmonicity dict
            h = entry.get("harmonicity") if isinstance(entry.get("harmonicity"), dict) else {}
            for key, label in METRIC_KEYS.items():
                if key in h and h[key] is not None:
                    try:
                        rows.append({"model": model, "metric": label, "value": float(h[key])})
                    except Exception:
                        continue
            # fallback: if top-level fields present (pitch_jsd-style) ignore here for harmonicity

    df = pd.DataFrame(rows, columns=["model", "metric", "value"]) if rows else pd.DataFrame(columns=["model", "metric", "value"])
    return df, summaries


def collect_many(paths: List[Path]) -> tuple[pd.DataFrame, Dict[str, Dict[str, dict]]]:
    dfs: List[pd.DataFrame] = []
    summaries_all: Dict[str, Dict[str, dict]] = {}
    for p in paths:
        df, stats = load_harmonicity_from_json(p, model_name=p.stem)
        if not df.empty:
            dfs.append(df)
        if stats:
            summaries_all[p.stem] = stats
    all_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=["model", "metric", "value"])
    return all_df, summaries_all


def plot_grouped_harmonicity(df: pd.DataFrame, summaries: Dict[str, Dict[str, dict]], out: Path, plot_type: str = "box") -> None:
    """绘图：优先使用原始样本；若无样本则用 summaries（bxp）。布局与 plot_jsd 一致。"""
    if (df.empty) and (not summaries):
        raise RuntimeError("既没有原始 samples，也没有 summary 可用来绘图。")

    if not df.empty:
        models = sorted(df["model"].unique(), key=_natural_key)
    else:
        models = sorted(summaries.keys(), key=_natural_key)

    n_models = len(models)
    n_metrics = len(METRIC_ORDER)

    box_width = 0.18
    inner_factor = 1.5
    group_padding = 0.45
    inner_spacing = box_width * inner_factor
    centers = np.arange(n_models) * (n_metrics * inner_spacing + group_padding)

    # 使用 samples 绘制
    if not df.empty:
        data_lists, positions, colors = [], [], []
        for i, model in enumerate(models):
            center = centers[i]
            for j, metric in enumerate(METRIC_ORDER):
                vals = df[(df["model"] == model) & (df["metric"] == metric)]["value"].dropna().astype(float).tolist()
                if not vals:
                    continue
                offset = (j - (n_metrics - 1) / 2) * inner_spacing
                positions.append(center + offset)
                data_lists.append(vals)
                colors.append(METRIC_COLORS.get(metric, "#777777"))

        if not data_lists:
            raise RuntimeError("没有找到任何数值条目用于绘图。")

        plt.style.use("seaborn-v0_8-whitegrid")
        fig_width = max(6, n_models * (n_metrics * box_width * inner_factor + 0.2))
        fig, ax = plt.subplots(figsize=(fig_width, 5))

        if plot_type == "box":
            bp = ax.boxplot(data_lists, positions=positions, widths=box_width, patch_artist=True, showmeans=True)
            for patch, col in zip(bp["boxes"], colors):
                patch.set_facecolor(col); patch.set_edgecolor(col); patch.set_alpha(BOX_ALPHA)
            for median in bp.get("medians", []):
                median.set_color("black")
            for mean in bp.get("means", []):
                mean.set_markerfacecolor("white"); mean.set_markeredgecolor("black")
        elif plot_type == "violin":
            vp = ax.violinplot(data_lists, positions=positions, widths=box_width, showmeans=False, showextrema=True)
            for body, col in zip(vp["bodies"], colors):
                body.set_facecolor(col); body.set_edgecolor(col); body.set_alpha(BOX_ALPHA)
            means = [float(np.mean(d)) for d in data_lists]; medians = [float(np.median(d)) for d in data_lists]
            ax.scatter(positions, means, marker="^", s=50, facecolors="white", edgecolors="black", zorder=3)
            ax.scatter(positions, medians, marker="s", s=30, facecolors="white", edgecolors="black", zorder=3)
        else:
            raise ValueError("plot_type must be 'box' or 'violin'")

        # 两行 x 标签与分组判断（interval / gen）
        tick_labels, group_labels = [], []
        for m in models:
            it = re.search(r"interval[_\-]?(\d+)", m, re.IGNORECASE)
            gi = re.search(r"gen(?:[_\-]?frame[_\-]?)?[_\-]?(\d+)", m, re.IGNORECASE)
            interval_label = f"interval{it.group(1)}" if it else "offline"
            gen_label = f"gen{gi.group(1)}" if gi else ""
            if interval_label == "offline":
                gen_label = ""
            label = m if (not interval_label and not gen_label) else f"{interval_label}\n{gen_label}"
            tick_labels.append(label); group_labels.append(interval_label)

        for i in range(len(centers) - 1):
            if group_labels[i] != group_labels[i + 1]:
                x = 0.5 * (centers[i] + centers[i + 1]); ax.axvline(x=x, color="#444444", linestyle="--", linewidth=0.8, alpha=0.7)

        ax.set_xticks(centers); ax.set_xticklabels(tick_labels, rotation=0, ha="center")
        ax.set_ylabel("harmonicity ratio"); ax.set_title("Harmonicity by model — " + " / ".join(METRIC_ORDER))

        metric_handles = [plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=METRIC_COLORS[m], markersize=8, alpha=BOX_ALPHA) for m in METRIC_ORDER]
        mean_handle = plt.Line2D([0], [0], marker="^", color="black", markersize=8, linestyle="None", markerfacecolor="white")
        median_handle = plt.Line2D([0], [0], marker="s", color="black", markersize=8, linestyle="None", markerfacecolor="white")

        fig.subplots_adjust(right=0.82); fig.tight_layout(rect=(0, 0, 0.78, 1.0))
        legend_x = 0.99; fig.subplots_adjust(right=0.92); fig.tight_layout(rect=(0, 0, 0.9, 1.0))
        fig.legend(metric_handles, METRIC_ORDER, title="Metric", loc="upper left", bbox_to_anchor=(legend_x, 0.98), bbox_transform=fig.transFigure)
        fig.legend([mean_handle, median_handle], ["Mean", "Median"], title="Summary", loc="center left", bbox_to_anchor=(legend_x, 0.60), bbox_transform=fig.transFigure)

        fig.tight_layout(); fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
        return

    # 若仅有 summaries，使用 bxp 绘制
    models = sorted(summaries.keys(), key=_natural_key)
    n_models = len(models); n_metrics = len(METRIC_ORDER)
    box_width = 0.18; inner_factor = 1.1; group_padding = 0.25
    inner_spacing = box_width * inner_factor
    centers = np.arange(n_models) * (n_metrics * inner_spacing + group_padding)

    stats_list, positions, colors = [], [], []
    for i, model in enumerate(models):
        center = centers[i]; mstats = summaries.get(model, {})
        for j, metric in enumerate(METRIC_ORDER):
            s = mstats.get(metric)
            if not s:
                continue
            med_val = s.get("med", s.get("median"))
            stat = {"med": float(med_val), "q1": float(s["q1"]), "q3": float(s["q3"]), "whislo": float(s["min"]), "whishi": float(s["max"]), "fliers": []}
            offset = (j - (n_metrics - 1) / 2) * inner_spacing; pos = center + offset
            stats_list.append(stat); positions.append(pos); colors.append(METRIC_COLORS.get(metric, "#777777"))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig_width = max(6, n_models * (n_metrics * box_width * inner_factor + 0.2))
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    bxp = ax.bxp(stats_list, positions=positions, widths=box_width, showmeans=False, patch_artist=True)
    for patch, col in zip(bxp["boxes"], colors):
        patch.set_facecolor(col); patch.set_edgecolor(col); patch.set_alpha(BOX_ALPHA)
    for median in bxp.get("medians", []):
        median.set_color("black")

    tick_labels, group_labels = [], []
    for m in models:
        it = re.search(r"interval[_\-]?(\d+)", m, re.IGNORECASE)
        gi = re.search(r"gen(?:[_\-]?frame[_\-]?)?[_\-]?(\d+)", m, re.IGNORECASE)
        interval_label = f"interval{it.group(1)}" if it else "offline"
        gen_label = f"gen{gi.group(1)}" if gi else ""
        if interval_label == "offline":
            gen_label = ""
        label = m if (not interval_label and not gen_label) else f"{interval_label}\n{gen_label}"
        tick_labels.append(label); group_labels.append(interval_label)

    for i in range(len(centers) - 1):
        if group_labels[i] != group_labels[i + 1]:
            x = 0.5 * (centers[i] + centers[i + 1]); ax.axvline(x=x, color="#444444", linestyle="--", linewidth=0.8, alpha=0.7)

    ax.set_xticks(centers); ax.set_xticklabels(tick_labels, rotation=0, ha="center")
    ax.set_ylabel("harmonicity ratio"); ax.set_title("Harmonicity by model — " + " / ".join(METRIC_ORDER))

    metric_handles = [plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=METRIC_COLORS[m], markersize=8, alpha=BOX_ALPHA) for m in METRIC_ORDER]
    ax.legend(metric_handles, METRIC_ORDER, title="Metric", loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout(); fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="从 JSON(s) 绘制分模型 harmonicity boxplot。")
    parser.add_argument("jsons", nargs="+", help="一个或多个 JSON 文件路径（可用通配符）。")
    parser.add_argument("-o", "--out", type=Path, default=Path("harmonicity_grouped_boxplot.png"))
    parser.add_argument("--plot-type", choices=("box", "violin"), default="box")
    args = parser.parse_args()

    paths = [Path(p) for p in args.jsons]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"找不到文件: {missing}")

    df, summaries = collect_many(paths)
    if df.empty and not summaries:
        raise SystemExit("未能从输入 JSON 中提取到任何 harmonicity 值。")

    plot_grouped_harmonicity(df, summaries, args.out, plot_type=args.plot_type)
    print(f"Saved harmonicity plot to {args.out.resolve()}")


if __name__ == "__main__":
    main()