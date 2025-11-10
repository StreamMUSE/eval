# ...existing code...
#!/usr/bin/env python3
"""
从你提供的 JSON（像 attentiondropout_128_wo_prompt.json）读取 per-piece JSD 值并绘制按 model 分组的 boxplot。
用法示例：
  python plot_jsd.py results/attentiondropout_128_wo_prompt.json -o jsd.png
  python plot_jsd.py results/*.json -o jsd_all_models.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

METRIC_KEYS = {
    "pitch_jsd": "Pitch",
    "onset_jsd": "Onset",
    "duration_jsd": "Duration",
}
METRIC_ORDER = ["Pitch", "Onset", "Duration"]
METRIC_COLORS = {"Pitch": "#4C72B0", "Onset": "#55A868", "Duration": "#DD8452"}
BOX_ALPHA = 0.75

# Natural sort key: splits on runs of digits and converts digit runs to int
def _natural_key(s: str):
    parts = re.split(r'(\d+)', str(s))
    return tuple(int(p) if p.isdigit() else p.lower() for p in parts)

def load_jsd_from_json(path: Path, model_name: str | None = None) -> pd.DataFrame:
    """从单个 JSON 文件提取每-piece 的 pitch/onset/duration JSD，返回 DataFrame rows: model, metric, value."""
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    model = model_name or path.stem

    rows: List[Dict[str, object]] = []

    # 尝试从 details 数组读取每-piece 值（这是你给出的文件格式）
    details = None
    if isinstance(obj, dict):
        details = obj.get("details") or obj.get("results") or obj.get("items")
    if isinstance(details, list):
        for entry in details:
            if not isinstance(entry, dict):
                continue
            for key, label in METRIC_KEYS.items():
                if key in entry and entry[key] is not None:
                    try:
                        val = float(entry[key])
                        rows.append({"model": model, "metric": label, "value": val})
                    except Exception:
                        continue

    # 如果没有 details，尝试从 summary -> accompaniment_vs_groundtruth 提取统计信息（无法做 boxplot，只保留 median/count）
    if not rows and isinstance(obj, dict):
        summary = obj.get("summary") or {}
        acc = (
            summary.get("accompaniment_vs_groundtruth")
            if isinstance(summary, dict)
            else None
        )
        if isinstance(acc, dict):
            # 只能得到单个统计值（median），以 count 个重复值模拟分布（fallback，仅用于可视化占位）
            for k, label in METRIC_KEYS.items():
                stat = acc.get(k)
                if isinstance(stat, dict) and "median" in stat and "count" in stat:
                    try:
                        val = float(stat["median"])
                        cnt = int(stat.get("count", 1))
                        for _ in range(cnt):
                            rows.append({"model": model, "metric": label, "value": val})
                    except Exception:
                        continue

    return pd.DataFrame(rows)


def collect_many(paths: List[Path]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for p in paths:
        df = load_jsd_from_json(p, model_name=p.stem)
        if not df.empty:
            rows.append(df)
    if not rows:
        return pd.DataFrame(columns=["model", "metric", "value"])
    return pd.concat(rows, ignore_index=True)


def plot_grouped_jsd(df: pd.DataFrame, out: Path, plot_type: str = "box") -> None:
    if df.empty:
        raise RuntimeError(
            "没有可绘制的数据（请检查 JSON 是否包含 details 数组或 summary 聚合）。"
        )

    models = sorted(df["model"].unique(), key=_natural_key)
    n_models = len(models)
    n_metrics = len(METRIC_ORDER)

    box_width = 0.18  # 单个箱线宽度
    inner_factor = 1.5  # 组内箱线间距倍数（越小越靠近）
    group_padding = 0.45  # 不同模型组之间的额外间距

    inner_spacing = box_width * inner_factor
    centers = np.arange(n_models) * (n_metrics * inner_spacing + group_padding)

    data_lists = []
    positions = []
    colors = []

    # 按 model 顺序与 METRIC_ORDER 填充
    for i, model in enumerate(models):
        center = centers[i]
        for j, metric in enumerate(METRIC_ORDER):
            vals = (
                df[(df["model"] == model) & (df["metric"] == metric)]["value"]
                .dropna()
                .astype(float)
                .tolist()
            )
            if len(vals) == 0:
                # 如果该 model 缺少该 metric，跳过
                continue
            # 使三个箱线紧密排列在中心附近
            offset = (j - (n_metrics - 1) / 2) * inner_spacing
            pos = center + offset
            data_lists.append(vals)
            positions.append(pos)
            colors.append(METRIC_COLORS.get(metric, "#777777"))

    if not data_lists:
        raise RuntimeError("没有找到任何数值 JSD 条目。")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig_width = max(6, n_models * (n_metrics * box_width * inner_factor + 0.2))
    fig, ax = plt.subplots(figsize=(fig_width, 5))

    if plot_type == "box":
        bp = ax.boxplot(
            data_lists,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            showmeans=True,
        )

        # 着色 boxplot
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col)
            patch.set_edgecolor(col)
            patch.set_alpha(BOX_ALPHA)
        for median in bp.get("medians", []):
            median.set_color("black")
        for mean in bp.get("means", []):
            mean.set_markerfacecolor("white")
            mean.set_markeredgecolor("black")

    elif plot_type == "violin":
        # 使用 matplotlib 的 violinplot，然后对每个 body 着色，并画出均值/中位数点
        vp = ax.violinplot(
            data_lists,
            positions=positions,
            widths=box_width,
            showmeans=False,
            showextrema=True,
        )
        # bodies 顺序与 data_lists 一致
        for body, col in zip(vp["bodies"], colors):
            body.set_facecolor(col)
            body.set_edgecolor(col)
            body.set_alpha(BOX_ALPHA)

        # 绘制均值与中位数标记（白底三角表示均值，黑边方块表示中位数）
        means = [float(np.mean(d)) for d in data_lists]
        medians = [float(np.median(d)) for d in data_lists]
        ax.scatter(
            positions,
            means,
            marker="^",
            s=50,
            facecolors="white",
            edgecolors="black",
            zorder=3,
        )
        ax.scatter(
            positions,
            medians,
            marker="s",
            s=30,
            facecolors="white",
            edgecolors="black",
            zorder=3,
        )

    else:
        raise ValueError(f"未知的 plot_type: {plot_type!r}，可选 'box' 或 'violin'。")

    # 构造两行标签：第一行 interval，第二行 gen
    tick_labels: List[str] = []
    group_labels: List[str] = []
    for m in models:
        it = re.search(r"interval[_\-]?(\d+)", m, re.IGNORECASE)
        gi = re.search(r"gen(?:[_\-]?frame[_\-]?)?[_\-]?(\d+)", m, re.IGNORECASE)
        interval_label = f"interval{it.group(1)}" if it else "offline"
        gen_label = f"gen{gi.group(1)}" if gi else ""
        if interval_label == "offline":
            gen_label = ""
        if not interval_label and not gen_label:
            label = m
        else:
            label = f"{interval_label}\n{gen_label}"
        tick_labels.append(label)
        group_labels.append(interval_label)  # 保存用于分组判定

    # 在不同 interval 组之间画竖线分隔（使用 model centers 的中点）
    for i in range(len(centers) - 1):
        if group_labels[i] != group_labels[i + 1]:
            x = 0.5 * (centers[i] + centers[i + 1])
            ax.axvline(x=x, color="#444444", linestyle="--", linewidth=0.8, alpha=0.7)
            # 若想更明显可改 linewidth 或 color

    ax.set_xticks(centers)
    # 两行标签通常不需要旋转，居中显示
    ax.set_xticklabels(tick_labels, rotation=0, ha="center")
    ax.set_ylabel("Jensen-Shannon Divergence")
    ax.set_title("JSD by model — Pitch / Onset / Duration")

    metric_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=METRIC_COLORS[metric],
            markersize=8,
            alpha=BOX_ALPHA,
        )
        for metric in METRIC_ORDER
    ]
    metric_labels = METRIC_ORDER

    mean_handle = plt.Line2D(
        [0],
        [0],
        marker="^",
        color="black",
        markersize=8,
        linestyle="None",
        markerfacecolor="white",
    )
    median_handle = plt.Line2D(
        [0],
        [0],
        marker="s",
        color="black",
        markersize=8,
        linestyle="None",
        markerfacecolor="white",
    )

    # 预留右侧空间，使用 tight_layout 的 rect 参数避免图例被覆盖或裁剪
    fig.subplots_adjust(right=0.82)
    fig.tight_layout(rect=(0, 0, 0.78, 1.0))

    legend_x = 0.99  # 可调（例如 0.95 -> 更远；0.88 -> 更近）
    fig.subplots_adjust(right=0.92)  # 预留空间，不要太大
    fig.tight_layout(rect=(0, 0, 0.9, 1.0))

    # 在 figure 级别放置图例（使用 figure 坐标），确保在保存时可见
    metric_legend = fig.legend(
        metric_handles,
        metric_labels,
        title="Metric",
        loc="upper left",
        bbox_to_anchor=(legend_x, 0.98),
        bbox_transform=fig.transFigure,
        borderaxespad=0.0,
    )
    summary_legend = fig.legend(
        [mean_handle, median_handle],
        ["Mean", "Median"],
        title="Summary",
        loc="center left",
        bbox_to_anchor=(legend_x, 0.60),
        bbox_transform=fig.transFigure,
        borderaxespad=0.0,
    )

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="从 JSON(s) 绘制分模型 JSD boxplot。每个 JSON 期望包含 details 数组（每条记录含 pitch_jsd/onset_jsd/duration_jsd）。"
    )
    parser.add_argument(
        "jsons", nargs="+", help="一个或多个 JSON 文件路径（可用通配符）。"
    )
    parser.add_argument(
        "-o", "--out", type=Path, default=Path("jsd_grouped_boxplot.png")
    )
    parser.add_argument(
        "--plot-type",
        choices=("box", "violin"),
        default="box",
        help="绘图类型：box（默认）或 violin。",
    )
    args = parser.parse_args()

    paths = [Path(p) for p in args.jsons]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"找不到文件: {missing}")

    df = collect_many(paths)
    if df.empty:
        raise SystemExit("未能从输入 JSON 中提取到 JSD 值。")

    plot_grouped_jsd(df, args.out, plot_type=args.plot_type)
    print(f"Saved JSD plot to {args.out.resolve()}")


if __name__ == "__main__":
    main()
# ...existing code...
