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

METRIC_KEYS = {
    "pitch_jsd": "Pitch",
    "onset_jsd": "Onset",
    "duration_jsd": "Duration",
}
METRIC_ORDER = ["Pitch", "Onset", "Duration"]
METRIC_COLORS = {"Pitch": "#4C72B0", "Onset": "#55A868", "Duration": "#DD8452"}
BOX_ALPHA = 0.75


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
        acc = summary.get("accompaniment_vs_groundtruth") if isinstance(summary, dict) else None
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


def plot_grouped_jsd(df: pd.DataFrame, out: Path) -> None:
    if df.empty:
        raise RuntimeError("没有可绘制的数据（请检查 JSON 是否包含 details 数组或 summary 聚合）。")

    models = sorted(df["model"].unique())
    n_models = len(models)
    n_metrics = len(METRIC_ORDER)

    box_width = 0.18                # 单个箱线宽度
    inner_factor = 1.5              # 组内箱线间距倍数（越小越靠近）
    group_padding = 0.25            # 不同模型组之间的额外间距

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

    bp = ax.boxplot(data_lists, positions=positions, widths=box_width, patch_artist=True, showmeans=True)

    # 着色
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_edgecolor(col)
        patch.set_alpha(BOX_ALPHA)
    for median in bp.get("medians", []):
        median.set_color("black")
    for mean in bp.get("means", []):
        mean.set_markerfacecolor("white")
        mean.set_markeredgecolor("black")

    ax.set_xticks(centers)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("Jensen-Shannon Divergence")
    ax.set_title("JSD by model — Pitch / Onset / Duration")

    # 图例（metric）
    handles = []
    labels = []
    for metric in METRIC_ORDER:
        handles.append(plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=METRIC_COLORS[metric], markersize=8, alpha=BOX_ALPHA))
        labels.append(metric)
    ax.legend(handles, labels, title="Metric", loc="upper right")

    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="从 JSON(s) 绘制分模型 JSD boxplot。每个 JSON 期望包含 details 数组（每条记录含 pitch_jsd/onset_jsd/duration_jsd）。")
    parser.add_argument("jsons", nargs="+", help="一个或多个 JSON 文件路径（可用通配符）。")
    parser.add_argument("-o", "--out", type=Path, default=Path("jsd_grouped_boxplot.png"))
    args = parser.parse_args()

    paths = [Path(p) for p in args.jsons]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"找不到文件: {missing}")

    df = collect_many(paths)
    if df.empty:
        raise SystemExit("未能从输入 JSON 中提取到 JSD 值。")

    plot_grouped_jsd(df, args.out)
    print(f"Saved JSD plot to {args.out.resolve()}")


if __name__ == "__main__":
    main()
# ...existing code...