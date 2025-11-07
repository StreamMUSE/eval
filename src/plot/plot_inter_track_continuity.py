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
    "rhythm_density": "RD",           # 如果你修改了 load_jsd_from_json 去读取这个嵌套路径
    "voice_number": "VN",
}
METRIC_ORDER = ["RD", "VN"]
METRIC_COLORS = {"RD": "#4C72B0", "VN": "#55A868"}
BOX_ALPHA = 0.75


# ...existing code...
def load_jsd_from_json(path: Path, model_name: str | None = None) -> tuple[pd.DataFrame, Dict[str, dict]]:
    """从 file 读取 samples 与 summary。
    - 优先尝试按 METRIC_KEYS 的点路径直接从 details 读取数值；
    - 如果未命中，则从 details[].auto_phrase_metrics 里的 generated / ground_truth 计算 RD/VN 差异（每个 piece 产一个样本）；
    - 同时提取 summary.inter_track_continuity 的五数（若存在）。"""
    def get_by_path(obj, path: str):
        cur = obj
        for part in path.split("."):
            if not isinstance(cur, dict):
                return None
            if part not in cur:
                return None
            cur = cur[part]
        return cur

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    model = model_name or path.stem
    rows: List[Dict[str, object]] = []
    summaries: Dict[str, dict] = {}

    # 1) summary.inter_track_continuity -> summaries（如果存在）
    if isinstance(obj, dict):
        summary = obj.get("summary") or {}
        inter = summary.get("inter_track_continuity") if isinstance(summary, dict) else None
        if isinstance(inter, dict):
            for key, label in METRIC_KEYS.items():
                stat = get_by_path(inter, key)
                if isinstance(stat, dict):
                    try:
                        q1 = stat.get("q1", stat.get("q25"))
                        med = stat.get("median", stat.get("med"), stat.get("q2"))
                        q3 = stat.get("q3", stat.get("q75"))
                        mn = stat.get("min")
                        mx = stat.get("max")
                        cnt = int(stat.get("count", stat.get("n", 0)) or 0)
                        if None not in (q1, med, q3, mn, mx):
                            summaries[label] = {"q1": float(q1), "med": float(med), "q3": float(q3),
                                                "min": float(mn), "max": float(mx), "count": cnt}
                    except Exception:
                        continue

    # 2) details -> 尝试按 METRIC_KEYS，若未命中则从 auto_phrase_metrics 计算 RD/VN 差异
    details = obj.get("details") if isinstance(obj, dict) else None
    if not details:
        details = obj.get("results") if isinstance(obj, dict) else None

    if isinstance(details, list):
        for entry in details:
            if not isinstance(entry, dict):
                continue

            found = False
            # 先按 METRIC_KEYS 的路径尝试取值（兼容之前的直接字段）
            for key, label in METRIC_KEYS.items():
                val = get_by_path(entry, key)
                if val is None:
                    continue
                try:
                    rows.append({"model": model, "metric": label, "value": float(val)})
                    found = True
                except Exception:
                    continue
            if found:
                continue

            # fallback: 从 auto_phrase_metrics 的 generated / ground_truth 计算 per-piece 差异样本
            ap = entry.get("auto_phrase_metrics") or {}
            gen_list = ap.get("generated") if isinstance(ap.get("generated"), list) else []
            gt_list = ap.get("ground_truth") if isinstance(ap.get("ground_truth"), list) else []
            # 计算按 index 对齐的 rhythm_density 与 voice_number 的绝对差，取每个 piece 的均值作为样本
            if gen_list and gt_list:
                rd_diffs = []
                vn_diffs = []
                n = min(len(gen_list), len(gt_list))
                for k in range(n):
                    g = gen_list[k]
                    h = gt_list[k]
                    try:
                        if isinstance(g, dict) and isinstance(h, dict):
                            if "rhythm_density" in g and "rhythm_density" in h:
                                rd_diffs.append(abs(float(g["rhythm_density"]) - float(h["rhythm_density"])))
                            if "voice_number" in g and "voice_number" in h:
                                vn_diffs.append(abs(float(g["voice_number"]) - float(h["voice_number"])))
                    except Exception:
                        continue
                if rd_diffs:
                    rows.append({"model": model, "metric": METRIC_KEYS.get("auto_phrase_pairs.rd", "RD"), "value": float(np.mean(rd_diffs))})
                if vn_diffs:
                    rows.append({"model": model, "metric": METRIC_KEYS.get("auto_phrase_pairs.vn", "VN"), "value": float(np.mean(vn_diffs))})

    return pd.DataFrame(rows), summaries
# ...existing code...

def collect_many(paths: List[Path]) -> tuple[pd.DataFrame, Dict[str, Dict[str, dict]]]:
    """收集多个文件，返回 (all_samples_df, summaries_by_model)
    summaries_by_model: { model_stem: { metric_label: summary_dict, ... }, ... }
    """
    dfs: List[pd.DataFrame] = []
    summaries_all: Dict[str, Dict[str, dict]] = {}
    for p in paths:
        df, stats = load_jsd_from_json(p, model_name=p.stem)
        if not df.empty:
            dfs.append(df)
        if stats:
            summaries_all[p.stem] = stats
    all_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=["model", "metric", "value"])
    return all_df, summaries_all


def plot_grouped_jsd(df: pd.DataFrame, summaries: Dict[str, Dict[str, dict]], out: Path) -> None:
    """优先用 df 的原始样本绘制 boxplot；若 df 为空则用 summaries 绘 bxp（五数）。"""
    if (df.empty) and (not summaries):
        raise RuntimeError("既没有原始 samples，也没有 summary 可用来绘图。")

    if not df.empty:
        models = sorted(df["model"].unique())
        n_models = len(models)
        n_metrics = len(METRIC_ORDER)

        box_width = 0.18
        inner_factor = 1.5
        group_padding = 0.25
        inner_spacing = box_width * inner_factor
        centers = np.arange(n_models) * (n_metrics * inner_spacing + group_padding)

        data_lists = []
        positions = []
        colors = []

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
                    continue
                offset = (j - (n_metrics - 1) / 2) * inner_spacing
                pos = center + offset
                data_lists.append(vals)
                positions.append(pos)
                colors.append(METRIC_COLORS.get(metric, "#777777"))

        plt.style.use("seaborn-v0_8-whitegrid")
        fig_width = max(6, n_models * (n_metrics * box_width * inner_factor + 0.2))
        fig, ax = plt.subplots(figsize=(fig_width, 5))

        bp = ax.boxplot(data_lists, positions=positions, widths=box_width, patch_artist=True, showmeans=True)
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col); patch.set_edgecolor(col); patch.set_alpha(BOX_ALPHA)
        for median in bp.get("medians", []):
            median.set_color("black")
        for mean in bp.get("means", []):
            mean.set_markerfacecolor("white"); mean.set_markeredgecolor("black")

        ax.set_xticks(centers); ax.set_xticklabels(models, rotation=30, ha="right")
        ax.set_ylabel("inter_track_continuity"); ax.set_title("inter_track_continuity by model — " + " / ".join(METRIC_ORDER))
        handles = [plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=METRIC_COLORS[m], markersize=8, alpha=BOX_ALPHA) for m in METRIC_ORDER]
        ax.legend(handles, METRIC_ORDER, title="Metric", loc="upper right")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        fig.tight_layout(); fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
        return

    # 如果没有原始样本，使用 summaries 绘制基于统计量的箱线（ax.bxp）
    models = sorted(summaries.keys())
    n_models = len(models)
    n_metrics = len(METRIC_ORDER)

    box_width = 0.18
    inner_factor = 1.1
    group_padding = 0.25
    inner_spacing = box_width * inner_factor
    centers = np.arange(n_models) * (n_metrics * inner_spacing + group_padding)

    stats_list = []
    positions = []
    colors = []

    for i, model in enumerate(models):
        center = centers[i]
        mstats = summaries.get(model, {})
        for j, metric in enumerate(METRIC_ORDER):
            s = mstats.get(metric)
            if not s:
                continue
            stat = {
                "med": float(s["med"] if "med" in s else s["med"] if "med" in s else s["med"]) if "med" in s else float(s["median"]),
                "q1": float(s["q1"]), "q3": float(s["q3"]),
                "whislo": float(s["min"]), "whishi": float(s["max"]),
                "fliers": []
            }
            offset = (j - (n_metrics - 1) / 2) * inner_spacing
            pos = center + offset
            stats_list.append(stat); positions.append(pos); colors.append(METRIC_COLORS.get(metric, "#777777"))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig_width = max(6, n_models * (n_metrics * box_width * inner_factor + 0.2))
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    bxp = ax.bxp(stats_list, positions=positions, widths=box_width, showmeans=False, patch_artist=True)
    for patch, col in zip(bxp["boxes"], colors):
        patch.set_facecolor(col); patch.set_edgecolor(col); patch.set_alpha(BOX_ALPHA)
    for median in bxp.get("medians", []):
        median.set_color("black")

    ax.set_xticks(centers); ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("inter_track_continuity"); ax.set_title("inter_track_continuity by model — " + " / ".join(METRIC_ORDER))
    handles = [plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=METRIC_COLORS[m], markersize=8, alpha=BOX_ALPHA) for m in METRIC_ORDER]
    ax.legend(handles, METRIC_ORDER, title="Metric", loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="从 JSON(s) 绘制分模型 inter_track_continuity boxplot。")
    parser.add_argument("jsons", nargs="+", help="一个或多个 JSON 文件路径（可用通配符）。")
    parser.add_argument("-o", "--out", type=Path, default=Path("itc_grouped_boxplot.png"))
    args = parser.parse_args()

    paths = [Path(p) for p in args.jsons]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"找不到文件: {missing}")

    df, summaries = collect_many(paths)
    if df.empty and not summaries:
        raise SystemExit("未能从输入 JSON 中提取到任何 inter_track_continuity 值。")

    plot_grouped_jsd(df, summaries, args.out)
    print(f"Saved inter-track continuity plot to {args.out.resolve()}")

if __name__ == "__main__":
    main()