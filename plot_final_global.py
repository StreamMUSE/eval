#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        fv = float(v)
        if np.isnan(fv):
            return None
        return fv
    except Exception:
        return None

def extract_id(fname: str) -> str:
    # 优先去掉 prompt 之后的部分，再取前几段作为 id（兼容现有文件名格式）
    parts = fname.split("_")
    if "prompt" in parts:
        idx = parts.index("prompt")
        parts = parts[:idx]
    # 若有足够段，取前 6 段（与示例 baseline_interval_2_gen_frame_9 匹配）
    take = 6 if len(parts) >= 6 else len(parts)
    return "_".join(parts[:take])

def collect_metrics(json_path: Path) -> List[Dict[str, Optional[float]]]:
    j = json.loads(json_path.read_text(encoding="utf-8"))
    entries = j.get("collected", [])
    out = []
    for item in entries:
        fn = item.get("filename", "")
        g = item.get("global", {}) or {}
        out.append({
            "id": extract_id(fn),
            "filename": fn,
            "global_hit_rate": _safe_float(g.get("global_hit_rate")),
            "global_avg_backup": _safe_float(g.get("global_avg_backup")),
            "ISR_w": _safe_float(g.get("ISR_w")),
        })
    return out

def plot_metrics(entries: List[Dict[str, Optional[float]]], out_png: Path) -> None:
    ids = [e["id"] for e in entries]
    hit = [e["global_hit_rate"] if e["global_hit_rate"] is not None else 0.0 for e in entries]
    isrw = [e["ISR_w"] if e["ISR_w"] is not None else 0.0 for e in entries]
    avgb = [e["global_avg_backup"] if e["global_avg_backup"] is not None else 0.0 for e in entries]

    x = np.arange(len(ids))
    width = 0.6

    fig, axes = plt.subplots(3, 1, figsize=(max(8, len(ids)*0.6), 9), constrained_layout=True, sharex=True)

    axes[0].bar(x, hit, width=width, color="#2b8cbe")
    axes[0].set_ylabel("global_hit_rate")
    axes[0].set_ylim(0, 1.0)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, isrw, width=width, color="#7bccc4")
    axes[1].set_ylabel("ISR_w")
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].bar(x, avgb, width=width, color="#f03b20")
    axes[2].set_ylabel("global_avg_backup")
    axes[2].grid(axis="y", alpha=0.3)

    plt.xticks(x, ids, rotation=45, ha="right", fontsize=9)
    axes[2].set_xlabel("id (from filename)")

    fig.suptitle("Final system globals: hit rate / ISR_w / avg backup", fontsize=14)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    print(f"Saved plot -> {out_png}")

def main() -> None:
    p = argparse.ArgumentParser(description="从 results json 提取 global 指标并画图（3 行条形图）。")
    p.add_argument("input", type=Path, help="输入 results json（包含 collected 字段）")
    p.add_argument("-o", "--out", type=Path, default=Path("final_globals_plot.png"), help="输出 PNG 路径")
    args = p.parse_args()

    entries = collect_metrics(args.input)
    if not entries:
        raise SystemExit("No entries found in input json.")
    plot_metrics(entries, args.out)

if __name__ == "__main__":
    main()