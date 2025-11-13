# ...existing code...
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Any, Optional

import matplotlib.pyplot as plt
import numpy as np

def _try_float(s: Any) -> Optional[float]:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None

def read_csv_rows(p: Path):
    with p.open("r", encoding="utf-8", newline="") as fh:
        rdr = csv.DictReader(fh)
        fieldnames = rdr.fieldnames or []
        rows = list(rdr)
    return fieldnames, rows

def prepare_series(rows: List[dict], name_col: str, isr_col: str, nll_col: str, cr_col: str, isr_threshold: Optional[float]):
    labels: List[str] = []
    isr_vals: List[float] = []
    nll_vals: List[float] = []
    cr_vals: List[float] = []
    for r in rows:
        label = (r.get(name_col) or "").strip()
        if label == "":
            # try first value
            label = list(r.values())[0] if r else ""
        isr = _try_float(r.get(isr_col))
        if isr_threshold is not None:
            if isr is None or isr <= isr_threshold:
                continue
        nll = _try_float(r.get(nll_col))
        cr = _try_float(r.get(cr_col))
        labels.append(label)
        # convert missing -> np.nan so we can detect later
        isr_vals.append(np.nan if isr is None else isr)
        nll_vals.append(np.nan if nll is None else nll)
        cr_vals.append(np.nan if cr is None else cr)
    return labels, np.array(isr_vals, dtype=float), np.array(nll_vals, dtype=float), np.array(cr_vals, dtype=float)

def plot_three_bars(labels, isr, nll, cr, out_png: Path, show: bool = False):
    if len(labels) == 0:
        raise SystemExit("没有满足阈值的行，无法绘图。")

    x = np.arange(len(labels))
    width = 0.7

    fig, axs = plt.subplots(3, 1, figsize=(max(8, len(labels) * 0.5), 10), constrained_layout=True, sharex=True)

    # Helper to plot with missing values (NaN -> plot as 0 and hatch to indicate missing)
    def bar_with_missing(ax, data, color, ylabel):
        mask = np.isnan(data)
        data_plot = np.where(mask, 0.0, data)
        bars = ax.bar(x, data_plot, width=width, color=color)
        # mark missing bars
        for i, m in enumerate(mask):
            if m:
                bars[i].set_hatch("///")
                bars[i].set_edgecolor("gray")
                bars[i].set_facecolor("none")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)

    bar_with_missing(axs[0], isr, "#2b8cbe", "ISR_w")
    bar_with_missing(axs[1], nll, "#7bccc4", "weighted_ave_nll")
    bar_with_missing(axs[2], cr, "#f03b20", "consonant_ratio")

    axs[0].set_ylim(bottom=0)
    # x labels
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    axs[2].set_xlabel("row (first column)")

    fig.suptitle("ISR_w / weighted_ave_nll / consonant_ratio", fontsize=14)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    print(f"Saved plot -> {out_png}")
    if show:
        plt.show()

def main():
    p = argparse.ArgumentParser(description="从 CSV 绘制 ISR_w, weighted_ave_nll, consonant_ratio 三个子图，可按 ISR_w 阈值筛选并按 ISR_w 排序。")
    p.add_argument("csv", type=Path, help="输入 CSV 文件路径")
    p.add_argument("--isr-col", default="ISR_w", help="CSV 中 ISR_w 列名（默认 ISR_w）")
    p.add_argument("--nll-col", default="weighted_ave_nll", help="CSV 中 weighted_ave_nll 列名（默认 weighted_ave_nll）")
    p.add_argument("--cr-col", default="consonant_ratio", help="CSV 中 consonant_ratio 列名（默认 consonant_ratio）")
    p.add_argument("--name-col", default=None, help="CSV 中用于作为每行标签的列名（默认使用第一列）")
    p.add_argument("--threshold", type=float, default=None, help="只绘制 ISR_w 大于此阈值的行（默认不筛选）")
    p.add_argument("--sort", choices=["none","asc","desc"], default="none", help="是否按 ISR_w 排序：none(默认)/asc/desc")
    p.add_argument("-o", "--out", type=Path, default=Path("isr_nll_cr_plot.png"), help="输出 PNG 路径")
    p.add_argument("--show", action="store_true", help="绘制后弹出显示窗口")
    args = p.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"CSV 文件不存在: {args.csv}")

    fieldnames, rows = read_csv_rows(args.csv)
    if not fieldnames:
        raise SystemExit("CSV 文件没有表头。")

    # 归一化 header -> 实际 header 的映射，方便容错匹配（忽略空格、下划线、大小写）
    def _norm(s: str) -> str:
        return ''.join(c for c in s.lower().strip() if c.isalnum())

    norm_map = { _norm(fn): fn for fn in fieldnames }

    def find_header_candidate(desired: str) -> Optional[str]:
        if desired in fieldnames:
            return desired
        nd = _norm(desired)
        if nd in norm_map:
            return norm_map[nd]
        # 尝试部分匹配
        for k, v in norm_map.items():
            if nd in k or k in nd:
                return v
        return None

    # 选择 name_col（行标签），isr_col, nll_col, cr_col：支持模糊匹配 header
    name_col = args.name_col if args.name_col else fieldnames[0]
    mapped_name = find_header_candidate(name_col)
    if mapped_name is None:
        print(f"警告：指定的 name-col({name_col}) 未找到，使用第一列 {fieldnames[0]}")
        mapped_name = fieldnames[0]
    name_col = mapped_name

    isr_col_mapped = find_header_candidate(args.isr_col) or args.isr_col
    nll_col_mapped = find_header_candidate(args.nll_col) or args.nll_col
    cr_col_mapped = find_header_candidate(args.cr_col) or args.cr_col

    if isr_col_mapped not in fieldnames:
        print(f"警告：未在 CSV header 中找到 ISR 列（{args.isr_col}），绘图可能为空。可用 header: {fieldnames}")
    if nll_col_mapped not in fieldnames:
        print(f"警告：未在 CSV header 中找到 NLL 列（{args.nll_col}），绘图可能为空。可用 header: {fieldnames}")
    if cr_col_mapped not in fieldnames:
        print(f"警告：未在 CSV header 中找到 consonant_ratio 列（{args.cr_col}），绘图可能为空。可用 header: {fieldnames}")

    labels, isr, nll, cr = prepare_series(rows, name_col, isr_col_mapped, nll_col_mapped, cr_col_mapped, args.threshold)

    # 按 ISR_w 排序（可选），NaN 放到末尾
    if args.sort != "none" and len(isr) > 0:
        if args.sort == "desc":
            # NaN -> -inf, 以便排到末尾
            key = np.nan_to_num(isr, nan=-np.inf)
            order = np.argsort(-key)
        else:  # asc
            key = np.nan_to_num(isr, nan=np.inf)
            order = np.argsort(key)
        # apply ordering
        labels = [labels[i] for i in order]
        isr = isr[order]
        nll = nll[order]
        cr = cr[order]

    plot_three_bars(labels, isr, nll, cr, args.out, show=args.show)

if __name__ == "__main__":
    main()
# ...existing code...