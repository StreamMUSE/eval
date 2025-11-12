#!/usr/bin/env python3
"""
遍历指定目录，查找所有名为 `batch_run` 的文件夹；对每个 batch_run 下的子文件夹，
读取其中名为 `tick_history.json` 的文件并聚合保存/返回。
并为每个 batch_run 画图（两行子图）：
 - 上图：每个 tick 的 hit rate（bar）
 - 下图：每个 tick 的平均 backup_level（bar）
输出文件名基于 batch_run 的上三层，例如
  a/b/realtime/interval1_gen10/prompt10_gen100/batch_run -> realtime_interval1_gen10_prompt10_gen100.png
用法：
  python plot_system_performance.py /path/to/root -o out.json --plot-dir plots
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Iterable
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_batch_runs(root: Path) -> List[Path]:
    """递归查找所有名为 'batch_run' 的目录。"""
    root = root.expanduser().resolve()
    return [p for p in root.rglob("batch_run") if p.is_dir()]


def read_tick_history_file(p: Path) -> Any:
    """读取单个 tick_history.json，返回解析后的对象；出错返回 None（记录警告）。"""
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.warning("Failed to read tick_history.json at %s: %s", p, e)
        return None


def collect_from_batch_run(batch_dir: Path) -> List[Dict[str, Any]]:
    """
    遍历 batch_run 下的直接子目录，尝试读取每个子目录中的 tick_history.json。
    返回列表，每项包含 metadata 与解析后的内容。
    """
    res: List[Dict[str, Any]] = []
    for sub in sorted(batch_dir.iterdir()):
        if not sub.is_dir():
            continue
        th_path = sub / "tick_history.json"
        if not th_path.exists():
            continue
        obj = read_tick_history_file(th_path)
        if obj is None:
            continue
        res.append(
            {
                "batch_run_path": str(batch_dir),
                "batch_run_name": batch_dir.name,
                "subdir": str(sub),
                "subdir_name": sub.name,
                "tick_history_path": str(th_path),
                "tick_history": obj,
            }
        )
    return res


def collect_all(root: Path) -> List[Dict[str, Any]]:
    """在 root 下找到所有 batch_run 并收集它们的 tick_history 条目。"""
    all_entries: List[Dict[str, Any]] = []
    batches = find_batch_runs(root)
    for b in sorted(batches):
        entries = collect_from_batch_run(b)
        all_entries.extend(entries)
    return all_entries


def to_dataframe(entries: List[Dict[str, Any]]) -> pd.DataFrame:
    """将收集到的条目转为 DataFrame（tick_history 保留为 object 列）。"""
    if not entries:
        return pd.DataFrame(columns=["batch_run_path", "batch_run_name", "subdir", "subdir_name", "tick_history_path", "tick_history"])
    return pd.DataFrame(entries)


def save_aggregated(entries: List[Dict[str, Any]], out: Path, *, fmt: str = "json") -> None:
    out = out.expanduser().resolve()
    try:
        if fmt == "json":
            with out.open("w", encoding="utf-8") as f:
                json.dump(entries, f, ensure_ascii=False, indent=2)
        elif fmt == "csv":
            df = to_dataframe(entries).copy()
            # tick_history 转为字符串以便 CSV 存储（保证安全性）
            df["tick_history"] = df["tick_history"].apply(lambda x: json.dumps(x, ensure_ascii=False) if x is not None else "")
            df.to_csv(out, index=False)
        else:
            raise ValueError("未知输出格式: " + fmt)
    except Exception as e:
        logging.error("Failed to save aggregated data to %s: %s", out, e)
        raise


def _aggregate_tick_stats(histories: Iterable[List[dict]]):
    """
    histories: 可迭代的 tick_history 列表（每个是 list[dict]）
    返回两个 dict: hit_rate_by_tick{tick: rate}, avg_backup_by_tick{tick: avg}
    """
    hits = defaultdict(list)   # tick -> list of 0/1
    backups = defaultdict(list)  # tick -> list of backup_level (floats)
    for h in histories:
        if not isinstance(h, list):
            continue
        for entry in h:
            if not isinstance(entry, dict):
                continue
            tick = entry.get("tick")
            if tick is None:
                continue
            try:
                tick_i = int(tick)
            except Exception:
                continue
            is_hit = bool(entry.get("is_hit", False))
            backup = entry.get("backup_level")
            try:
                backup_val = float(backup) if backup is not None else np.nan
            except Exception:
                backup_val = np.nan
            hits[tick_i].append(1 if is_hit else 0)
            # 仅当 is_hit 为 True 时才将 backup_level 纳入 backups
            if is_hit and not np.isnan(backup_val):
                backups[tick_i].append(backup_val)

    if not hits and not backups:
        return {}, {}

    ticks = sorted(set(list(hits.keys()) + list(backups.keys())))
    hit_rate = {}
    avg_backup = {}
    for t in ticks:
        hlist = hits.get(t, [])
        hit_rate[t] = float(sum(hlist)) / float(len(hlist)) if hlist else float("nan")
        blist = backups.get(t, [])
        avg_backup[t] = float(np.mean(blist)) if blist else float("nan")
    return hit_rate, avg_backup


def _sanitize_name(s: str) -> str:
    """把路径替换为安全的文件名：用下划线代替非法字符，并限制长度。"""
    s = s.rstrip(os.sep)
    s = s.replace("..", "_")
    # 用正则替换所有非字母数字、下划线、点、连字符为下划线
    s = re.sub(r"[^0-9A-Za-z._-]+", "_", s)
    # 压缩连续下划线
    s = re.sub(r"_+", "_", s).strip("_")
    # 限制长度，保留尾部有意义部分
    if len(s) > 200:
        s = s[-200:]
    return s or "batch_run"


def plot_batch_run(entries: List[Dict[str, Any]], out_dir: Path) -> Path | None:
    """
    entries: 来自同一 batch_run 的条目列表
    为该 batch_run 生成一张图并保存到 out_dir，返回保存路径或 None（若无有效数据）。
    """
    if not entries:
        return None
    histories = [e["tick_history"] for e in entries if e.get("tick_history")]

    hit_rate, avg_backup = _aggregate_tick_stats(histories)
    if not hit_rate and not avg_backup:
        return None

    ticks = sorted(set(list(hit_rate.keys()) + list(avg_backup.keys())))
    x = np.array(ticks)

    # 构造图
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(6, len(x) * 0.12), 6), sharex=True)

    # 上图：hit rate bar
    y_hit = np.array([hit_rate.get(t, np.nan) for t in x])
    ax1.bar(x, y_hit, color="#4C72B0", alpha=0.8)
    ax1.set_ylim(0.0, 1.0)
    ax1.set_ylabel("Hit Rate")
    ax1.set_title("Per-tick Hit Rate")

    # 下图：平均 backup_level bar
    y_backup = np.array([avg_backup.get(t, np.nan) for t in x])
    ax2.bar(x, y_backup, color="#55A868", alpha=0.85)
    ax2.set_ylabel("Avg Backup Level")
    ax2.set_xlabel("Tick")
    ax2.set_title("Per-tick Avg Backup Level")

    # 美化 x ticks：若太多则只显示部分
    if len(x) > 40:
        step = max(1, len(x) // 20)
        ax2.set_xticks(x[::step])
    else:
        ax2.set_xticks(x)

    fig.tight_layout()

    # 生成文件名：只保留 batch_run 往上三层的路径片段作为文件名基底
    batch_run_path = entries[0]["batch_run_path"]
    parent = Path(batch_run_path).parent
    parts = [p for p in parent.parts if p not in (os.sep, "/")]
    last_parts = parts[-3:] if len(parts) >= 3 else parts
    if not last_parts:
        fname_base = _sanitize_name(parent.name)
    else:
        fname_base = _sanitize_name("_".join(last_parts))

    # 保存图像
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{fname_base}_performance.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 保存图片中使用的数据（CSV + JSON）
    try:
        df = pd.DataFrame(
            {
                "tick": x.tolist(),
                "hit_rate": [float(v) if not np.isnan(v) else None for v in y_hit.tolist()],
                "avg_backup": [float(v) if not np.isnan(v) else None for v in y_backup.tolist()],
            }
        )

        # 计算全局指标（遍历原始 histories，total_samples 包括所有记录，total_hits 只计 is_hit==True）
        total_samples = 0
        total_hits = 0
        backup_vals: List[float] = []
        for h in histories:
            if not isinstance(h, list):
                continue
            for rec in h:
                if not isinstance(rec, dict):
                    continue
                total_samples += 1
                if bool(rec.get("is_hit", False)):
                    total_hits += 1
                    b = rec.get("backup_level")
                    try:
                        if b is not None:
                            backup_vals.append(float(b))
                    except Exception:
                        continue

        global_hit_rate = float(total_hits) / float(total_samples) if total_samples > 0 else float("nan")
        global_avg_backup = float(np.mean(backup_vals)) if backup_vals else float("nan")

        # 为 CSV：在 per-tick 表末尾追加一行 summary （tick 字段用字符串 "GLOBAL"）
        summary_row = {
            "tick": "GLOBAL",
            "hit_rate": (float(global_hit_rate) if not np.isnan(global_hit_rate) else None),
            "avg_backup": (float(global_avg_backup) if not np.isnan(global_avg_backup) else None),
            "total_samples": int(total_samples),
            "total_hits": int(total_hits),
        }
        # 将 df 转为 object 类型以容纳混合类型列，然后追加 summary
        df_csv = df.copy().astype(object)
        # 添加 cols total_samples/total_hits 到 df_csv（行为空）
        df_csv["total_samples"] = None
        df_csv["total_hits"] = None
        df_csv = pd.concat([df_csv, pd.DataFrame([summary_row])], ignore_index=True)

        csv_path = out_dir / f"{fname_base}_performance.csv"
        df_csv.to_csv(csv_path, index=False)

        # 为 JSON：保存 records + global summary 字段
        json_path = out_dir / f"{fname_base}_performance.json"
        out_obj = {
            "records": df.to_dict(orient="records"),
            "global": {
                "global_hit_rate": None if np.isnan(global_hit_rate) else float(global_hit_rate),
                "global_avg_backup": None if np.isnan(global_avg_backup) else float(global_avg_backup),
                "total_samples": int(total_samples),
                "total_hits": int(total_hits),
            },
        }
        with json_path.open("w", encoding="utf-8") as jf:
            json.dump(out_obj, jf, ensure_ascii=False, indent=2)
            
    except Exception as e:
        logging.warning("Failed to save data file for %s: %s", out_path, e)

    return out_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="收集所有 batch_run 下的 tick_history.json，并按 batch_run 绘制性能图。")
    parser.add_argument("root", nargs="?", default=".", help="要扫描的根目录（默认当前目录）")
    parser.add_argument("-o", "--out", type=Path, default=Path("collected_tick_histories.json"), help="聚合输出文件（json/csv）")
    parser.add_argument("--format", choices=("json", "csv"), default="json")
    parser.add_argument("--plot-dir", type=Path, default=Path("plots"), help="每个 batch_run 输出图的目录")
    parser.add_argument("--no-save-agg", action="store_false", help="不保存聚合 JSON/CSV，只生成图片, 默认不保存")
    args = parser.parse_args()

    root = Path(args.root)
    logging.info("Scanning %s for batch_run directories...", root)
    entries = collect_all(root)
    if not entries:
        raise SystemExit("未找到任何 batch_run 下的 tick_history.json。")

    # 可选保存聚合数据
    if not args.no_save_agg:
        try:
            save_aggregated(entries, args.out, fmt=args.format)
            logging.info("Collected %d tick_history files -> %s", len(entries), args.out.resolve())
        except Exception:
            raise SystemExit("保存聚合文件失败，详见日志。")

    # 按 batch_run 分组绘图
    by_batch: Dict[str, List[Dict[str, Any]]] = {}
    for e in entries:
        key = e["batch_run_path"]
        by_batch.setdefault(key, []).append(e)

    out_paths = []
    for batch_path, group in sorted(by_batch.items()):
        saved = plot_batch_run(group, args.plot_dir)
        if saved:
            out_paths.append(str(saved))

    if out_paths:
        print("Saved plots:")
        for p in out_paths:
            print("  " + p)
    else:
        print("No plots were generated (no valid data).")


if __name__ == "__main__":
    main()