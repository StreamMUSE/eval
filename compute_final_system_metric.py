# ...existing code...
#!/usr/bin/env python3
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

def _aggregate_tick_stats(histories: Iterable[List[dict]], c : float = 32.0):
    """
    histories: 可迭代的 tick_history 列表（每个是 list[dict]）
    返回三个 dict:
      - hit_rate_by_tick {tick: rate}
      - avg_backup_by_tick {tick: avg}
      - weighted_hit_by_tick {tick: mean weighted contribution}
    加权贡献 per-response = (C - clamp(backup, 0, C)) / C * I
      - I = 1 for hit, 0 for miss
    对每个 tick，weighted_hit_by_tick 为该 tick 所有 responses 加权贡献的平均值（若无 response 则为 0.0）。
    """
    hits = defaultdict(list)   # tick -> list of 0/1
    backups = defaultdict(list)  # tick -> list of backup_level (floats)
    weighted_hits = defaultdict(list)  # tick -> list of (is_hit * weight)
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
            if not is_hit:
                weighted_hits[tick_i].append(0.0)
            else:
                I = 1.0 if is_hit else 0.0
                backup_val = float(backup) if backup is not None else c
                weighted_hit_t = (c - backup_val)/c * I
                weighted_hits[tick_i].append(weighted_hit_t)

    if not hits and not backups:
        return {}, {}, {}

    ticks = sorted(set(list(hits.keys()) + list(backups.keys())))
    hit_rate = {}
    avg_backup = {}
    weighted_hit_rate = {}
    for t in ticks:
        hlist = hits.get(t, [])
        hit_rate[t] = float(sum(hlist)) / float(len(hlist)) if hlist else float("nan")
        blist = backups.get(t, [])
        avg_backup[t] = float(np.mean(blist)) if blist else float("nan")
        whlist = weighted_hits.get(t, [])
        weighted_hit_rate[t] = float(np.mean(whlist)) if whlist else float("nan")
    return hit_rate, avg_backup, weighted_hit_rate

def _sanitize_name(s: str) -> str:
    """把路径替换为安全的文件名：用下划线代替非法字符，并限制长度。"""
    s = s.rstrip(os.sep)
    s = s.replace("..", "_")
    s = re.sub(r"[^0-9A-Za-z._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if len(s) > 200:
        s = s[-200:]
    return s or "batch_run"

def save_metrics_per_batch(fname_base: str, out_dir: Path, hit_rate: Dict[int, float], avg_backup: Dict[int, float], weighted_hit_rate: Dict[int, float], histories: List[Any], weight_const: float = 32.0) -> List[Path]:
    """
    将每个 batch 的 per-tick 指标与全局汇总保存为 JSON 和 CSV，返回写入的路径列表。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ticks = sorted(set(list(hit_rate.keys()) + list(avg_backup.keys())))
    records = []
    for t in ticks:
        records.append({
            "tick": int(t), 
            "hit_rate": None if np.isnan(hit_rate.get(t, np.nan)) else float(hit_rate.get(t)), 
            "avg_backup": None if np.isnan(avg_backup.get(t, np.nan)) else float(avg_backup.get(t)), 
            "weighted_hit_rate": None if np.isnan(weighted_hit_rate.get(t, np.nan)) else float(weighted_hit_rate.get(t))})

    # 全局统计
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

    # 计算全局加权 hit rate（ISR_w）：对所有 ticks 求平均，N = len(ticks)
    if ticks:
        sum_contribs = sum(float(weighted_hit_rate.get(t, 0.0)) for t in ticks)
        isr_w = sum_contribs / float(len(ticks))
    else:
        isr_w = float("nan")

    out_paths: List[Path] = []
    # JSON
    json_obj = {
        "records": records, 
        "global": {
            "global_hit_rate": None if np.isnan(global_hit_rate) else float(global_hit_rate), 
            "global_avg_backup": None if np.isnan(global_avg_backup) else float(global_avg_backup), 
            "total_samples": int(total_samples), 
            "total_hits": int(total_hits),
            "weight_const": float(weight_const),
            "ISR_w": None if np.isnan(isr_w) else float(isr_w)
            },
        }
    json_path = out_dir / f"{fname_base}_performance.json"
    with json_path.open("w", encoding="utf-8") as jf:
        json.dump(json_obj, jf, ensure_ascii=False, indent=2)
    out_paths.append(json_path)

    # CSV
    df = pd.DataFrame(records)
    summary_row = {
        "tick": "GLOBAL", 
        "hit_rate": (float(global_hit_rate) if not np.isnan(global_hit_rate) else None), 
        "avg_backup": (float(global_avg_backup) if not np.isnan(global_avg_backup) else None),
        "weighted_hit_rate": (float(isr_w) if not np.isnan(isr_w) else None),
    }
    df_csv = df.copy().astype(object)
    df_csv["total_samples"] = None
    df_csv["total_hits"] = None
    df_csv = pd.concat([df_csv, pd.DataFrame([summary_row])], ignore_index=True)
    csv_path = out_dir / f"{fname_base}_performance.csv"
    df_csv.to_csv(csv_path, index=False)
    out_paths.append(csv_path)

    return out_paths

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="收集所有 batch_run 下的 tick_history.json，并计算 per-tick hit rate 与 avg backup（不绘图）。")
    parser.add_argument("root", nargs="?", default=".", help="要扫描的根目录（默认当前目录）")
    parser.add_argument("-o", "--out-dir", type=Path, default=Path("performance_results"), help="每个 batch_run 输出 JSON/CSV 的目录")
    parser.add_argument("--save-agg", action="store_true", help="同时保存原始聚合条目到 aggregated.json")
    args = parser.parse_args()

    root = Path(args.root)
    logging.info("Scanning %s for batch_run directories...", root)
    entries = collect_all(root)
    if not entries:
        raise SystemExit("未找到任何 batch_run 下的 tick_history.json。")

    if args.save_agg:
        try:
            agg_path = Path("collected_tick_histories.json")
            with agg_path.open("w", encoding="utf-8") as f:
                json.dump(entries, f, ensure_ascii=False, indent=2)
            logging.info("Saved aggregated tick histories -> %s", agg_path.resolve())
        except Exception as e:
            logging.warning("Failed to save aggregated entries: %s", e)

    # 按 batch_run 分组并计算指标
    by_batch: Dict[str, List[Dict[str, Any]]] = {}
    for e in entries:
        key = e["batch_run_path"]
        by_batch.setdefault(key, []).append(e)

    saved_files: List[str] = []
    for batch_path, group in sorted(by_batch.items()):
        histories = [e["tick_history"] for e in group if e.get("tick_history")]

        hit_rate, avg_backup, weighted_hit_rate = _aggregate_tick_stats(histories)
        if not hit_rate and not avg_backup:
            logging.info("No valid tick data for %s, skipping", batch_path)
            continue

        parent = Path(batch_path).parent
        parts = [p for p in parent.parts if p not in (os.sep, "/")]
        last_parts = parts[-3:] if len(parts) >= 3 else parts
        fname_base = _sanitize_name("_".join(last_parts)) if last_parts else _sanitize_name(parent.name)

        out_paths = save_metrics_per_batch(fname_base, args.out_dir, hit_rate, avg_backup, weighted_hit_rate, histories)
        saved_files.extend([str(p) for p in out_paths])
        logging.info("Saved metrics for %s -> %s", batch_path, ", ".join([str(p) for p in out_paths]))

    if saved_files:
        print("Saved metrics files:")
        for p in saved_files:
            print("  " + p)
    else:
        print("No metrics files generated.")

if __name__ == "__main__":
    main()
# ...existing code...