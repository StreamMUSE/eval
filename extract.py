#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

def safe_float(d: Dict[str, Any], key: str) -> Optional[float]:
    v = d.get(key)
    if v is None:
        return None
    try:
        fv = float(v)
        if math.isnan(fv):
            return None
        return fv
    except Exception:
        return None

def collect_globals(root: Path) -> List[Dict[str, Any]]:
    root = root.expanduser().resolve()
    results: List[Dict[str, Any]] = []
    for final_dir in root.rglob("final-sys-results"):
        if not final_dir.is_dir():
            continue
        for jf in sorted(final_dir.rglob("*.json")):
            try:
                with jf.open("r", encoding="utf-8") as f:
                    obj = json.load(f)
            except Exception:
                continue
            glob = obj.get("global")
            if not isinstance(glob, dict):
                continue
            results.append({
                "source_path": str(jf.resolve()),
                "rel_path": str(jf.resolve().relative_to(root)) if root in jf.resolve().parents or jf.resolve()==root else str(jf.resolve().name),
                "filename": jf.name,
                "global": glob,
            })
    return results

def summarize(globals_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(globals_list)
    if n == 0:
        return {"num_files": 0}
    sum_hit = 0.0; cnt_hit = 0
    sum_isrw = 0.0; cnt_isrw = 0
    sum_avgb = 0.0; cnt_avgb = 0
    total_samples = 0
    total_hits = 0
    for item in globals_list:
        g = item["global"]
        v = safe_float(g, "global_hit_rate")
        if v is not None:
            sum_hit += v; cnt_hit += 1
        v = safe_float(g, "ISR_w")
        if v is not None:
            sum_isrw += v; cnt_isrw += 1
        v = safe_float(g, "global_avg_backup")
        if v is not None:
            sum_avgb += v; cnt_avgb += 1
        ts = g.get("total_samples")
        th = g.get("total_hits")
        try:
            if ts is not None:
                total_samples += int(ts)
            if th is not None:
                total_hits += int(th)
        except Exception:
            pass
    return {
        "num_files": n,
        "avg_global_hit_rate": (sum_hit / cnt_hit) if cnt_hit else None,
        "avg_ISR_w": (sum_isrw / cnt_isrw) if cnt_isrw else None,
        "avg_global_avg_backup": (sum_avgb / cnt_avgb) if cnt_avgb else None,
        "total_samples_sum": total_samples,
        "total_hits_sum": total_hits,
    }

def main() -> None:
    p = argparse.ArgumentParser(description="从 final-sys-results 里收集每个 json 的 global 字段，汇总成一个 json。")
    p.add_argument("root", type=Path, help="要扫描的根目录")
    p.add_argument("--out", "-o", type=Path, default=Path("final_sys_globals_summary.json"), help="输出 JSON 文件")
    args = p.parse_args()

    globals_list = collect_globals(args.root)
    summary = summarize(globals_list)
    out_obj = {"root": str(args.root.resolve()), "collected": globals_list, "summary": summary}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)
    print(f"Saved -> {args.out}")

if __name__ == "__main__":
    main()