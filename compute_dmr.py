#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import List, Optional, Dict, Any

def _sanitize_name(s: str) -> str:
    s = s.rstrip(os.sep)
    s = s.replace("..", "_")
    s = re.sub(r"[^0-9A-Za-z._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if len(s) > 200:
        s = s[-200:]
    return s or "batch_run"

# 只接受严格形式 gen_frame_{number}
RE_GEN_FRAME = re.compile(r"gen[_-]?frame[_-]?(\d+)", re.IGNORECASE)

def find_batch_runs(root: Path) -> List[Path]:
    root = root.expanduser().resolve()
    return [p for p in root.rglob("batch_run") if p.is_dir()]

def _extract_frames_from_path(p: Path) -> Optional[int]:
    """严格从父目录名中匹配 gen_frame_{number}（最近的匹配优先）。"""
    for parent in p.parents:
        m = RE_GEN_FRAME.search(parent.name)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
    return None

def _read_inference_file(p: Path) -> Optional[List[Dict[str, Any]]]:
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return None
    except Exception as e:
        logging.warning("Failed to read inference file %s: %s", p, e)
        return None

def compute_dmr_for_batch(batch_dir: Path, time_per_tick: float) -> Optional[Dict[str, Any]]:
    """
    遍历 batch_run 的直接子目录，查找每个子目录下名为 inference.json 的文件，
    统计所有 response 的 round_trip_time，与 deadline 比较得到 DMR。

    tick 计算规则：gen_ticks = (generation_frames + 1) // 2
    deadline_seconds = gen_ticks * time_per_tick
    """
    # collect inference.json from immediate subdirectories (非递归)
    items: List[tuple[Path, List[Dict[str, Any]]]] = []
    for sub in sorted(batch_dir.iterdir()):
        if not sub.is_dir():
            continue
        inf = sub / "inferences.json"
        if not inf.exists() or not inf.is_file():
            continue
        data = _read_inference_file(inf)
        if not data:
            continue
        items.append((sub, data))

    if not items:
        logging.info("No inferences.json found under %s", batch_dir)
        return None

    # extract generation frames from path (using batch_dir parents)
    frames = _extract_frames_from_path(batch_dir)
    if frames is None:
        logging.warning("Could not infer generation frames from path %s; skipping", batch_dir)
        return None

    gen_frames = int(frames)
    gen_ticks = (gen_frames + 1) // 2  # frames -> ticks
    deadline = gen_ticks * float(time_per_tick)

    total_responses = 0
    miss_count = 0
    values = []  # store per-response details optionally

    for sub, data in items:
        for rec in data:
            resp = rec.get("response") or {}
            timings = resp.get("timings") or {}
            rtt = timings.get("round_trip_time")
            # fallback: compute from response_output_time - request_arrival_time
            if rtt is None:
                ra = timings.get("request_arrival_time")
                ro = timings.get("response_output_time")
                if ra is not None and ro is not None:
                    try:
                        rtt = float(ro) - float(ra)
                    except Exception:
                        rtt = None
            if rtt is None:
                # skip if no timing info
                continue
            total_responses += 1
            is_miss = float(rtt) > deadline
            if is_miss:
                miss_count += 1
            values.append({"subdir": str(sub), "round_trip_time": float(rtt), "miss": is_miss})

    if total_responses == 0:
        logging.warning("No valid response timings found under %s", batch_dir)
        return None

    dmr = float(miss_count) / float(total_responses)
    # fname base: use batch_dir's parent last three parts
    parent = batch_dir.parent
    parts = [p for p in parent.parts if p not in (os.sep, "/")]
    last_parts = parts[-3:] if len(parts) >= 3 else parts
    if last_parts:
        fname_base = _sanitize_name("_".join(last_parts))
    else:
        fname_base = _sanitize_name(parent.name)

    return {
        "batch_run_path": str(batch_dir),
        "fname_base": fname_base,
        "generation_frames": gen_frames,
        "generation_ticks": gen_ticks,
        "time_per_tick": float(time_per_tick),
        "deadline_seconds": float(deadline),
        "total_responses": int(total_responses),
        "miss_count": int(miss_count),
        "dmr": float(dmr),
        "details_sample": values[:100],  # keep small sample for inspection
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Deadline Miss Ratio (DMR) per batch_run from inference.json files.")
    parser.add_argument("--root", type=Path, default=Path("."), help="根目录，递归查找 batch_run")
    parser.add_argument("--time-per-tick", type=float, default=1/6, help="每个 tick 的时长（秒），deadline = gen_ticks * time_per_tick")
    parser.add_argument("--out-dir", type=Path, default=Path("dmr_results"), help="输出结果目录")
    parser.add_argument("--min-sample", action="store_true", help="输出时只保留 details_sample 而不是全部 detail（避免大文件）")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    root = args.root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_runs = find_batch_runs(root)
    if not batch_runs:
        logging.error("No batch_run directories found under %s", root)
        return

    results = []
    for b in sorted(batch_runs):
        res = compute_dmr_for_batch(b, args.time_per_tick)
        if res is None:
            continue
        # save per-batch file
        out_path = out_dir / f"{res['fname_base']}_dmr.json"
        # if min_sample requested, ensure details_sample only
        if args.min_sample:
            res_to_write = {k: v for k, v in res.items() if k != "details_sample"}
            res_to_write["details_sample"] = res.get("details_sample", [])
        else:
            res_to_write = res
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(res_to_write, f, indent=2, ensure_ascii=False)
        logging.info("Saved DMR -> %s (dmr=%.4f, responses=%d)", out_path, res["dmr"], res["total_responses"])
        results.append(str(out_path))

    if results:
        print("Saved DMR files:")
        for p in results:
            print("  " + p)
    else:
        print("No DMR results generated.")

if __name__ == "__main__":
    main()