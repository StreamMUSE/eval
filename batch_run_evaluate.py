#!/usr/bin/env python3
"""
批量查找所有名为 `generated` 的目录并对每个目录运行 evaluate_accompaniment_metrics.py。
输出文件名基于路径中提取的 interval/gen_frame/prompt/gen 等信息。

用法示例：
  python3 scripts/batch_run_evaluate.py \
    --scan-root /home/ubuntu/ugrip/stanleyz/StreamMUSE/experiments1 \
    --groundtruth-root /home/ubuntu/ugrip/stanleyz/StreamMUSE/experiments1/baseline/Baseline0.12B/prompt128_gen576/gt_without_prompt \
    --out-root results-experiment1 \
    --melody Guitar \
    --polydis-root ./icm-deep-music-generation/ \
    --dry-run

默认按发现顺序顺序执行；可用 --workers 并发（简单进程池）。
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed


RE_INTERVAL = re.compile(r"interval[_\-]?(\d+)", re.IGNORECASE)
RE_GEN_FRAME = re.compile(r"gen[_\-]?frame[_\-]?(\d+)|gen_frame[_\-]?(\d+)|genframe[_\-]?(\d+)", re.IGNORECASE)
RE_PROMPT_GEN = re.compile(r"prompt[_\-]?(\d+)[_\-]?gen[_\-]?(\d+)|prompt(\d+)[_\-]?gen(\d+)", re.IGNORECASE)
RE_GEN_ALONE = re.compile(r"gen[_\-]?(\d+)(?:$|[_/\\])", re.IGNORECASE)


def extract_params(p: Path) -> dict:
    s = str(p)
    interval = None
    gen_frame = None
    prompt = None
    gen = None

    m = RE_INTERVAL.search(s)
    if m:
        interval = m.group(1)

    m = RE_GEN_FRAME.search(s)
    if m:
        # regex has multiple groups; pick first non-None
        gen_frame = next(g for g in m.groups() if g)

    m = RE_PROMPT_GEN.search(s)
    if m:
        groups = [g for g in m.groups() if g]
        if len(groups) >= 2:
            prompt, gen = groups[0], groups[1]

    # fallback: try to find a gen number elsewhere
    if gen is None:
        m2 = RE_GEN_ALONE.search(s)
        if m2:
            gen = m2.group(1)

    return {"interval": interval or "NA", "gen_frame": gen_frame or "NA", "prompt": prompt or "NA", "gen": gen or "NA"}


def make_output_name(params: dict) -> str:
    return f"interval{params['interval']}_genframe{params['gen_frame']}_prompt{params['prompt']}_gen{params['gen']}.json"


def run_one(generated_dir: Path, groundtruth_root: Path, out_root: Path, melody: str, polydis_root: Path,
            extra_args: Optional[str], dry_run: bool) -> tuple[Path, int]:
    params = extract_params(generated_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    out_name = make_output_name(params)
    out_path = out_root / out_name

    cmd = [
        "uv", "run", "evaluate_accompaniment_metrics.py",
        "--generated-dir", str(generated_dir),
        "--groundtruth-dir", str(groundtruth_root),
        "--output-json", str(out_path),
        "--melody-track-names", melody,
        "--auto-phrase-analysis",
        "--frechet-music-distance",
        "--polydis-root", str(polydis_root),
    ]
    if extra_args:
        # naive split; callers can pass quoted args
        cmd += extra_args.split()

    print("CMD:", " ".join(cmd))
    if dry_run:
        return out_path, 0

    proc = subprocess.run(cmd)
    return out_path, proc.returncode


def find_generated_dirs(scan_root: Path) -> list[Path]:
    res = []
    for p in scan_root.rglob("generated"):
        if p.is_dir():
            res.append(p)
    return sorted(res)


def main():
    parser = argparse.ArgumentParser(description="批量运行 evaluate_accompaniment_metrics.py（按 generated 目录遍历）")
    parser.add_argument("--scan-root", type=Path, required=True, help="根目录，递归查找名为 generated 的目录")
    parser.add_argument("--groundtruth-root", type=Path, required=True, help="对应的 groundtruth 根目录（传给 --groundtruth-dir）")
    parser.add_argument("--out-root", type=Path, required=True, help="保存 output json 的目录")
    parser.add_argument("--melody", default="Guitar", help="--melody-track-names 参数")
    parser.add_argument("--polydis-root", type=Path, required=True)
    parser.add_argument("--extra-args", default="", help="传递给命令的额外参数（字符串形式）")
    parser.add_argument("--workers", type=int, default=1, help="并发 worker 数，默认 1（顺序执行）")
    parser.add_argument("--dry-run", action="store_true", help="只打印命令不执行")
    args = parser.parse_args()

    gen_dirs = find_generated_dirs(args.scan_root)
    if not gen_dirs:
        print("未找到任何 generated 目录。")
        return

    print(f"发现 {len(gen_dirs)} 个 generated 目录，保存结果到 {args.out_root}")
    results = []
    if args.workers <= 1:
        for g in gen_dirs:
            out_path, rc = run_one(g, args.groundtruth_root, args.out_root, args.melody, args.polydis_root, args.extra_args, args.dry_run)
            results.append((g, out_path, rc))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(run_one, g, args.groundtruth_root, args.out_root, args.melody, args.polydis_root, args.extra_args, args.dry_run): g for g in gen_dirs}
            for fut in as_completed(futures):
                g = futures[fut]
                out_path, rc = fut.result()
                results.append((g, out_path, rc))

    # 简要报告
    for g, outp, rc in results:
        status = "OK" if rc == 0 else f"ERROR({rc})"
        print(f"{g} -> {outp} : {status}")


if __name__ == "__main__":
    main()