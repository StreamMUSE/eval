#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import csv
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


def _parse_interval_gen(name: str) -> Tuple[Optional[int], Optional[int]]:
    s = name.lower()
    m_int = re.search(r"interval[_-]?(\d+)", s)
    interval = int(m_int.group(1)) if m_int else None
    m_gf = re.search(r"gen[_-]?frame[_-]?(\d+)", s)
    if m_gf:
        return interval, int(m_gf.group(1))
    if m_int:
        start_pos = m_int.end()
        m_after = re.search(r"gen[_-]?(\d+)", s[start_pos:])
        if m_after:
            return interval, int(m_after.group(1))
    m_any = re.search(r"gen[_-]?(\d+)", s)
    if m_any:
        return interval, int(m_any.group(1))
    return None, None


def compute_nll_metrics(p: Path) -> Tuple[Optional[float], Optional[float], int]:
    """
    返回 (ave_nll, weighted_ave_nll, n_entries)
      - ave_nll: entries 中 avg_nll 的算术平均（跳过缺失）
      - weighted_ave_nll: sum(total_nll) / sum(total_tokens) （若 sum tokens == 0 则 None）
    支持的 input 形式：
      - list[dict]（每项含 avg_nll/total_nll/total_tokens）
      - dict where values are dicts (如你提供的 experiments*.json)
      - dict with keys 'entries' or 'results' -> list
    """
    obj = json.loads(p.read_text(encoding="utf-8"))

    # 识别 entries 数据来源
    entries = []
    if isinstance(obj, dict):
        # 1) 如果 dict 的 value 都是 dict 且多数包含 avg_nll/total_nll，视为 mapping-of-entries
        vals = list(obj.values())
        if (
            vals
            and all(isinstance(v, dict) for v in vals)
            and any(("avg_nll" in v or "total_nll" in v or "total_tokens" in v) for v in vals)
        ):
            entries = vals
        # 2) 明确的 entries 或 results 字段
        elif "entries" in obj and isinstance(obj["entries"], list):
            entries = obj["entries"]
        elif "results" in obj and isinstance(obj["results"], list):
            entries = obj["results"]
        else:
            # 3) 退回：搜索 dict 中的第一个合格 list
            for v in vals:
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    if any(("avg_nll" in it or "total_nll" in it or "total_tokens" in it) for it in v):
                        entries = v
                        break
    elif isinstance(obj, list):
        entries = obj

    # 计算指标
    sum_avg = 0.0
    cnt_avg = 0
    sum_total_nll = 0.0
    sum_total_tokens = 0
    for e in entries:
        if not isinstance(e, dict):
            continue
        avg = e.get("avg_nll")
        if avg is None:
            avg = e.get("avg_loss") or e.get("mean_nll")
        try:
            if avg is not None:
                sum_avg += float(avg)
                cnt_avg += 1
        except Exception:
            pass
        tn = e.get("total_nll") or e.get("total_loss") or e.get("sum_nll")
        tt = e.get("total_tokens") or e.get("tokens") or e.get("total_len") or e.get("total_tokens_count")
        try:
            if tn is not None:
                sum_total_nll += float(tn)
            if tt is not None:
                sum_total_tokens += int(tt)
        except Exception:
            pass

    ave_nll = (sum_avg / cnt_avg) if cnt_avg > 0 else None
    weighted = (sum_total_nll / sum_total_tokens) if sum_total_tokens > 0 else None
    return ave_nll, weighted, len(entries)


def find_experiments_jsons(nll_runs_dir: Path) -> List[Path]:
    if not nll_runs_dir.exists():
        return []
    return sorted(
        [p for p in nll_runs_dir.iterdir() if p.is_file() and p.name.startswith("experiments") and p.suffix == ".json"]
    )


def best_match_row_to_json(row_name: str, candidates: List[str]) -> Optional[str]:
    rn = row_name.lower()
    # exact
    for c in candidates:
        if rn == c.lower():
            return c
    # containment
    for c in candidates:
        if rn in c.lower() or c.lower() in rn:
            return c
    # numeric interval/gen match
    r_int, r_gen = _parse_interval_gen(rn)
    if r_int is not None and r_gen is not None:
        for c in candidates:
            ci, cg = _parse_interval_gen(c)
            if ci == r_int and cg == r_gen:
                return c
    # fallback: underscore-free containment
    rn2 = rn.replace("_", "")
    for c in candidates:
        if rn2 in c.lower().replace("_", "") or c.lower().replace("_", "") in rn2:
            return c
    return None


def main(root: str, out_name: Optional[str] = None, nll_dir: Optional[str] = None) -> None:
    rootp = Path(root).expanduser().resolve()
    if not rootp.is_dir():
        raise SystemExit(f"错误：{root} 不是目录")

    # 找到 XXX music quality summary.csv（只看顶层）
    csv_candidates = [p for p in rootp.iterdir() if p.is_file() and p.name.endswith("music quality summary.csv")]
    if not csv_candidates:
        raise SystemExit("未在目录顶层找到 '*music quality summary.csv' 文件")
    csv_path = csv_candidates[0]

    # 读取 CSV
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    # 收集 experiments*.json
    # --nll-dir 可以是任意路径（e.g. 服务器上的 StreamMUSE-1 输出目录）
    # 默认退回 results-exp/nll_runs/
    if nll_dir:
        nll_runs_dir = Path(nll_dir).expanduser().resolve()
    else:
        nll_runs_dir = rootp / "nll_runs"
    exp_files = find_experiments_jsons(nll_runs_dir)
    metrics_map: Dict[str, Dict[str, Any]] = {}
    for jf in exp_files:
        ave, weighted, n_entries = compute_nll_metrics(jf)
        metrics_map[jf.stem] = {"ave_nll": ave, "weighted_ave_nll": weighted, "n_entries": n_entries, "path": str(jf)}

    # 匹配并为每行添加两列
    json_names = list(metrics_map.keys())
    if "filename" not in (fn := [fn for fn in fieldnames if fn is not None]):
        # try common name guess
        pass
    # ensure header contains new columns
    new_cols = ["ave_nll", "weighted_ave_nll"]
    out_fieldnames = fieldnames[:] if fieldnames else ["filename"]  # safe fallback
    for c in new_cols:
        if c not in out_fieldnames:
            out_fieldnames.append(c)

    # For each CSV row determine key to match: prefer 'filename' column if present
    key_col = "filename" if "filename" in out_fieldnames else out_fieldnames[0]

    for r in rows:
        row_name = (r.get("filename") or "").strip()
        if not row_name:
            # try first column
            firstcol = list(r.values())[0] if r else ""
            row_name = str(firstcol).strip()
        matched = best_match_row_to_json(row_name, json_names)
        if matched:
            mm = metrics_map[matched]
            r["ave_nll"] = "" if mm["ave_nll"] is None else mm["ave_nll"]
            r["weighted_ave_nll"] = "" if mm["weighted_ave_nll"] is None else mm["weighted_ave_nll"]
        else:
            r["ave_nll"] = ""
            r["weighted_ave_nll"] = ""

    # 写出新的 CSV（不覆盖，默认添加 suffix）
    out_path = rootp / (out_name if out_name else (csv_path.stem + " with nll.csv"))
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"完成：输出 -> {out_path}")
    print(f"从 nll_runs 中找到 {len(exp_files)} 个 experiments*.json；匹配到的行会填充 ave_nll / weighted_ave_nll。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="在 music quality summary CSV 中加入来自 experiments*.json 的 nll 指标"
    )
    parser.add_argument("folder", help="包含 summary CSV 的根目录（results-exp1/）")
    parser.add_argument("--out", "-o", help="输出 CSV 文件名（相对于 folder）", default=None)
    parser.add_argument(
        "--nll-dir",
        help=(
            "存放 experiments*.json 的目录。"
            "可以是任意绝对路径，例如 StreamMUSE-1 服务器输出目录。"
            "若不指定，默认使用 <folder>/nll_runs/"
        ),
        default=None,
    )
    args = parser.parse_args()
    main(args.folder, args.out, args.nll_dir)
