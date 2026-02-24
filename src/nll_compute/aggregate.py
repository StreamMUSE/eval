"""Aggregation utilities for per-file NLL JSONs into dataset-level statistics.

Reads raw NLL JSON output (produced by StreamMUSE's nll_compute module)
and produces dataset-level summary statistics using eval_toolkit.stats.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from eval_toolkit.stats import compute_stats


def aggregate_nll(input_json: Union[str, Path]) -> Dict[str, Any]:
    """Read a raw NLL JSON file and compute summary statistics.

    The input JSON is expected to have the structure produced by
    `nll_compute.batch.make_nll_dir` in the StreamMUSE repository:
    {
        "<filename>": {
            "total_nll": float,
            "avg_nll": float,
            "total_tokens": int
        },
        ...
    }

    Returns a dict with two top-level keys:
        "summary": dataset-level aggregate statistics
        "per_file": per-file parsed values (or error strings)
    """
    p = Path(input_json)
    if not p.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_json}")

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    files_count = len(data)
    total_tokens = 0
    total_nll_sum = 0.0

    avg_nll_list = []
    total_nll_list = []
    per_file: Dict[str, Any] = {}

    for fname, val in data.items():
        if not isinstance(val, dict):
            continue
        if "error" in val:
            per_file[fname] = {"error": val["error"]}
            continue
        try:
            total_nll = float(val["total_nll"])
            avg_nll = float(val["avg_nll"])
            tokens = int(val["total_tokens"])
        except Exception as e:
            per_file[fname] = {"error": f"parse_error: {e}"}
            continue

        total_tokens += tokens
        total_nll_sum += total_nll
        avg_nll_list.append(avg_nll)
        total_nll_list.append(total_nll)
        per_file[fname] = {"total_nll": total_nll, "avg_nll": avg_nll, "total_tokens": tokens}

    successful = len(avg_nll_list)

    avg_stats = compute_stats(avg_nll_list)
    total_stats = compute_stats(total_nll_list)

    weighted_avg_nll: Optional[float] = None
    if total_tokens > 0:
        weighted_avg_nll = total_nll_sum / float(total_tokens)

    summary = {
        "files_count": files_count,
        "files_successful": successful,
        "total_tokens": total_tokens,
        "total_nll_sum": total_nll_sum,
        "weighted_avg_nll": weighted_avg_nll,
        "per_file_stats": {
            "avg_nll": avg_stats,
            "total_nll": total_stats,
        },
    }

    return {"summary": summary, "per_file": per_file}
