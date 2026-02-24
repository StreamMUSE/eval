"""Aggregation utilities for per-file NLL JSONs into dataset-level statistics."""

from __future__ import annotations

import os
import json
from typing import Dict, Any, Optional
import math
import statistics


def aggregate_nll(input_json: str) -> Dict[str, Any]:
    """Read input_json and compute summary statistics.

    Returns a dict suitable for JSON serialization.
    """
    if not os.path.exists(input_json):
        raise FileNotFoundError(f"Input JSON not found: {input_json}")

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    files_count = len(data)
    total_tokens = 0
    total_nll_sum = 0.0

    avg_nll_list = []
    total_nll_list = []
    per_file = {}

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

    def safe_stats(xs):
        if not xs:
            return {"count": 0, "mean": None, "std": None, "var": None, "median": None, "min": None, "max": None}
        mean = statistics.mean(xs)
        var = statistics.pvariance(xs) if len(xs) > 0 else 0.0
        std = math.sqrt(var)
        med = statistics.median(xs)
        return {"count": len(xs), "mean": mean, "std": std, "var": var, "median": med, "min": min(xs), "max": max(xs)}

    avg_stats = safe_stats(avg_nll_list)
    total_stats = safe_stats(total_nll_list)

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
