"""JSON parsing helpers for different metric types.

This module provides focused parsers for extracting evaluation metrics
from JSON files and returning normalized flat lists of metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Union, Optional

from .path_utils import (
    TypeFromResult,
    TypeFromExpRaw,
    TypeFromNll,
    Type,
    EXP_RAW,
    NLL,
    RESULT,
)


def _load_json_file(p: Path) -> dict:
    """Load and return JSON content from a file path.

    Raises FileNotFoundError / JSONDecodeError to the caller.
    """
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_result_value(item: dict, type_str: str) -> Optional[float]:
    """Extract a single value from a `detail` item according to `type_str`.

    Returns None if the requested value can't be found on the item.
    """
    if type_str in item:
        return item.get(type_str)

    if type_str in ("consonant_ratio", "unsupported_ratio"):
        h = item.get("harmonicity") if isinstance(item, dict) else None
        if isinstance(h, dict):
            return h.get(type_str)
        return None

    if type_str == "prompt_generated_txt_mean_distance":
        pg = item.get("prompt_generated_continuation_polydis")
        if isinstance(pg, dict) and "txt_mean_distance" in pg:
            return pg.get("txt_mean_distance")
        pp = item.get("prompt_polydis")
        if isinstance(pp, dict) and "txt_mean_distance" in pp:
            return pp.get("txt_mean_distance")
        return None

    if type_str == "frechet_music_distance":
        return item.get("frechet_music_distance")

    return None


def parse_result_file(key: str, type: TypeFromResult, path: Union[str, Path]) -> List[Optional[float]]:
    """Parse a single RESULT JSON file and return a list of extracted values."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    data = _load_json_file(p)
    if not isinstance(data, dict):
        return []

    details = data.get("details")
    if not isinstance(details, list):
        return []

    tstr = str(type)
    out: List[Optional[float]] = []
    for item in details:
        if not isinstance(item, dict):
            out.append(None)
            continue
        out.append(_extract_result_value(item, tstr))
    return out


def _extract_nll_values_from_data(data: object, type: TypeFromNll) -> List[float]:
    """Extract numeric NLL values from a loaded JSON object.

    Handles lists of dicts, or dictionary of dicts forms, robustly
    extracting avg_nll, total_nll and weights correctly for metric calculation.
    """
    tstr = str(type)
    out: List[Union[float, tuple]] = []

    def _process_record(rec: object):
        if not isinstance(rec, dict):
            return

        avg = rec.get("avg_nll")
        total = rec.get("total_nll")
        tokens = rec.get("total_tokens")

        def _as_float(x):
            try:
                return float(x)
            except Exception:
                return None

        if tstr == "nll":
            v = _as_float(avg)
            if v is None and total is not None and tokens is not None:
                tval = _as_float(total)
                tok = _as_float(tokens)
                if tval is not None and tok not in (None, 0):
                    v = tval / tok
            if v is not None:
                out.append(v)
            return

        if tstr == "nll_weighted":
            v_total = _as_float(total)
            v_tokens = _as_float(tokens)
            if v_total is None:
                if avg is not None and v_tokens not in (None, 0):
                    a = _as_float(avg)
                    if a is not None:
                        v_total = a * v_tokens
            if v_total is not None and v_tokens not in (None, 0):
                out.append((v_total, v_tokens))
            return

    if isinstance(data, dict):
        if any(k in data for k in ("avg_nll", "total_nll", "total_tokens")):
            _process_record(data)
        else:
            for v in data.values():
                _process_record(v)

    elif isinstance(data, list):
        for item in data:
            _process_record(item)

    if tstr == "nll_weighted":
        sum_nll = 0.0
        sum_tokens = 0.0
        for item in out:
            if isinstance(item, tuple) and len(item) == 2:
                try:
                    sum_nll += float(item[0])
                    sum_tokens += float(item[1])
                except Exception:
                    continue
        if sum_tokens > 0:
            return [sum_nll / sum_tokens]
        return []

    return out  # type: ignore


def parse_nll_file(key: str, type: TypeFromNll, path: Union[str, Path]) -> List[float]:
    """Parse an NLL result file (JSON) and return flat metrics object."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    data = _load_json_file(p)
    return _extract_nll_values_from_data(data, type)


def parse_json_dir(path: Union[str, Path]) -> List[dict]:
    p = Path(path)
    if not p.exists() or not p.is_dir():
        return []
    results: List[dict] = []
    for f in sorted(p.glob("*.json")):
        try:
            results.append(_load_json_file(f))
        except Exception:
            continue
    return results


def _extract_tick_history_from_data(data: dict, type: TypeFromExpRaw) -> List:
    tstr = str(type)

    def _map_tick_to_value(tick) -> object:
        if not isinstance(tick, dict):
            return tick

        if tstr == "hit_rate":
            val = tick.get("is_hit")
            if isinstance(val, bool):
                return 1 if val else 0
            try:
                return 1 if int(val) else 0
            except Exception:
                return 0

        if tstr == "backup_level":
            return tick.get("backup_level")

        if tstr == "hit_rate_weighted":
            backup_level = tick.get("backup_level")
            try:
                weight = (32 - float(backup_level)) / 32
            except Exception:
                weight = 0.0
            is_hit = tick.get("is_hit")
            if isinstance(is_hit, bool):
                return weight if is_hit else 0
            try:
                return weight if int(is_hit) else 0
            except Exception:
                return 0

        if tstr in tick:
            return tick.get(tstr)

        return None

    return [_map_tick_to_value(t) for t in data]


def _extract_tick_history_from_file(path: Union[str, Path], type: TypeFromExpRaw) -> List:
    p = Path(path)
    data = _load_json_file(p)
    return _extract_tick_history_from_data(data, type)


def parse_exp_raw_dir(key: str, type: TypeFromExpRaw, path: Union[str, Path]) -> List:
    p = Path(path)
    if not p.exists() or not p.is_dir():
        return []

    ticks: List = []
    files = sorted(p.glob("**/tick_history.json"))

    for f in files:
        try:
            part = _extract_tick_history_from_file(f, type)
            if part:
                ticks.extend(part)
        except Exception:
            continue

    tstr = str(type)

    def _map_tick_to_value(tick) -> object:
        if not isinstance(tick, dict):
            return tick
        if tstr == "hit_rate":
            val = tick.get("is_hit")
            if isinstance(val, bool):
                return 1 if val else 0
            try:
                return 1 if int(val) else 0
            except Exception:
                return 0
        if tstr == "backup_level":
            return tick.get("backup_level")
        if tstr == "hit_rate_weighted":
            backup_level = tick.get("backup_level")
            weight = (32 - float(backup_level)) / 32 if backup_level is not None else 0.0
            is_hit = tick.get("is_hit")
            if isinstance(is_hit, bool):
                return weight if is_hit else 0
            try:
                return weight if int(is_hit) else 0
            except Exception:
                return 0
        if tstr in tick:
            return tick.get(tstr)
        return None

    return [_map_tick_to_value(t) for t in ticks]


def parse_by_type(key: str, type: Type, path: Union[str, Path]) -> List:
    """Parse data for (key, type) using the provided `path` object."""
    t = str(type)
    if t in RESULT:
        return parse_result_file(key, type, path)  # type: ignore
    if t in NLL:
        return parse_nll_file(key, type, path)  # type: ignore
    if t in EXP_RAW:
        return parse_exp_raw_dir(key, type, path)  # type: ignore
    raise ValueError(f"Unrecognized type: {type}")
