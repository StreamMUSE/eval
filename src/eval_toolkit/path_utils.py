"""Path manipulation and string utilities.

Provides helper functions to derive result, raw, or NLL file paths
from run keys, and other string operations.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Union, List

TypeFromResult = Literal[
    "pitch_jsd",
    "onset_jsd",
    "duration_jsd",
    "consonant_ratio",
    "unsupported_ratio",
    "prompt_generated_txt_mean_distance",
    "frechet_music_distance",
    "chord_accuracy",
]

TypeFromExpRaw = Literal[
    "hit_rate",
    "hit_rate_weighted",
    "backup_level",
]

TypeFromNll = Literal[
    "nll",
    "nll_weighted",
]

Type = Union[TypeFromResult, TypeFromExpRaw, TypeFromNll]

RESULT = {
    "pitch_jsd",
    "onset_jsd",
    "duration_jsd",
    "consonant_ratio",
    "unsupported_ratio",
    "prompt_generated_txt_mean_distance",
    "frechet_music_distance",
    "chord_accuracy",
}
EXP_RAW = {
    "hit_rate",
    "hit_rate_weighted",
    "backup_level",
}

NLL = {
    "nll",
    "nll_weighted",
}


def split_at_nth_underscore(text: str, n: int) -> tuple[str, str] | None:
    """Split a string at the nth underscore (_).

    Args:
        text: Original string.
        n: The underscore to split at (e.g., n=2 splits at the second underscore).

    Returns:
        A tuple containing (part1, part2), or None if there are fewer than n underscores.
    """
    parts = text.split("_")
    if len(parts) < n + 1:
        return None
    part1 = "_".join(parts[:n])
    part2 = "_".join(parts[n:])
    return (part1, part2)


def find_unique_path_with_target(paths: list[str], target: str) -> str | None:
    """Find a unique path in a list of paths containing the target string.

    Args:
        paths: List of path strings.
        target: Substring to find, assumed to exist in exactly one path.

    Returns:
        The matched path string, or None if no match is found.
    """
    try:
        return next(path for path in paths if target in path)
    except StopIteration:
        return None


def get_dir_from_result(key: str, type: TypeFromResult, base_dir: Union[str, Path] = "result") -> Path:
    """Build the file path for a result metric.

    Parameters:
    - key: metric key/name (file name without suffix)
    - type: one of `TypeFromResult` literal values
    - base_dir: base directory or path (defaults to `'result'`)

    Returns:
    - Path to the JSON file for the metric (adds `.json` suffix).
    """
    return Path(base_dir) / (key + ".json")


def get_dir_from_exp_raw(key: str, type: TypeFromExpRaw, base_dir: Union[str, Path] = "records") -> Path:
    """Build the path to an experiment-raw `batch_run` directory.

    Parameters:
    - key: a compound key expected to contain underscores; the function
        splits it at the 2nd underscore to derive path components.
    - type: one of `TypeFromExpRaw` literal values
    - base_dir: base directory or path (defaults to `'records'`)

    Returns:
    - Path to the `batch_run` directory for the provided key.
    """
    split_res = split_at_nth_underscore(key, 2)
    if split_res is None:
        raise ValueError(f"Key {key} does not contain enough underscores to parse raw dir.")
    pre_key, post_key = split_res
    pre_key = pre_key.replace("_gen", "_gen_frame_")
    pre_key = pre_key.replace("interval", "interval_")
    return Path(base_dir) / "raw" / "realtime" / "baseline" / pre_key / post_key / "batch_run"


def get_dir_from_nll(key: str, type: TypeFromNll, base_dir: Union[str, Path] = "records") -> Path:
    """Locate the NLL result JSON file for a given key.

    Parameters:
    - key: search key used to find a unique NLL run name
    - type: one of `TypeFromNll`
    - base_dir: base directory or path containing an `nll_runs` directory

    Returns:
    - Path to the matching NLL JSON file inside `nll_runs`.
    """
    nll_runs_dir = Path(base_dir) / "nll_runs"
    nll_list = os.listdir(nll_runs_dir) if nll_runs_dir.exists() else []

    if "offline" in key:
        nll_name = find_unique_path_with_target(nll_list, "generated_without_prompt")
        if nll_name is None:
            raise FileNotFoundError(f"Could not find NLL offline target in {nll_runs_dir}")
        return nll_runs_dir / nll_name
    else:
        split_res = split_at_nth_underscore(key, 2)
        if split_res is None:
            raise ValueError(f"Key {key} does not contain enough underscores to parse NLL dir.")
        pre_key, post_key = split_res
        pre_key = pre_key.replace("_gen", "_gen_frame_")
        pre_key = pre_key.replace("interval", "interval_")
        formatted_key = pre_key + "_" + post_key
        nll_name = find_unique_path_with_target(nll_list, formatted_key)
        if nll_name is None:
            raise FileNotFoundError(f"Could not find NLL target {formatted_key} in {nll_runs_dir}")
        return nll_runs_dir / nll_name


def get_keys_from_dir(path: Union[str, Path]) -> List[str]:
    """List candidate keys (directory names) inside `path`.

    Returns only the immediate child directory names (not files). If the
    path does not exist, returns an empty list.
    """
    p = Path(path)
    if not p.exists() or not p.is_dir():
        return []
    keys: List[str] = []
    for child in p.iterdir():
        if child.is_dir() and not child.name.startswith("offline"):
            keys.append(child.name)
    return sorted(keys)


def get_path(key: str, type: Type, base_dir: Union[str, Path] | None = None) -> Path:
    """For a directory `key`, build a mapping from each key -> target Path.

    The mapping uses helper functions above depending on which literal
    `type` belongs to. If `base_dir` is provided it will be forwarded to
    the helper; otherwise each helper uses its default base (e.g. `result` or
    `records`). Raises ValueError if `type` is not recognized.
    """
    t = str(type)
    if t in RESULT:
        return get_dir_from_result(key, t, base_dir or "result")  # type: ignore
    elif t in EXP_RAW:
        return get_dir_from_exp_raw(key, t, base_dir or "records")  # type: ignore
    elif t in NLL:
        return get_dir_from_nll(key, t, base_dir or "records")  # type: ignore
    else:
        raise ValueError(f"Unrecognized type: {type}")
