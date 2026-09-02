"""Summarize matched music metrics with piece-cluster bootstrap intervals."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_PIECE_COUNT = 40
EXPECTED_SEEDS = ("0", "1", "2")
DEFAULT_BOOTSTRAP_REPLICATES = 10_000
DEFAULT_BOOTSTRAP_SEED = 0
DETAIL_METRICS = {
    "pitch_jsd": ("pitch_jsd",),
    "onset_jsd": ("onset_jsd",),
    "duration_jsd": ("duration_jsd",),
    "cr": ("harmonicity", "consonant_ratio"),
    "ur": ("harmonicity", "unsupported_ratio"),
}
SUMMARY_COUNT_PATHS = {
    "pitch_jsd": ("accompaniment_vs_groundtruth", "pitch_jsd", "count"),
    "onset_jsd": ("accompaniment_vs_groundtruth", "onset_jsd", "count"),
    "duration_jsd": ("accompaniment_vs_groundtruth", "duration_jsd", "count"),
    "cr": ("melody_relationship", "harmonicity", "consonant_ratio", "count"),
    "ur": ("melody_relationship", "harmonicity", "unsupported_ratio", "count"),
}
FMD_PATH = ("accompaniment_vs_groundtruth", "frechet_music_distance")
CSV_FIELDS = (
    "system_id",
    "metric",
    "estimate",
    "ci_low",
    "ci_high",
    "ci_status",
    "bootstrap_valid_replicates",
    "numerator",
    "denominator",
    "scope",
)


class MusicSummaryValidationError(ValueError):
    """Raised when matched audit or metric inputs violate the formal contract."""


@dataclass(frozen=True)
class AuditTrial:
    piece_id: str
    seed: str
    system_id: str
    valid_output: bool
    line_number: int


@dataclass(frozen=True)
class SystemInput:
    trials: tuple[AuditTrial, ...]
    details: dict[tuple[str, str], dict[str, float]]
    fmd: float | None
    metrics_path: Path


def _finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MusicSummaryValidationError(f"{context} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise MusicSummaryValidationError(f"{context} must be a finite number")
    return number


def _strict_integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MusicSummaryValidationError(f"{context} must be an integer")
    return value


def _strict_bool(value: str, context: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise MusicSummaryValidationError(
        f"{context} must be exactly true or false, found {value!r}"
    )


def _nested_optional(root: Any, path: Sequence[str]) -> Any:
    current = root
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _nested_required(root: Any, path: Sequence[str], context: str) -> Any:
    current = root
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            joined = ".".join(path)
            raise MusicSummaryValidationError(f"{context} is missing {joined}")
        current = current[key]
    return current


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise MusicSummaryValidationError(f"cannot read JSON {path}: {exc}") from exc


def parse_metrics_mapping(values: Sequence[str]) -> dict[str, Path]:
    mappings: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise MusicSummaryValidationError(
                f"--metrics requires SYSTEM_ID=PATH, found {value!r}"
            )
        system_id, raw_path = value.split("=", 1)
        system_id = system_id.strip()
        raw_path = raw_path.strip()
        if not system_id or not raw_path:
            raise MusicSummaryValidationError(
                f"--metrics requires non-empty SYSTEM_ID and PATH: {value!r}"
            )
        if system_id in mappings:
            raise MusicSummaryValidationError(
                f"duplicate --metrics mapping for system {system_id!r}"
            )
        mappings[system_id] = Path(raw_path).expanduser().resolve()
    if not mappings:
        raise MusicSummaryValidationError("at least one --metrics mapping is required")
    return mappings


def load_matched_audit(
    path: Path,
    system_ids: Sequence[str],
    *,
    expected_piece_count: int,
    expected_seeds: Sequence[str],
) -> tuple[dict[str, tuple[AuditTrial, ...]], tuple[str, ...]]:
    path = path.resolve()
    if expected_piece_count <= 0:
        raise MusicSummaryValidationError("expected_piece_count must be positive")
    expected_seed_tuple = tuple(expected_seeds)
    if not expected_seed_tuple or len(set(expected_seed_tuple)) != len(
        expected_seed_tuple
    ):
        raise MusicSummaryValidationError("expected_seeds must be non-empty and unique")
    try:
        handle = path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise MusicSummaryValidationError(
            f"cannot read audit CSV {path}: {exc}"
        ) from exc

    required = {
        "piece_id",
        "seed",
        "system_id",
        "source_status",
        "preparation_status",
        "valid_output",
    }
    selected = set(system_ids)
    rows_by_system: dict[str, list[AuditTrial]] = {
        system_id: [] for system_id in system_ids
    }
    keys_by_system: dict[str, set[tuple[str, str]]] = {
        system_id: set() for system_id in system_ids
    }
    try:
        with handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise MusicSummaryValidationError(f"audit CSV has no header: {path}")
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise MusicSummaryValidationError(
                    f"audit CSV has duplicate columns: {path}"
                )
            missing_fields = sorted(required - set(reader.fieldnames))
            if missing_fields:
                raise MusicSummaryValidationError(
                    "audit CSV is missing required columns: "
                    + ", ".join(missing_fields)
                )
            for line_number, row in enumerate(reader, start=2):
                system_id = str(row.get("system_id") or "").strip()
                if system_id not in selected:
                    continue
                piece_id = str(row.get("piece_id") or "").strip()
                seed = str(row.get("seed") or "").strip()
                if not piece_id or not seed:
                    raise MusicSummaryValidationError(
                        f"audit line {line_number} has empty piece_id or seed"
                    )
                if str(row.get("source_status") or "").strip() != "complete":
                    raise MusicSummaryValidationError(
                        f"audit line {line_number} is not a complete source trial"
                    )
                if str(row.get("preparation_status") or "").strip() != "prepared":
                    raise MusicSummaryValidationError(
                        f"audit line {line_number} is not prepared"
                    )
                key = (piece_id, seed)
                if key in keys_by_system[system_id]:
                    raise MusicSummaryValidationError(
                        f"duplicate audit key for {system_id}/{piece_id}/seed {seed}"
                    )
                keys_by_system[system_id].add(key)
                rows_by_system[system_id].append(
                    AuditTrial(
                        piece_id=piece_id,
                        seed=seed,
                        system_id=system_id,
                        valid_output=_strict_bool(
                            str(row.get("valid_output") or ""),
                            f"audit line {line_number} valid_output",
                        ),
                        line_number=line_number,
                    )
                )
    except csv.Error as exc:
        raise MusicSummaryValidationError(f"invalid audit CSV {path}: {exc}") from exc

    reference_pieces: tuple[str, ...] | None = None
    expected_seed_set = set(expected_seed_tuple)
    for system_id in system_ids:
        trials = rows_by_system[system_id]
        if not trials:
            raise MusicSummaryValidationError(
                f"audit has no rows for mapped system {system_id!r}"
            )
        pieces = tuple(sorted({trial.piece_id for trial in trials}))
        seeds = {trial.seed for trial in trials}
        if len(pieces) != expected_piece_count:
            raise MusicSummaryValidationError(
                f"system {system_id!r} must contain exactly {expected_piece_count} "
                f"pieces; found {len(pieces)}"
            )
        if seeds != expected_seed_set:
            raise MusicSummaryValidationError(
                f"system {system_id!r} seeds differ from expected "
                f"{sorted(expected_seed_set)}; found {sorted(seeds)}"
            )
        expected_keys = {
            (piece_id, seed) for piece_id in pieces for seed in expected_seed_tuple
        }
        actual_keys = keys_by_system[system_id]
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            raise MusicSummaryValidationError(
                f"system {system_id!r} audit grid mismatch; missing={missing[:5]}, "
                f"extra={extra[:5]}"
            )
        if reference_pieces is None:
            reference_pieces = pieces
        elif pieces != reference_pieces:
            raise MusicSummaryValidationError(
                f"system {system_id!r} does not share the same piece IDs"
            )
        rows_by_system[system_id] = sorted(
            trials, key=lambda trial: (trial.piece_id, trial.seed)
        )

    assert reference_pieces is not None
    return {
        system_id: tuple(rows_by_system[system_id]) for system_id in system_ids
    }, reference_pieces


def _load_system_metrics(
    path: Path,
    trials: Sequence[AuditTrial],
    *,
    system_id: str,
) -> SystemInput:
    data = _read_json(path)
    if not isinstance(data, Mapping):
        raise MusicSummaryValidationError(f"metrics root must be an object: {path}")
    meta = data.get("meta")
    if not isinstance(meta, Mapping) or "pairs" not in meta:
        raise MusicSummaryValidationError(f"metrics meta.pairs is required: {path}")
    pairs = _strict_integer(meta["pairs"], f"{path}: meta.pairs")
    details_raw = data.get("details")
    if not isinstance(details_raw, list):
        raise MusicSummaryValidationError(f"metrics details must be a list: {path}")

    valid_trials = {
        (trial.piece_id, trial.seed): trial for trial in trials if trial.valid_output
    }
    invalid_piece_names = {
        f"piece-{trial.piece_id}__seed-{trial.seed}"
        for trial in trials
        if not trial.valid_output
    }
    expected_names = {
        f"piece-{piece_id}__seed-{seed}": (piece_id, seed)
        for piece_id, seed in valid_trials
    }
    detail_values: dict[tuple[str, str], dict[str, float]] = {}
    seen_names: set[str] = set()
    for index, detail in enumerate(details_raw):
        context = f"{path}: details[{index}]"
        if not isinstance(detail, Mapping):
            raise MusicSummaryValidationError(f"{context} must be an object")
        piece_name = detail.get("piece")
        if not isinstance(piece_name, str):
            raise MusicSummaryValidationError(f"{context}.piece must be a string")
        if piece_name in seen_names:
            raise MusicSummaryValidationError(
                f"duplicate metrics detail piece {piece_name!r}"
            )
        seen_names.add(piece_name)
        if piece_name in invalid_piece_names:
            raise MusicSummaryValidationError(
                f"invalid-output trial appears in metrics details: {piece_name}"
            )
        trial_key = expected_names.get(piece_name)
        if trial_key is None:
            raise MusicSummaryValidationError(
                f"unexpected metrics detail piece for {system_id!r}: {piece_name!r}"
            )
        values: dict[str, float] = {}
        for metric, metric_path in DETAIL_METRICS.items():
            values[metric] = _finite_number(
                _nested_required(detail, metric_path, context),
                f"{context}.{'.'.join(metric_path)}",
            )
        detail_values[trial_key] = values

    missing_names = sorted(set(expected_names) - seen_names)
    if missing_names:
        raise MusicSummaryValidationError(
            f"metrics details are missing valid trials: {missing_names[:5]}"
        )
    valid_count = len(valid_trials)
    if pairs != valid_count:
        raise MusicSummaryValidationError(
            f"{path}: meta.pairs={pairs} does not equal valid detail count {valid_count}"
        )
    if len(details_raw) != valid_count:
        raise MusicSummaryValidationError(
            f"{path}: details count does not equal valid detail count {valid_count}"
        )

    summary = data.get("summary")
    if summary is not None and not isinstance(summary, Mapping):
        raise MusicSummaryValidationError(f"metrics summary must be an object: {path}")
    summary_root: Mapping[str, Any] = summary if isinstance(summary, Mapping) else {}
    for metric, count_path in SUMMARY_COUNT_PATHS.items():
        count = _nested_optional(summary_root, count_path)
        if count is None:
            continue
        parsed_count = _strict_integer(count, f"{path}: summary {metric} count")
        if parsed_count != valid_count:
            raise MusicSummaryValidationError(
                f"{path}: summary {metric} count {parsed_count} does not equal "
                f"valid detail count {valid_count}"
            )

    fmd_raw = _nested_optional(summary_root, FMD_PATH)
    fmd = None if fmd_raw is None else _finite_number(fmd_raw, f"{path}: FMD")
    if fmd is not None and valid_count == 0:
        raise MusicSummaryValidationError(
            f"{path}: non-null FMD is invalid when there are zero valid details"
        )
    return SystemInput(
        trials=tuple(trials),
        details=detail_values,
        fmd=fmd,
        metrics_path=path,
    )


def _percentile_ci(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(np.percentile(array, 2.5)), float(np.percentile(array, 97.5))


def _metric_record(
    *,
    estimate: float | None,
    ci: tuple[float, float] | None,
    ci_status: str,
    valid_replicates: int,
    numerator: float | None,
    denominator: int,
    scope: str,
) -> dict[str, Any]:
    return {
        "estimate": estimate,
        "ci_low": ci[0] if ci is not None else None,
        "ci_high": ci[1] if ci is not None else None,
        "ci_status": ci_status,
        "bootstrap_valid_replicates": valid_replicates,
        "numerator": numerator,
        "denominator": denominator,
        "scope": scope,
    }


def summarize_matched_music_metrics(
    *,
    audit_path: Path,
    metrics_paths: Mapping[str, Path],
    expected_piece_count: int = EXPECTED_PIECE_COUNT,
    expected_seeds: Sequence[str] = EXPECTED_SEEDS,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if bootstrap_replicates <= 0:
        raise MusicSummaryValidationError("bootstrap_replicates must be positive")
    if not metrics_paths:
        raise MusicSummaryValidationError("at least one metrics system is required")
    system_ids = tuple(sorted(metrics_paths))
    expected_seeds = tuple(str(seed) for seed in expected_seeds)
    audit_by_system, pieces = load_matched_audit(
        audit_path,
        system_ids,
        expected_piece_count=expected_piece_count,
        expected_seeds=expected_seeds,
    )
    system_inputs = {
        system_id: _load_system_metrics(
            Path(metrics_paths[system_id]).resolve(),
            audit_by_system[system_id],
            system_id=system_id,
        )
        for system_id in system_ids
    }

    rng = np.random.default_rng(bootstrap_seed)
    draws = rng.integers(
        0,
        len(pieces),
        size=(bootstrap_replicates, len(pieces)),
        dtype=np.int64,
    )
    draw_hash = hashlib.sha256(draws.tobytes(order="C")).hexdigest()
    piece_to_index = {piece_id: index for index, piece_id in enumerate(pieces)}
    system_results: dict[str, Any] = {}

    for system_id in system_ids:
        system_input = system_inputs[system_id]
        piece_valid = np.zeros(len(pieces), dtype=np.int64)
        piece_metric_sums = {
            metric: np.zeros(len(pieces), dtype=np.float64) for metric in DETAIL_METRICS
        }
        piece_metric_counts = np.zeros(len(pieces), dtype=np.int64)
        for trial in system_input.trials:
            piece_index = piece_to_index[trial.piece_id]
            if not trial.valid_output:
                continue
            piece_valid[piece_index] += 1
            piece_metric_counts[piece_index] += 1
            values = system_input.details[(trial.piece_id, trial.seed)]
            for metric in DETAIL_METRICS:
                piece_metric_sums[metric][piece_index] += values[metric]

        all_trial_count = len(system_input.trials)
        valid_count = int(piece_valid.sum())
        vor_estimate = valid_count / all_trial_count
        vor_replicates: list[float] = []
        conditional_replicates: dict[str, list[float]] = {
            metric: [] for metric in DETAIL_METRICS
        }
        for draw in draws:
            weights = np.bincount(draw, minlength=len(pieces))
            replicate_valid = int(weights @ piece_valid)
            vor_replicates.append(replicate_valid / all_trial_count)
            replicate_denominator = int(weights @ piece_metric_counts)
            if replicate_denominator == 0:
                continue
            for metric in DETAIL_METRICS:
                numerator = float(weights @ piece_metric_sums[metric])
                conditional_replicates[metric].append(numerator / replicate_denominator)

        metrics: dict[str, dict[str, Any]] = {
            "valid_output_rate": _metric_record(
                estimate=vor_estimate,
                ci=_percentile_ci(vor_replicates),
                ci_status="computed",
                valid_replicates=len(vor_replicates),
                numerator=valid_count,
                denominator=all_trial_count,
                scope="all_trials",
            )
        }
        for metric in DETAIL_METRICS:
            numerator = float(piece_metric_sums[metric].sum())
            if valid_count == 0:
                metrics[metric] = _metric_record(
                    estimate=None,
                    ci=None,
                    ci_status="not_computed",
                    valid_replicates=0,
                    numerator=0,
                    denominator=0,
                    scope="conditional_on_valid_output",
                )
                continue
            replicates = conditional_replicates[metric]
            metrics[metric] = _metric_record(
                estimate=numerator / valid_count,
                ci=_percentile_ci(replicates) if replicates else None,
                ci_status="computed" if replicates else "not_computed",
                valid_replicates=len(replicates),
                numerator=numerator,
                denominator=valid_count,
                scope="conditional_on_valid_output",
            )
        metrics["fmd"] = _metric_record(
            estimate=system_input.fmd,
            ci=None,
            ci_status="not_computed",
            valid_replicates=0,
            numerator=None,
            denominator=valid_count,
            scope="dataset_level_valid_output",
        )
        system_results[system_id] = {
            "metrics_path": str(system_input.metrics_path),
            "piece_count": len(pieces),
            "seed_count": len(expected_seeds),
            "trial_count": all_trial_count,
            "valid_output_count": valid_count,
            "metrics": metrics,
        }

    return {
        "schema_version": 1,
        "audit_path": str(audit_path.resolve()),
        "expected_grid": {
            "piece_count": expected_piece_count,
            "seeds": list(expected_seeds),
            "systems": list(system_ids),
        },
        "bootstrap": {
            "method": "matched_piece_cluster_percentile",
            "cluster": "piece_id",
            "retain_all_seeds": True,
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "percentiles": [2.5, 97.5],
            "piece_order": list(pieces),
            "shared_draw_matrix_sha256": draw_hash,
        },
        "systems": system_results,
    }


def write_summary_json(path: Path, summary: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_flat_csv(path: Path, summary: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        systems = summary["systems"]
        for system_id in sorted(systems):
            for metric, values in systems[system_id]["metrics"].items():
                writer.writerow(
                    {
                        "system_id": system_id,
                        "metric": metric,
                        **{field: values[field] for field in CSV_FIELDS[2:]},
                    }
                )


def _parse_expected_seeds(value: str) -> tuple[str, ...]:
    seeds = tuple(part.strip() for part in value.split(",") if part.strip())
    if not seeds or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("expected seeds must be non-empty and unique")
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize matched music metrics with piece-cluster bootstrap."
    )
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument(
        "--metrics",
        action="append",
        default=[],
        metavar="SYSTEM_ID=PATH",
        help="Repeat once for each system metrics JSON.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--expected-piece-count", type=int, default=EXPECTED_PIECE_COUNT
    )
    parser.add_argument(
        "--expected-seeds",
        type=_parse_expected_seeds,
        default=EXPECTED_SEEDS,
    )
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        metrics_paths = parse_metrics_mapping(args.metrics)
        summary = summarize_matched_music_metrics(
            audit_path=args.audit,
            metrics_paths=metrics_paths,
            expected_piece_count=args.expected_piece_count,
            expected_seeds=args.expected_seeds,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
        )
        write_summary_json(args.output_json, summary)
        write_flat_csv(args.output_csv, summary)
    except MusicSummaryValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote matched music summary for {len(summary['systems'])} system(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
