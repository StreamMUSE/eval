"""Matched realtime system metrics for StreamMUSE system-trace schema v2.

This evaluator intentionally uses frame deadlines and decision-availability
spans only. MIDI events and sparse note-on rows are not delivery evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

TRACE_FILENAME = "system_trace.jsonl"
CONFIG_FILENAME = "session_config.json"
TRACE_SCHEMA_VERSION = 2
AUDIT_FIELDS = ("condition", "mode", "clock_domain")
EXPECTED_PIECE_COUNT = 40
EXPECTED_SEED_COUNT = 3
DEFAULT_BOOTSTRAP_REPLICATES = 10_000
DEFAULT_BOOTSTRAP_SEED = 0
MANIFEST_REQUIRED_FIELDS = (
    "piece_id",
    "seed",
    "system_id",
    "session_dir",
    "run_status",
    "melody_input_sha256",
    "failure_reason",
)
MANIFEST_RUN_STATUSES = {"complete", "failed", "missing"}
BOOTSTRAP_METRICS = (
    "ttfp_p50_ms",
    "ttfp_p95_ms",
    "isr_f",
    "delivery_rate",
    "staleness_p50_ms",
    "staleness_p95_ms",
    "missing_frames",
)
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


class TraceValidationError(ValueError):
    """Raised when a session does not satisfy the schema-v2 contract."""


@dataclass(frozen=True)
class FrameDeadline:
    tick: int
    nominal_tick_time_s: float
    deadline_time_s: float


@dataclass(frozen=True)
class AvailabilitySpan:
    start_tick: int
    end_tick_exclusive: int
    availability_time_s: float


@dataclass(frozen=True)
class ParsedTrace:
    deadlines: dict[int, FrameDeadline]
    spans: tuple[AvailabilitySpan, ...]
    audit: dict[str, Any]


@dataclass(frozen=True)
class ManifestEntry:
    piece_id: str
    seed: str
    system_id: str
    session_dir: Path | None
    session_dir_raw: str
    run_status: str
    melody_input_sha256: str
    failure_reason: str
    line_number: int


PER_SESSION_FIELDS = (
    "session_id",
    "session_dir",
    "condition",
    "mode",
    "clock_domain",
    "continuation_mode",
    "inference_type",
    "input_type",
    "tempo_bpm",
    "ticks_per_beat",
    "schema_version",
    "observation_tick",
    "end_tick_exclusive",
    "window_frames",
    "delivered_frames",
    "missing_frames",
    "on_time_frames",
    "delivery_rate",
    "isr_f",
    "ttfp_ms",
    "staleness_p50_ms",
    "staleness_p95_ms",
)

PER_FRAME_FIELDS = (
    "session_id",
    "session_dir",
    "condition",
    "tick",
    "nominal_tick_time_s",
    "deadline_time_s",
    "availability_time_s",
    "delivered",
    "on_time",
    "staleness_ms",
)

MANIFEST_AUDIT_FIELDS = MANIFEST_REQUIRED_FIELDS + (
    "resolved_session_dir",
    "evaluation_status",
    "evaluation_error",
    "condition",
    "continuation_mode",
    "session_id",
)

SUMMARY_METRICS = (
    "ttfp_ms",
    "isr_f",
    "delivery_rate",
    "missing_frames",
    "staleness_p50_ms",
    "staleness_p95_ms",
)


def _context(path: Path, line_number: int | None, message: str) -> TraceValidationError:
    location = str(path)
    if line_number is not None:
        location += f":{line_number}"
    return TraceValidationError(f"{location}: {message}")


def _required_int(row: dict[str, Any], key: str, path: Path, line_number: int) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise _context(path, line_number, f"{key!r} must be an integer")
    return value


def _required_finite(
    row: dict[str, Any], key: str, path: Path, line_number: int
) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _context(path, line_number, f"{key!r} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise _context(path, line_number, f"{key!r} must be a finite number")
    return number


def _merge_audit_fields(audit_values: dict[str, set[str]], row: dict[str, Any]) -> None:
    for field in AUDIT_FIELDS:
        value = row.get(field)
        if value is not None:
            audit_values[field].add(str(value))


def load_trace(path: Path) -> ParsedTrace:
    """Parse and strictly validate one system_trace.jsonl file."""
    deadlines: dict[int, FrameDeadline] = {}
    spans: list[AvailabilitySpan] = []
    audit_values = {field: set() for field in AUDIT_FIELDS}
    record_count = 0

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise _context(path, None, f"cannot read trace: {exc}") from exc

    for line_number, text in enumerate(lines, start=1):
        if not text.strip():
            continue
        record_count += 1
        try:
            row = json.loads(text)
        except json.JSONDecodeError as exc:
            raise _context(path, line_number, f"invalid JSON: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise _context(path, line_number, "record must be a JSON object")

        schema_version = row.get("schema_version")
        if schema_version == 1:
            raise _context(
                path,
                line_number,
                "schema_version 1 is unsupported; delivery cannot be inferred "
                "from sparse note_on decisions",
            )
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version != TRACE_SCHEMA_VERSION
        ):
            raise _context(
                path,
                line_number,
                f"schema_version must be {TRACE_SCHEMA_VERSION}",
            )

        record_type = row.get("record_type")
        if record_type not in {"frame_deadline", "availability_span"}:
            raise _context(
                path,
                line_number,
                "record_type must be 'frame_deadline' or 'availability_span'",
            )
        _merge_audit_fields(audit_values, row)

        if record_type == "frame_deadline":
            tick = _required_int(row, "tick", path, line_number)
            if tick < 0:
                raise _context(path, line_number, "'tick' must be >= 0")
            if tick in deadlines:
                raise _context(
                    path, line_number, f"duplicate frame_deadline for tick {tick}"
                )
            deadlines[tick] = FrameDeadline(
                tick=tick,
                nominal_tick_time_s=_required_finite(
                    row, "nominal_tick_time_s", path, line_number
                ),
                deadline_time_s=_required_finite(
                    row, "deadline_time_s", path, line_number
                ),
            )
            continue

        start_tick = _required_int(row, "start_tick", path, line_number)
        end_tick = _required_int(row, "end_tick_exclusive", path, line_number)
        if start_tick < 0:
            raise _context(path, line_number, "'start_tick' must be >= 0")
        if end_tick <= start_tick:
            raise _context(
                path,
                line_number,
                "'end_tick_exclusive' must be greater than 'start_tick'",
            )
        spans.append(
            AvailabilitySpan(
                start_tick=start_tick,
                end_tick_exclusive=end_tick,
                availability_time_s=_required_finite(
                    row, "availability_time_s", path, line_number
                ),
            )
        )

    if record_count == 0:
        raise _context(path, None, "trace is empty")
    if not deadlines:
        raise _context(path, None, "trace contains no frame_deadline records")

    audit: dict[str, Any] = {}
    for field, values in audit_values.items():
        if len(values) > 1:
            shown = ", ".join(sorted(values))
            raise _context(path, None, f"inconsistent {field!r} values: {shown}")
        audit[field] = next(iter(values), None)
    return ParsedTrace(deadlines=deadlines, spans=tuple(spans), audit=audit)


def load_session_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise _context(path, None, f"invalid JSON: {exc.msg}") from exc
    except OSError as exc:
        raise _context(path, None, f"cannot read config: {exc}") from exc
    if not isinstance(config, dict):
        raise _context(path, None, "session_config.json must contain a JSON object")
    return config


def _config_observation_tick(config: dict[str, Any], path: Path) -> int:
    value = config.get("prompt_length_ticks")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _context(
            path,
            None,
            "prompt_length_ticks must be a non-negative integer when "
            "--observation-tick is not supplied",
        )
    return value


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (percentile / 100.0) * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _milliseconds(seconds: float) -> float:
    """Convert a derived duration to milliseconds without timestamp tail noise."""
    return round(max(0.0, seconds) * 1000.0, 6)


def evaluate_session(
    session_dir: Path,
    *,
    observation_tick: int | None = None,
    end_tick: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate one session and return its aggregate and per-frame rows."""
    session_dir = Path(session_dir)
    config_path = session_dir / CONFIG_FILENAME
    trace_path = session_dir / TRACE_FILENAME
    if not config_path.is_file():
        raise TraceValidationError(f"{session_dir}: missing {CONFIG_FILENAME}")
    if not trace_path.is_file():
        raise TraceValidationError(f"{session_dir}: missing {TRACE_FILENAME}")

    config = load_session_config(config_path)
    trace = load_trace(trace_path)
    start = (
        _config_observation_tick(config, config_path)
        if observation_tick is None
        else observation_tick
    )
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise TraceValidationError("observation tick must be a non-negative integer")

    stop = max(trace.deadlines) + 1 if end_tick is None else end_tick
    if isinstance(stop, bool) or not isinstance(stop, int):
        raise TraceValidationError("end tick must be an integer")
    if stop <= start:
        raise TraceValidationError(
            f"{session_dir}: end tick {stop} must be greater than observation tick {start}"
        )

    missing_deadlines = [
        tick for tick in range(start, stop) if tick not in trace.deadlines
    ]
    if missing_deadlines:
        preview = ", ".join(str(tick) for tick in missing_deadlines[:8])
        suffix = "..." if len(missing_deadlines) > 8 else ""
        raise TraceValidationError(
            f"{trace_path}: frame_deadline window [{start}, {stop}) is not "
            f"continuous; missing ticks: {preview}{suffix}"
        )

    availability_by_tick: dict[int, float] = {}
    for span in trace.spans:
        overlap_start = max(start, span.start_tick)
        overlap_stop = min(stop, span.end_tick_exclusive)
        for tick in range(overlap_start, overlap_stop):
            previous = availability_by_tick.get(tick)
            if previous is None or span.availability_time_s < previous:
                availability_by_tick[tick] = span.availability_time_s

    session_id = str(config.get("session_id") or session_dir.name)
    per_frame: list[dict[str, Any]] = []
    staleness_values: list[float] = []
    on_time_count = 0
    for tick in range(start, stop):
        deadline = trace.deadlines[tick]
        availability = availability_by_tick.get(tick)
        delivered = availability is not None
        on_time = bool(delivered and availability <= deadline.deadline_time_s)
        staleness_ms = (
            _milliseconds(availability - deadline.deadline_time_s)
            if availability is not None
            else None
        )
        if on_time:
            on_time_count += 1
        if staleness_ms is not None:
            staleness_values.append(staleness_ms)
        per_frame.append(
            {
                "session_id": session_id,
                "session_dir": str(session_dir.resolve()),
                "condition": trace.audit.get("condition"),
                "tick": tick,
                "nominal_tick_time_s": deadline.nominal_tick_time_s,
                "deadline_time_s": deadline.deadline_time_s,
                "availability_time_s": availability,
                "delivered": delivered,
                "on_time": on_time,
                "staleness_ms": staleness_ms,
            }
        )

    window_frames = stop - start
    delivered_frames = len(availability_by_tick)
    earliest_availability = min(availability_by_tick.values(), default=None)
    ttfp_ms = (
        _milliseconds(
            earliest_availability - trace.deadlines[start].nominal_tick_time_s
        )
        if earliest_availability is not None
        else None
    )
    row = {
        "session_id": session_id,
        "session_dir": str(session_dir.resolve()),
        "condition": trace.audit.get("condition"),
        "mode": trace.audit.get("mode"),
        "clock_domain": trace.audit.get("clock_domain"),
        "continuation_mode": config.get("continuation_mode"),
        "inference_type": config.get("inference_type"),
        "input_type": config.get("input_type"),
        "tempo_bpm": config.get("tempo_bpm"),
        "ticks_per_beat": config.get("ticks_per_beat"),
        "schema_version": TRACE_SCHEMA_VERSION,
        "observation_tick": start,
        "end_tick_exclusive": stop,
        "window_frames": window_frames,
        "delivered_frames": delivered_frames,
        "missing_frames": window_frames - delivered_frames,
        "on_time_frames": on_time_count,
        "delivery_rate": delivered_frames / window_frames,
        "isr_f": on_time_count / window_frames,
        "ttfp_ms": ttfp_ms,
        "staleness_p50_ms": _percentile(staleness_values, 50.0),
        "staleness_p95_ms": _percentile(staleness_values, 95.0),
    }
    return row, per_frame


def discover_sessions(
    session_dirs: Iterable[Path], roots: Iterable[Path]
) -> list[Path]:
    discovered: list[Path] = []
    for session_dir in session_dirs:
        path = Path(session_dir)
        if not path.is_dir():
            raise TraceValidationError(f"{path}: session directory does not exist")
        discovered.append(path)
    for root in roots:
        root_path = Path(root)
        if not root_path.is_dir():
            raise TraceValidationError(f"{root_path}: root directory does not exist")
        discovered.extend(path.parent for path in root_path.rglob(TRACE_FILENAME))

    unique: dict[str, Path] = {}
    for path in discovered:
        resolved = path.resolve()
        unique[str(resolved).casefold()] = resolved
    sessions = sorted(unique.values(), key=lambda path: str(path).casefold())
    if not sessions:
        raise TraceValidationError(
            "no sessions found; pass --session-dir or a --root containing "
            f"{TRACE_FILENAME}"
        )
    return sessions


def _manifest_value(
    row: dict[str, str | None], field: str, path: Path, line_number: int
) -> str:
    value = row.get(field)
    if value is None:
        raise _context(path, line_number, f"manifest field {field!r} is missing")
    return value.strip()


def load_manifest(
    path: Path,
    *,
    expected_piece_count: int = EXPECTED_PIECE_COUNT,
    expected_seed_count: int = EXPECTED_SEED_COUNT,
) -> list[ManifestEntry]:
    """Load and validate a complete matched piece x seed x system manifest."""
    path = Path(path)
    try:
        handle = path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise _context(path, None, f"cannot read manifest: {exc}") from exc

    entries: list[ManifestEntry] = []
    try:
        with handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            if fields is None:
                raise _context(path, None, "manifest has no header")
            if len(fields) != len(set(fields)):
                raise _context(path, None, "manifest header contains duplicate columns")
            missing_fields = [
                field for field in MANIFEST_REQUIRED_FIELDS if field not in fields
            ]
            if missing_fields:
                raise _context(
                    path,
                    None,
                    "manifest is missing required columns: "
                    + ", ".join(missing_fields),
                )

            for line_number, row in enumerate(reader, start=2):
                if not any(str(value or "").strip() for value in row.values()):
                    continue
                piece_id = _manifest_value(row, "piece_id", path, line_number)
                seed = _manifest_value(row, "seed", path, line_number)
                system_id = _manifest_value(row, "system_id", path, line_number)
                session_dir_raw = _manifest_value(row, "session_dir", path, line_number)
                run_status = _manifest_value(
                    row, "run_status", path, line_number
                ).lower()
                melody_hash = _manifest_value(
                    row, "melody_input_sha256", path, line_number
                ).lower()
                failure_reason = _manifest_value(
                    row, "failure_reason", path, line_number
                )

                for field, value in (
                    ("piece_id", piece_id),
                    ("seed", seed),
                    ("system_id", system_id),
                ):
                    if not value:
                        raise _context(
                            path, line_number, f"manifest field {field!r} is empty"
                        )
                if run_status not in MANIFEST_RUN_STATUSES:
                    allowed = ", ".join(sorted(MANIFEST_RUN_STATUSES))
                    raise _context(
                        path,
                        line_number,
                        f"run_status must be one of: {allowed}",
                    )
                if not _SHA256_PATTERN.fullmatch(melody_hash):
                    raise _context(
                        path,
                        line_number,
                        "melody_input_sha256 must contain exactly 64 hex digits",
                    )
                if run_status == "complete" and not session_dir_raw:
                    raise _context(
                        path,
                        line_number,
                        "complete manifest rows require session_dir",
                    )
                if run_status in {"failed", "missing"} and not failure_reason:
                    raise _context(
                        path,
                        line_number,
                        f"{run_status} manifest rows require failure_reason",
                    )

                session_dir: Path | None = None
                if session_dir_raw:
                    raw_path = Path(session_dir_raw).expanduser()
                    session_dir = (
                        raw_path if raw_path.is_absolute() else path.parent / raw_path
                    ).resolve()
                entries.append(
                    ManifestEntry(
                        piece_id=piece_id,
                        seed=seed,
                        system_id=system_id,
                        session_dir=session_dir,
                        session_dir_raw=session_dir_raw,
                        run_status=run_status,
                        melody_input_sha256=melody_hash,
                        failure_reason=failure_reason,
                        line_number=line_number,
                    )
                )
    except csv.Error as exc:
        raise _context(path, None, f"invalid manifest CSV: {exc}") from exc

    if not entries:
        raise _context(path, None, "manifest contains no data rows")

    keys: dict[tuple[str, str, str], ManifestEntry] = {}
    session_paths: dict[str, ManifestEntry] = {}
    for entry in entries:
        key = (entry.piece_id, entry.seed, entry.system_id)
        previous = keys.get(key)
        if previous is not None:
            raise _context(
                path,
                entry.line_number,
                "duplicate (piece_id, seed, system_id); first seen at line "
                f"{previous.line_number}",
            )
        keys[key] = entry
        if entry.session_dir is not None:
            session_key = str(entry.session_dir).casefold()
            reused = session_paths.get(session_key)
            if reused is not None:
                raise _context(
                    path,
                    entry.line_number,
                    f"session_dir is reused from manifest line {reused.line_number}",
                )
            session_paths[session_key] = entry

    pieces = sorted({entry.piece_id for entry in entries})
    seeds = sorted({entry.seed for entry in entries})
    systems = sorted({entry.system_id for entry in entries})
    if len(pieces) != expected_piece_count:
        raise _context(
            path,
            None,
            f"manifest must contain exactly {expected_piece_count} pieces; "
            f"found {len(pieces)}",
        )
    if len(seeds) != expected_seed_count:
        raise _context(
            path,
            None,
            f"manifest must contain exactly {expected_seed_count} seeds; "
            f"found {len(seeds)}",
        )
    if len(systems) < 2:
        raise _context(path, None, "manifest must contain at least two systems")

    expected_keys = {
        (piece_id, seed, system_id)
        for piece_id in pieces
        for seed in seeds
        for system_id in systems
    }
    missing_keys = sorted(expected_keys - set(keys))
    if missing_keys:
        preview = ", ".join("/".join(key) for key in missing_keys[:5])
        suffix = "..." if len(missing_keys) > 5 else ""
        raise _context(
            path,
            None,
            f"manifest matched grid is incomplete; missing: {preview}{suffix}",
        )

    hashes_by_piece: dict[str, set[str]] = {}
    for entry in entries:
        hashes_by_piece.setdefault(entry.piece_id, set()).add(entry.melody_input_sha256)
    mismatched_pieces = sorted(
        piece_id for piece_id, hashes in hashes_by_piece.items() if len(hashes) != 1
    )
    if mismatched_pieces:
        raise _context(
            path,
            None,
            "melody_input_sha256 differs within pieces: "
            + ", ".join(mismatched_pieces[:8]),
        )

    return sorted(
        entries, key=lambda entry: (entry.piece_id, entry.seed, entry.system_id)
    )


def _descriptive(values: Sequence[float | int | None]) -> dict[str, Any]:
    numeric = [float(value) for value in values if value is not None]
    return {
        "count": len(numeric),
        "missing_count": len(values) - len(numeric),
        "mean": statistics.fmean(numeric) if numeric else None,
        "min": min(numeric) if numeric else None,
        "max": max(numeric) if numeric else None,
        "p50": _percentile(numeric, 50.0),
        "p95": _percentile(numeric, 95.0),
    }


def _metric_summaries(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        metric: _descriptive([row.get(metric) for row in rows])
        for metric in SUMMARY_METRICS
    }


def _group_value(value: Any) -> str | None:
    return None if value is None else str(value)


def _group_key(condition: str | None, continuation_mode: str | None) -> str:
    condition_label = condition if condition is not None else "<missing>"
    mode_label = continuation_mode if continuation_mode is not None else "<missing>"
    return f"condition={condition_label}__continuation_mode={mode_label}"


def _group_identity(row: dict[str, Any]) -> tuple[str | None, str | None]:
    return (
        _group_value(row.get("condition")),
        _group_value(row.get("continuation_mode")),
    )


def _table_percentile(values: Sequence[float], percentile: float) -> float | None:
    value = _percentile(values, percentile)
    return round(value, 6) if value is not None else None


def _table_metrics(
    session_rows: Sequence[dict[str, Any]],
    frame_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    ttfp_values = [
        float(row["ttfp_ms"]) for row in session_rows if row.get("ttfp_ms") is not None
    ]
    delivered_frames = sum(bool(row.get("delivered")) for row in frame_rows)
    on_time_frames = sum(bool(row.get("on_time")) for row in frame_rows)
    window_frames = len(frame_rows)
    missing_frames = window_frames - delivered_frames
    staleness_values = [
        float(row["staleness_ms"])
        for row in frame_rows
        if row.get("staleness_ms") is not None
    ]
    if len(staleness_values) != delivered_frames:
        raise TraceValidationError(
            "delivered per-frame rows must each contain a finite staleness_ms"
        )

    return {
        "ttfp_p50_ms": _table_percentile(ttfp_values, 50.0),
        "ttfp_p95_ms": _table_percentile(ttfp_values, 95.0),
        "ttfp_available_sessions": len(ttfp_values),
        "ttfp_missing_sessions": len(session_rows) - len(ttfp_values),
        "isr_f": on_time_frames / window_frames if window_frames else None,
        "delivery_rate": (delivered_frames / window_frames if window_frames else None),
        "staleness_p50_ms": _table_percentile(staleness_values, 50.0),
        "staleness_p95_ms": _table_percentile(staleness_values, 95.0),
        "window_frames": window_frames,
        "delivered_frames": delivered_frames,
        "on_time_frames": on_time_frames,
        "missing_frames": missing_frames,
    }


def build_summary(
    rows: Sequence[dict[str, Any]], frame_rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    grouped_rows: dict[tuple[str | None, str | None], list[dict[str, Any]]] = {}
    session_groups: dict[str, tuple[str | None, str | None]] = {}
    for row in rows:
        identity = _group_identity(row)
        grouped_rows.setdefault(identity, []).append(row)
        session_dir = str(row.get("session_dir") or "")
        if not session_dir:
            raise TraceValidationError(
                "per-session rows must contain session_dir for frame grouping"
            )
        previous = session_groups.get(session_dir)
        if previous is not None and previous != identity:
            raise TraceValidationError(
                f"session_dir {session_dir!r} belongs to multiple summary groups"
            )
        session_groups[session_dir] = identity

    grouped_frames: dict[tuple[str | None, str | None], list[dict[str, Any]]] = {
        identity: [] for identity in grouped_rows
    }
    for frame in frame_rows:
        session_dir = str(frame.get("session_dir") or "")
        identity = session_groups.get(session_dir)
        if identity is None:
            raise TraceValidationError(
                f"per-frame row references unknown session_dir {session_dir!r}"
            )
        grouped_frames[identity].append(frame)

    groups: dict[str, Any] = {}
    for (condition, continuation_mode), group_rows in sorted(
        grouped_rows.items(), key=lambda item: _group_key(*item[0])
    ):
        groups[_group_key(condition, continuation_mode)] = {
            "condition": condition,
            "continuation_mode": continuation_mode,
            "session_count": len(group_rows),
            "metrics_scope": "per_session_descriptive",
            "metrics": _metric_summaries(group_rows),
            "table_metrics": _table_metrics(
                group_rows, grouped_frames[(condition, continuation_mode)]
            ),
        }

    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "metric_contract": "matched_system_metrics_v2",
        "statistics": "descriptive_only",
        "bootstrap_confidence_intervals": {
            "computed": False,
            "reason": "95% bootstrap confidence intervals are not implemented",
        },
        "group_by": ["condition", "continuation_mode"],
        "overall": {
            "purpose": "audit_only",
            "session_count": len(rows),
            "metrics": _metric_summaries(rows),
        },
        "groups": groups,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(
    path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _count_values(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _difference(first: Any, second: Any) -> float | None:
    if first is None or second is None:
        return None
    return round(float(first) - float(second), 12)


def _ci_payload(
    estimate: float | None,
    replicate_values: Sequence[float],
    requested_replicates: int,
) -> dict[str, Any]:
    return {
        "estimate": estimate,
        "ci95_low": _table_percentile(replicate_values, 2.5),
        "ci95_high": _table_percentile(replicate_values, 97.5),
        "valid_replicates": len(replicate_values),
        "requested_replicates": requested_replicates,
    }


def _piece_cluster_bootstrap(
    *,
    pieces: Sequence[str],
    systems: Sequence[str],
    cluster_sessions: dict[tuple[str, str], list[dict[str, Any]]],
    cluster_frames: dict[tuple[str, str], list[dict[str, Any]]],
    point_metrics: dict[str, dict[str, Any]],
    replicates: int,
    seed: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    system_replicates: dict[str, dict[str, list[float]]] = {
        system_id: {metric: [] for metric in BOOTSTRAP_METRICS} for system_id in systems
    }
    pair_replicates: dict[tuple[str, str], dict[str, list[float]]] = {
        pair: {metric: [] for metric in BOOTSTRAP_METRICS}
        for pair in combinations(systems, 2)
    }

    for _ in range(replicates):
        draw = [pieces[rng.randrange(len(pieces))] for _ in pieces]
        replicate_metrics: dict[str, dict[str, Any]] = {}
        for system_id in systems:
            sampled_sessions: list[dict[str, Any]] = []
            sampled_frames: list[dict[str, Any]] = []
            for piece_id in draw:
                sampled_sessions.extend(cluster_sessions[(system_id, piece_id)])
                sampled_frames.extend(cluster_frames[(system_id, piece_id)])
            metrics = _table_metrics(sampled_sessions, sampled_frames)
            replicate_metrics[system_id] = metrics
            for metric in BOOTSTRAP_METRICS:
                value = metrics.get(metric)
                if value is not None:
                    system_replicates[system_id][metric].append(float(value))

        for first, second in combinations(systems, 2):
            for metric in BOOTSTRAP_METRICS:
                value = _difference(
                    replicate_metrics[first].get(metric),
                    replicate_metrics[second].get(metric),
                )
                if value is not None:
                    pair_replicates[(first, second)][metric].append(value)

    system_ci = {
        system_id: {
            metric: _ci_payload(
                point_metrics[system_id].get(metric),
                system_replicates[system_id][metric],
                replicates,
            )
            for metric in BOOTSTRAP_METRICS
        }
        for system_id in systems
    }
    paired: dict[str, Any] = {}
    for first, second in combinations(systems, 2):
        key = f"{first}__minus__{second}"
        paired[key] = {
            "minuend_system_id": first,
            "subtrahend_system_id": second,
            "metrics": {
                metric: _ci_payload(
                    _difference(
                        point_metrics[first].get(metric),
                        point_metrics[second].get(metric),
                    ),
                    pair_replicates[(first, second)][metric],
                    replicates,
                )
                for metric in BOOTSTRAP_METRICS
            },
        }
    return system_ci, paired


def evaluate_manifest(
    manifest_path: Path,
    output_dir: Path,
    *,
    observation_tick: int,
    end_tick: int,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    write_per_frame: bool = False,
    expected_piece_count: int = EXPECTED_PIECE_COUNT,
    expected_seed_count: int = EXPECTED_SEED_COUNT,
) -> dict[str, Any]:
    """Evaluate a fixed matched grid and optionally compute formal cluster CIs."""
    if (
        isinstance(observation_tick, bool)
        or not isinstance(observation_tick, int)
        or observation_tick < 0
    ):
        raise TraceValidationError("observation tick must be a non-negative integer")
    if (
        isinstance(end_tick, bool)
        or not isinstance(end_tick, int)
        or end_tick <= observation_tick
    ):
        raise TraceValidationError(
            "end tick must be an integer greater than observation tick"
        )
    if (
        isinstance(bootstrap_replicates, bool)
        or not isinstance(bootstrap_replicates, int)
        or bootstrap_replicates <= 0
    ):
        raise TraceValidationError("bootstrap replicates must be a positive integer")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
        raise TraceValidationError("bootstrap seed must be an integer")

    manifest_path = Path(manifest_path).resolve()
    entries = load_manifest(
        manifest_path,
        expected_piece_count=expected_piece_count,
        expected_seed_count=expected_seed_count,
    )
    pieces = sorted({entry.piece_id for entry in entries})
    seeds = sorted({entry.seed for entry in entries})
    systems = sorted({entry.system_id for entry in entries})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    system_sessions: dict[str, list[dict[str, Any]]] = {
        system_id: [] for system_id in systems
    }
    system_frames: dict[str, list[dict[str, Any]]] = {
        system_id: [] for system_id in systems
    }
    cluster_sessions: dict[tuple[str, str], list[dict[str, Any]]] = {
        (system_id, piece_id): [] for system_id in systems for piece_id in pieces
    }
    cluster_frames: dict[tuple[str, str], list[dict[str, Any]]] = {
        (system_id, piece_id): [] for system_id in systems for piece_id in pieces
    }

    for entry in entries:
        audit = {
            "piece_id": entry.piece_id,
            "seed": entry.seed,
            "system_id": entry.system_id,
            "session_dir": entry.session_dir_raw,
            "run_status": entry.run_status,
            "melody_input_sha256": entry.melody_input_sha256,
            "failure_reason": entry.failure_reason,
            "resolved_session_dir": (
                str(entry.session_dir) if entry.session_dir is not None else ""
            ),
            "evaluation_status": entry.run_status,
            "evaluation_error": "",
            "condition": "",
            "continuation_mode": "",
            "session_id": "",
        }
        if entry.run_status == "complete":
            assert entry.session_dir is not None
            try:
                aggregate, frames = evaluate_session(
                    entry.session_dir,
                    observation_tick=observation_tick,
                    end_tick=end_tick,
                )
            except TraceValidationError as exc:
                audit["evaluation_status"] = "invalid_complete"
                audit["evaluation_error"] = str(exc)
            else:
                audit["evaluation_status"] = "evaluated"
                audit["condition"] = aggregate.get("condition") or ""
                audit["continuation_mode"] = aggregate.get("continuation_mode") or ""
                audit["session_id"] = aggregate.get("session_id") or ""
                aggregate_rows.append(aggregate)
                frame_rows.extend(frames)
                system_sessions[entry.system_id].append(aggregate)
                system_frames[entry.system_id].extend(frames)
                cluster_sessions[(entry.system_id, entry.piece_id)].append(aggregate)
                cluster_frames[(entry.system_id, entry.piece_id)].extend(frames)
        audit_rows.append(audit)

    for system_id in systems:
        identities = {_group_identity(row) for row in system_sessions[system_id]}
        if len(identities) > 1:
            shown = ", ".join(
                _group_key(*identity)
                for identity in sorted(
                    identities, key=lambda identity: _group_key(*identity)
                )
            )
            raise TraceValidationError(
                f"system_id {system_id!r} maps to inconsistent conditions: {shown}"
            )

    _write_csv(output_dir / "manifest_audit.csv", audit_rows, MANIFEST_AUDIT_FIELDS)
    _write_csv(output_dir / "per_session.csv", aggregate_rows, PER_SESSION_FIELDS)
    _write_json(
        output_dir / "per_session.json",
        {
            "schema_version": TRACE_SCHEMA_VERSION,
            "metric_contract": "matched_system_metrics_v2",
            "sessions": aggregate_rows,
        },
    )
    if write_per_frame:
        _write_csv(output_dir / "per_frame.csv", frame_rows, PER_FRAME_FIELDS)

    manifest_status_counts = _count_values(entry.run_status for entry in entries)
    evaluation_status_counts = _count_values(
        str(row["evaluation_status"]) for row in audit_rows
    )
    blockers: list[str] = []
    noncomplete_count = sum(
        count
        for status, count in manifest_status_counts.items()
        if status != "complete"
    )
    invalid_complete_count = evaluation_status_counts.get("invalid_complete", 0)
    if noncomplete_count:
        blockers.append(f"{noncomplete_count} manifest rows are failed or missing")
    if invalid_complete_count:
        blockers.append(
            f"{invalid_complete_count} complete rows have invalid schema-v2 sessions"
        )
    primary_ci_eligible = not blockers

    point_metrics = {
        system_id: _table_metrics(system_sessions[system_id], system_frames[system_id])
        for system_id in systems
    }
    groups: dict[str, Any] = {}
    for system_id in systems:
        identities = {_group_identity(row) for row in system_sessions[system_id]}
        condition, continuation_mode = (
            next(iter(identities)) if identities else (None, None)
        )
        system_entries = [entry for entry in entries if entry.system_id == system_id]
        system_audit = [row for row in audit_rows if row["system_id"] == system_id]
        groups[system_id] = {
            "system_id": system_id,
            "condition": condition,
            "continuation_mode": continuation_mode,
            "planned_session_count": len(system_entries),
            "evaluated_session_count": len(system_sessions[system_id]),
            "manifest_status_counts": _count_values(
                entry.run_status for entry in system_entries
            ),
            "evaluation_status_counts": _count_values(
                str(row["evaluation_status"]) for row in system_audit
            ),
            "point_estimate_scope": (
                "all_planned_sessions"
                if primary_ci_eligible
                else "evaluable_complete_sessions_only"
            ),
            "metrics_scope": "per_session_descriptive",
            "metrics": _metric_summaries(system_sessions[system_id]),
            "table_metrics": point_metrics[system_id],
        }

    paired: dict[str, Any] = {}
    if primary_ci_eligible:
        system_ci, paired = _piece_cluster_bootstrap(
            pieces=pieces,
            systems=systems,
            cluster_sessions=cluster_sessions,
            cluster_frames=cluster_frames,
            point_metrics=point_metrics,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        )
        for system_id in systems:
            groups[system_id]["bootstrap_ci"] = system_ci[system_id]
    else:
        reason = "; ".join(blockers)
        for system_id in systems:
            groups[system_id]["bootstrap_ci"] = {
                "computed": False,
                "reason": reason,
            }

    summary = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "metric_contract": "matched_system_piece_cluster_bootstrap_v1",
        "statistics": (
            "descriptive_with_piece_cluster_bootstrap"
            if primary_ci_eligible
            else "descriptive_only_primary_ci_blocked"
        ),
        "group_by": ["system_id"],
        "manifest": {
            "path": str(manifest_path),
            "piece_count": len(pieces),
            "seed_count": len(seeds),
            "system_count": len(systems),
            "planned_trial_count": len(entries),
            "manifest_status_counts": manifest_status_counts,
            "evaluation_status_counts": evaluation_status_counts,
            "primary_ci_eligible": primary_ci_eligible,
            "primary_ci_blockers": blockers,
        },
        "bootstrap_confidence_intervals": {
            "computed": primary_ci_eligible,
            "method": "matched_piece_cluster_percentile",
            "confidence_level": 0.95,
            "cluster_unit": "piece_id",
            "piece_draws_are_shared_across_systems": True,
            "all_seeds_and_frames_retained_per_piece": True,
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "reason": None if primary_ci_eligible else "; ".join(blockers),
        },
        "overall": {
            "purpose": "audit_only",
            "planned_session_count": len(entries),
            "evaluated_session_count": len(aggregate_rows),
            "metrics": _metric_summaries(aggregate_rows),
        },
        "groups": groups,
        "paired_system_differences": paired,
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def evaluate_sessions(
    sessions: Sequence[Path],
    output_dir: Path,
    *,
    observation_tick: int | None = None,
    end_tick: int | None = None,
    write_per_frame: bool = False,
) -> list[dict[str, Any]]:
    aggregate_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    for session in sessions:
        aggregate, frames = evaluate_session(
            session, observation_tick=observation_tick, end_tick=end_tick
        )
        aggregate_rows.append(aggregate)
        frame_rows.extend(frames)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "per_session.csv", aggregate_rows, PER_SESSION_FIELDS)
    _write_json(
        output_dir / "per_session.json",
        {
            "schema_version": TRACE_SCHEMA_VERSION,
            "metric_contract": "matched_system_metrics_v2",
            "sessions": aggregate_rows,
        },
    )
    if write_per_frame:
        _write_csv(output_dir / "per_frame.csv", frame_rows, PER_FRAME_FIELDS)
    _write_json(output_dir / "summary.json", build_summary(aggregate_rows, frame_rows))
    return aggregate_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate matched realtime system metrics from schema-v2 traces."
    )
    parser.add_argument(
        "--session-dir",
        action="append",
        default=[],
        type=Path,
        help="Session directory containing session_config.json and system_trace.jsonl; repeatable.",
    )
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        type=Path,
        help="Recursively discover system_trace.jsonl files below this root; repeatable.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Formal matched-grid CSV manifest; cannot be combined with --root or --session-dir.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--observation-tick", type=int)
    parser.add_argument("--end-tick", type=int)
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
        help=f"Piece-cluster bootstrap replicates (default: {DEFAULT_BOOTSTRAP_REPLICATES}).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
        help=f"Deterministic piece-draw seed (default: {DEFAULT_BOOTSTRAP_SEED}).",
    )
    parser.add_argument(
        "--per-frame",
        action="store_true",
        help="Also write per_frame.csv.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.manifest is not None and (args.session_dir or args.root):
        parser.error("--manifest cannot be combined with --root or --session-dir")
    if args.manifest is not None and (
        args.observation_tick is None or args.end_tick is None
    ):
        parser.error("--manifest requires explicit --observation-tick and --end-tick")
    try:
        if args.manifest is not None:
            summary = evaluate_manifest(
                args.manifest,
                args.output_dir,
                observation_tick=args.observation_tick,
                end_tick=args.end_tick,
                bootstrap_replicates=args.bootstrap_replicates,
                bootstrap_seed=args.bootstrap_seed,
                write_per_frame=args.per_frame,
            )
            if not summary["bootstrap_confidence_intervals"]["computed"]:
                reason = summary["bootstrap_confidence_intervals"]["reason"]
                print(
                    f"error: manifest audit written, but primary CI is blocked: {reason}",
                    file=sys.stderr,
                )
                return 2
            print(
                f"Evaluated matched manifest with {summary['manifest']['piece_count']} "
                f"pieces; results written to {args.output_dir}"
            )
            return 0

        sessions = discover_sessions(args.session_dir, args.root)
        evaluate_sessions(
            sessions,
            args.output_dir,
            observation_tick=args.observation_tick,
            end_tick=args.end_tick,
            write_per_frame=args.per_frame,
        )
    except TraceValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Evaluated {len(sessions)} session(s); results written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
