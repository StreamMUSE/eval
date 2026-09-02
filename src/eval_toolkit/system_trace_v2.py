"""Matched realtime system metrics for StreamMUSE system-trace schema v2.

This evaluator intentionally uses frame deadlines and decision-availability
spans only. MIDI events and sparse note-on rows are not delivery evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TRACE_FILENAME = "system_trace.jsonl"
CONFIG_FILENAME = "session_config.json"
TRACE_SCHEMA_VERSION = 2
AUDIT_FIELDS = ("condition", "mode", "clock_domain")


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
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--observation-tick", type=int)
    parser.add_argument("--end-tick", type=int)
    parser.add_argument(
        "--per-frame",
        action="store_true",
        help="Also write per_frame.csv.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
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
