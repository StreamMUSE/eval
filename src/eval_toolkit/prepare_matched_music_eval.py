"""Prepare matched post-join MIDI pairs for music-quality evaluation.

Realtime session exports are cropped to the fixed [8, 32) beat window at
120 BPM. Future offline rows may point at files that are already expressed in
that post-join window. This module prepares files only; it never computes or
claims realtime system metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mido
import pretty_midi

BPM = 120.0
BEATS_PER_SECOND = BPM / 60.0
WINDOW_START_BEAT = 8
WINDOW_END_BEAT = 32
WINDOW_START_S = WINDOW_START_BEAT / BEATS_PER_SECOND
WINDOW_END_S = WINDOW_END_BEAT / BEATS_PER_SECOND
WINDOW_DURATION_S = WINDOW_END_S - WINDOW_START_S
TIME_EPSILON_S = 1e-6
DEFAULT_EXPECTED_PIECE_COUNT = 40
DEFAULT_EXPECTED_SEEDS = ("0", "1", "2")
RUN_STATUSES = {"complete", "failed", "missing"}
HASH_KIND = "melody_midi_file_sha256"

STATUS_ALIASES = ("run_status", "status")
HASH_ALIASES = ("melody_input_sha256", "hash")
REALTIME_REQUIRED_FIELDS = (
    "piece_id",
    "seed",
    "system_id",
    "session_dir",
    "failure_reason",
)
OFFLINE_REQUIRED_FIELDS = (
    "piece_id",
    "seed",
    "system_id",
    "postjoin_generated_midi",
    "postjoin_gt_midi",
    "failure_reason",
)

AUDIT_FIELDS = (
    "source_kind",
    "system_scope",
    "piece_id",
    "seed",
    "system_id",
    "source_status",
    "source_generated_midi",
    "source_gt_midi",
    "cohort_full_gt_midi",
    "source_melody_midi",
    "all_trials_generated_midi",
    "all_trials_metric_gt_midi",
    "valid_only_generated_midi",
    "valid_only_metric_gt_midi",
    "melody_input_sha256",
    "melody_hash_kind",
    "source_gt_sha256",
    "cohort_full_gt_sha256",
    "cohort_postjoin_melody_note_count",
    "cohort_postjoin_melody_sha256",
    "cohort_source_npz",
    "cohort_source_npz_sha256",
    "trial_source_npz_sha256",
    "offline_gt_roundtrip_exact",
    "generated_sha256",
    "metric_gt_sha256",
    "generated_mel_note_count",
    "generated_acc_note_count",
    "source_gt_mel_note_count",
    "source_gt_acc_note_count",
    "metric_gt_acc_note_count",
    "valid_output",
    "window_start_beat",
    "window_end_beat_exclusive",
    "window_start_s",
    "window_end_s",
    "preparation_status",
    "reason",
)
PUBLISHED_PATH_FIELDS = (
    "all_trials_generated_midi",
    "all_trials_metric_gt_midi",
    "valid_only_generated_midi",
    "valid_only_metric_gt_midi",
)


class PreparationError(ValueError):
    """Raised when manifests or MIDI inputs violate the preparation contract."""


class PreparationBlockedError(PreparationError):
    """Raised after audit output is written for an incomplete/invalid batch."""


@dataclass(frozen=True)
class CohortPiece:
    piece_id: str
    melody_midi: Path
    gt_midi: Path
    melody_midi_sha256: str
    gt_midi_sha256: str
    postjoin_gt: WindowMidi
    postjoin_melody_sha256: str
    source_npz: Path | None
    source_npz_sha256: str


@dataclass(frozen=True)
class Trial:
    source_kind: str
    system_scope: str
    piece_id: str
    seed: str
    system_id: str
    run_status: str
    failure_reason: str
    manifest_path: Path
    row_number: int
    session_dir: Path | None = None
    generated_midi: Path | None = None
    gt_midi: Path | None = None
    melody_input_sha256: str | None = None
    source_npz_sha256: str | None = None


@dataclass(frozen=True)
class NoteValue:
    pitch: int
    velocity: int
    start: float
    end: float


@dataclass(frozen=True)
class WindowMidi:
    melody: tuple[NoteValue, ...]
    accompaniment: tuple[NoteValue, ...]
    melody_program: int = 0
    accompaniment_program: int = 0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _existing_file_sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    try:
        return file_sha256(path)
    except OSError:
        return None


def _resolve_path(raw: str, manifest_path: Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationError(f"{path}: cannot read JSON: {exc}") from exc


def _require_string(row: Mapping[str, Any], field: str, context: str) -> str:
    value = row.get(field)
    if not isinstance(value, (str, int)):
        raise PreparationError(f"{context}: field {field!r} is missing or invalid")
    text = str(value).strip()
    if not text:
        raise PreparationError(f"{context}: field {field!r} is empty")
    return text


def _optional_string(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field, "")
    if value is None:
        return ""
    if not isinstance(value, (str, int)):
        return str(value).strip()
    return str(value).strip()


def _select_alias(
    row: Mapping[str, Any], aliases: Sequence[str], context: str, *, required: bool
) -> str:
    present = [name for name in aliases if name in row]
    if len(present) > 1:
        raise PreparationError(
            f"{context}: ambiguous aliases are both present: {', '.join(present)}"
        )
    if not present:
        if required:
            raise PreparationError(
                f"{context}: one of {', '.join(aliases)} is required"
            )
        return ""
    value = _optional_string(row, present[0])
    if required and not value:
        raise PreparationError(f"{context}: field {present[0]!r} is empty")
    return value


def _load_rows(path: Path) -> list[tuple[int, dict[str, Any]]]:
    path = path.resolve()
    if path.suffix.lower() == ".json":
        data = _read_json(path)
        if isinstance(data, dict):
            data = data.get("trials", data.get("rows"))
        if not isinstance(data, list):
            raise PreparationError(
                f"{path}: JSON manifest must be a list or contain trials/rows"
            )
        rows: list[tuple[int, dict[str, Any]]] = []
        for index, row in enumerate(data, start=1):
            if not isinstance(row, dict):
                raise PreparationError(f"{path}: JSON row {index} is not an object")
            rows.append((index, row))
        return rows

    if path.suffix.lower() != ".csv":
        raise PreparationError(f"{path}: manifest must be CSV or JSON")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise PreparationError(f"{path}: CSV manifest has no header")
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise PreparationError(f"{path}: CSV header has duplicate columns")
            return [
                (line_number, dict(row))
                for line_number, row in enumerate(reader, start=2)
                if any(str(value or "").strip() for value in row.values())
            ]
    except (OSError, csv.Error) as exc:
        raise PreparationError(f"{path}: cannot read CSV: {exc}") from exc


def load_cohort_manifest(path: Path) -> dict[str, CohortPiece]:
    path = path.resolve()
    data = _read_json(path)
    if not isinstance(data, dict) or not isinstance(data.get("samples"), list):
        raise PreparationError(f"{path}: cohort manifest requires top-level samples")

    pieces: dict[str, CohortPiece] = {}
    for index, raw in enumerate(data["samples"], start=1):
        context = f"{path}: sample {index}"
        if not isinstance(raw, dict):
            raise PreparationError(f"{context}: sample must be an object")
        piece_id = _require_string(raw, "piece_id", context)
        if piece_id in pieces:
            raise PreparationError(f"{context}: duplicate piece_id {piece_id!r}")
        melody = _resolve_path(_require_string(raw, "melody_midi", context), path)
        gt = _resolve_path(_require_string(raw, "gt_midi", context), path)
        declared_hashes = {
            "melody_midi_sha256": _require_string(
                raw, "melody_midi_sha256", context
            ).lower(),
            "gt_midi_sha256": _require_string(raw, "gt_midi_sha256", context).lower(),
            "source_npz_sha256": _require_string(
                raw, "source_npz_sha256", context
            ).lower(),
        }
        for field, declared_hash in declared_hashes.items():
            if not re.fullmatch(r"[0-9a-f]{64}", declared_hash):
                raise PreparationError(
                    f"{context}: {field} must be 64 lowercase hex digits"
                )
        for label, midi_path in (("melody_midi", melody), ("gt_midi", gt)):
            if not midi_path.is_file():
                raise PreparationError(f"{context}: {label} not found: {midi_path}")
        actual_hash = file_sha256(melody)
        if actual_hash != declared_hashes["melody_midi_sha256"]:
            raise PreparationError(
                f"{context}: melody MIDI file hash mismatch; declared "
                f"{declared_hashes['melody_midi_sha256']}, actual {actual_hash}"
            )
        actual_gt_hash = file_sha256(gt)
        if actual_gt_hash != declared_hashes["gt_midi_sha256"]:
            raise PreparationError(
                f"{context}: GT MIDI file hash mismatch; declared "
                f"{declared_hashes['gt_midi_sha256']}, actual {actual_gt_hash}"
            )
        source_npz_raw = _optional_string(raw, "source_npz")
        source_npz = _resolve_path(source_npz_raw, path) if source_npz_raw else None
        if source_npz is not None:
            if not source_npz.is_file():
                raise PreparationError(f"{context}: source_npz not found: {source_npz}")
            actual_source_npz_hash = file_sha256(source_npz)
            if actual_source_npz_hash != declared_hashes["source_npz_sha256"]:
                raise PreparationError(
                    f"{context}: source NPZ file hash mismatch; declared "
                    f"{declared_hashes['source_npz_sha256']}, actual "
                    f"{actual_source_npz_hash}"
                )
        postjoin_gt, postjoin_melody_sha256 = _validate_cohort_melody_pair(melody, gt)
        pieces[piece_id] = CohortPiece(
            piece_id,
            melody,
            gt,
            actual_hash,
            actual_gt_hash,
            postjoin_gt,
            postjoin_melody_sha256,
            source_npz,
            declared_hashes["source_npz_sha256"],
        )

    if not pieces:
        raise PreparationError(f"{path}: cohort contains no samples")
    return pieces


def _parse_status(row: Mapping[str, Any], context: str) -> tuple[str, str]:
    status = _select_alias(row, STATUS_ALIASES, context, required=True).lower()
    if status not in RUN_STATUSES:
        raise PreparationError(
            f"{context}: status must be one of {', '.join(sorted(RUN_STATUSES))}"
        )
    reason = _optional_string(row, "failure_reason")
    if status in {"failed", "missing"} and not reason:
        raise PreparationError(f"{context}: {status} rows require failure_reason")
    return status, reason


def load_realtime_manifest(path: Path) -> list[Trial]:
    path = path.resolve()
    rows = _load_rows(path)
    if not rows:
        raise PreparationError(f"{path}: manifest has no rows")
    trials: list[Trial] = []
    for row_number, row in rows:
        context = f"{path}: row {row_number}"
        for field in REALTIME_REQUIRED_FIELDS[:-1]:
            if field not in row:
                raise PreparationError(f"{context}: missing field {field!r}")
        piece_id = _require_string(row, "piece_id", context)
        seed = _require_string(row, "seed", context)
        system_id = _require_string(row, "system_id", context)
        status, reason = _parse_status(row, context)
        input_hash = _select_alias(row, HASH_ALIASES, context, required=True).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", input_hash):
            raise PreparationError(f"{context}: input hash must be 64 hex digits")
        session_raw = _optional_string(row, "session_dir")
        session_dir = _resolve_path(session_raw, path) if session_raw else None
        if status == "complete" and session_dir is None:
            raise PreparationError(f"{context}: complete row requires session_dir")
        trials.append(
            Trial(
                source_kind="realtime_session",
                system_scope="realtime_system_output",
                piece_id=piece_id,
                seed=seed,
                system_id=system_id,
                run_status=status,
                failure_reason=reason,
                manifest_path=path,
                row_number=row_number,
                session_dir=session_dir,
                melody_input_sha256=input_hash,
            )
        )
    return trials


def load_offline_manifest(path: Path) -> list[Trial]:
    path = path.resolve()
    rows = _load_rows(path)
    if not rows:
        raise PreparationError(f"{path}: manifest has no rows")
    trials: list[Trial] = []
    for row_number, row in rows:
        context = f"{path}: row {row_number}"
        for field in OFFLINE_REQUIRED_FIELDS[:-1]:
            if field not in row:
                raise PreparationError(f"{context}: missing field {field!r}")
        piece_id = _require_string(row, "piece_id", context)
        seed = _require_string(row, "seed", context)
        system_id = _require_string(row, "system_id", context)
        status, reason = _parse_status(row, context)
        generated_raw = _optional_string(row, "postjoin_generated_midi")
        gt_raw = _optional_string(row, "postjoin_gt_midi")
        source_npz_hash = _optional_string(row, "source_npz_sha256").lower()
        if source_npz_hash and not re.fullmatch(r"[0-9a-f]{64}", source_npz_hash):
            raise PreparationError(
                f"{context}: source_npz_sha256 must contain exactly 64 hex digits"
            )
        generated = _resolve_path(generated_raw, path) if generated_raw else None
        gt = _resolve_path(gt_raw, path) if gt_raw else None
        if status == "complete" and (generated is None or gt is None):
            raise PreparationError(
                f"{context}: complete offline row requires both postjoin MIDI paths"
            )
        trials.append(
            Trial(
                source_kind="offline_postjoin",
                system_scope="music_quality_only",
                piece_id=piece_id,
                seed=seed,
                system_id=system_id,
                run_status=status,
                failure_reason=reason,
                manifest_path=path,
                row_number=row_number,
                generated_midi=generated,
                gt_midi=gt,
                source_npz_sha256=source_npz_hash or None,
            )
        )
    return trials


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    if not slug:
        raise PreparationError(f"cannot derive safe output name from {value!r}")
    return slug


def _validate_grid(
    trials: Sequence[Trial],
    cohort: Mapping[str, CohortPiece],
    *,
    expected_piece_count: int,
    expected_seeds: Sequence[str],
) -> list[Trial]:
    if expected_piece_count <= 0:
        raise PreparationError("expected_piece_count must be positive")
    expected_seed_set = set(expected_seeds)
    if not expected_seed_set or len(expected_seed_set) != len(expected_seeds):
        raise PreparationError("expected_seeds must be non-empty and unique")
    if len(cohort) != expected_piece_count:
        raise PreparationError(
            f"cohort must contain exactly {expected_piece_count} pieces; "
            f"found {len(cohort)}"
        )
    if not trials:
        raise PreparationError("at least one realtime or offline manifest is required")

    keys: dict[tuple[str, str, str], Trial] = {}
    system_kinds: dict[str, set[str]] = {}
    for trial in trials:
        if trial.piece_id not in cohort:
            raise PreparationError(
                f"{trial.manifest_path}: row {trial.row_number}: unknown cohort "
                f"piece_id {trial.piece_id!r}"
            )
        if trial.seed not in expected_seed_set:
            raise PreparationError(
                f"{trial.manifest_path}: row {trial.row_number}: unexpected seed "
                f"{trial.seed!r}; expected {sorted(expected_seed_set)}"
            )
        key = (trial.piece_id, trial.seed, trial.system_id)
        if key in keys:
            previous = keys[key]
            raise PreparationError(
                f"duplicate trial key {key}; first seen in "
                f"{previous.manifest_path}: row {previous.row_number}"
            )
        keys[key] = trial
        system_kinds.setdefault(trial.system_id, set()).add(trial.source_kind)

    mixed = sorted(system for system, kinds in system_kinds.items() if len(kinds) > 1)
    if mixed:
        raise PreparationError(
            "system_id cannot mix realtime and offline rows: " + ", ".join(mixed)
        )
    systems = sorted(system_kinds)
    if not systems:
        raise PreparationError("no systems declared by input manifests")
    for label, values in (
        ("piece_id", sorted(cohort)),
        ("seed", list(expected_seeds)),
        ("system_id", systems),
    ):
        slugs: dict[str, str] = {}
        for value in values:
            slug = _slug(value)
            previous = slugs.get(slug)
            if previous is not None and previous != value:
                raise PreparationError(
                    f"{label} output-name collision: {previous!r} and {value!r}"
                )
            slugs[slug] = value

    completed = list(trials)
    for piece_id in sorted(cohort):
        for seed in expected_seeds:
            for system_id in systems:
                key = (piece_id, seed, system_id)
                if key in keys:
                    continue
                source_kind = next(iter(system_kinds[system_id]))
                completed.append(
                    Trial(
                        source_kind=source_kind,
                        system_scope=(
                            "music_quality_only"
                            if source_kind == "offline_postjoin"
                            else "realtime_system_output"
                        ),
                        piece_id=piece_id,
                        seed=seed,
                        system_id=system_id,
                        run_status="missing",
                        failure_reason="matched grid row is absent from manifest",
                        manifest_path=Path("<inferred>"),
                        row_number=0,
                        melody_input_sha256=(
                            cohort[piece_id].melody_midi_sha256
                            if source_kind == "realtime_session"
                            else None
                        ),
                    )
                )
    return sorted(completed, key=lambda row: (row.system_id, row.piece_id, row.seed))


def _track_role(name: str) -> str | None:
    tokens = {token for token in re.split(r"[^a-z0-9]+", name.casefold()) if token}
    melody = bool(tokens & {"melody", "mel"})
    accompaniment = bool(tokens & {"accompaniment", "acc"})
    if melody and accompaniment:
        raise PreparationError(f"ambiguous MIDI track name: {name!r}")
    if melody:
        return "melody"
    if accompaniment:
        return "accompaniment"
    return None


def _load_midi(path: Path) -> pretty_midi.PrettyMIDI:
    if not path.is_file():
        raise PreparationError(f"MIDI file not found: {path}")
    try:
        midi = pretty_midi.PrettyMIDI(str(path))
    except Exception as exc:
        raise PreparationError(f"cannot parse MIDI {path}: {exc}") from exc
    _, tempos = midi.get_tempo_changes()
    if any(not math.isfinite(float(tempo)) for tempo in tempos):
        raise PreparationError(f"MIDI has non-finite tempo: {path}")
    if any(abs(float(tempo) - BPM) > 1e-3 for tempo in tempos):
        raise PreparationError(f"MIDI must use 120 BPM throughout: {path}")
    return midi


def _extract_window(
    path: Path,
    *,
    already_postjoin: bool,
    require_accompaniment_track: bool,
    require_accompaniment_notes: bool,
) -> WindowMidi:
    midi = _load_midi(path)
    role_instruments: dict[str, list[pretty_midi.Instrument]] = {
        "melody": [],
        "accompaniment": [],
    }
    for instrument in midi.instruments:
        role = _track_role(instrument.name)
        if role is not None:
            role_instruments[role].append(instrument)

    if not role_instruments["melody"]:
        raise PreparationError(f"MIDI requires a named Melody track: {path}")
    if require_accompaniment_track and not role_instruments["accompaniment"]:
        raise PreparationError(f"MIDI requires a named Accompaniment track: {path}")

    start_s = 0.0 if already_postjoin else WINDOW_START_S
    end_s = WINDOW_DURATION_S if already_postjoin else WINDOW_END_S
    shift_s = 0.0 if already_postjoin else WINDOW_START_S

    def collect(role: str) -> tuple[NoteValue, ...]:
        notes: list[NoteValue] = []
        for instrument in role_instruments[role]:
            for note in instrument.notes:
                values = (float(note.start), float(note.end))
                if not all(math.isfinite(value) for value in values):
                    raise PreparationError(f"MIDI has non-finite note time: {path}")
                if note.end <= note.start:
                    raise PreparationError(
                        f"MIDI has non-positive note duration: {path}"
                    )
                if already_postjoin and (
                    note.start < -TIME_EPSILON_S
                    or note.end > WINDOW_DURATION_S + TIME_EPSILON_S
                ):
                    raise PreparationError(
                        f"declared postjoin MIDI note lies outside [0, 12]s: {path}"
                    )
                if note.end <= start_s + TIME_EPSILON_S:
                    continue
                if note.start >= end_s - TIME_EPSILON_S:
                    continue
                clipped_start = max(note.start, start_s) - shift_s
                clipped_end = min(note.end, end_s) - shift_s
                clipped_start = max(0.0, clipped_start)
                clipped_end = min(WINDOW_DURATION_S, clipped_end)
                if clipped_end - clipped_start <= TIME_EPSILON_S:
                    continue
                notes.append(
                    NoteValue(
                        pitch=int(note.pitch),
                        velocity=int(note.velocity),
                        start=clipped_start,
                        end=clipped_end,
                    )
                )
        return tuple(sorted(notes, key=lambda n: (n.start, n.pitch, n.end, n.velocity)))

    melody = collect("melody")
    accompaniment = collect("accompaniment")
    if not melody:
        raise PreparationError(f"postjoin Melody track is empty: {path}")
    if require_accompaniment_notes and not accompaniment:
        raise PreparationError(f"postjoin GT Accompaniment track is empty: {path}")

    melody_program = role_instruments["melody"][0].program
    accompaniment_program = (
        role_instruments["accompaniment"][0].program
        if role_instruments["accompaniment"]
        else 0
    )
    return WindowMidi(
        melody=melody,
        accompaniment=accompaniment,
        melody_program=melody_program,
        accompaniment_program=accompaniment_program,
    )


def _canonical_notes(window: WindowMidi) -> tuple[tuple[Any, ...], ...]:
    values: list[tuple[Any, ...]] = []
    for role, notes in (
        ("melody", window.melody),
        ("accompaniment", window.accompaniment),
    ):
        values.extend(
            (
                role,
                note.pitch,
                note.velocity,
                round(note.start, 6),
                round(note.end, 6),
            )
            for note in notes
        )
    return tuple(sorted(values))


def _canonical_note_values(notes: Sequence[NoteValue]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        sorted(
            (
                note.pitch,
                note.velocity,
                round(note.start, 6),
                round(note.end, 6),
            )
            for note in notes
        )
    )


def _canonical_note_geometry(
    notes: Sequence[NoteValue],
) -> tuple[tuple[int, float, float], ...]:
    """Canonical pitch/timing values for velocity-free 4-channel GT matching."""
    return tuple(
        sorted(
            (
                note.pitch,
                round(note.start, 6),
                round(note.end, 6),
            )
            for note in notes
        )
    )


def _canonical_note_sha256(notes: Sequence[NoteValue]) -> str:
    payload = json.dumps(
        _canonical_note_values(notes),
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_cohort_melody_pair(
    melody_path: Path, gt_path: Path
) -> tuple[WindowMidi, str]:
    melody_midi = _load_midi(melody_path)
    melody_tracks: list[pretty_midi.Instrument] = []
    for instrument in melody_midi.instruments:
        role = _track_role(instrument.name)
        if role == "melody":
            melody_tracks.append(instrument)
            continue
        if instrument.notes:
            label = instrument.name or "<unnamed>"
            raise PreparationError(
                "cohort melody_midi contains a non-Melody music track "
                f"{label!r}: {melody_path}"
            )
    if len(melody_tracks) != 1:
        raise PreparationError(
            f"cohort melody_midi requires exactly one named Melody track: {melody_path}"
        )
    if not melody_tracks[0].notes:
        raise PreparationError(f"cohort Melody track is empty: {melody_path}")

    postjoin_melody = _extract_window(
        melody_path,
        already_postjoin=False,
        require_accompaniment_track=False,
        require_accompaniment_notes=False,
    )
    postjoin_gt = _extract_window(
        gt_path,
        already_postjoin=False,
        require_accompaniment_track=True,
        require_accompaniment_notes=True,
    )
    if _canonical_note_values(postjoin_melody.melody) != _canonical_note_values(
        postjoin_gt.melody
    ):
        raise PreparationError(
            "cohort melody_midi postjoin Melody does not match gt_midi Melody: "
            f"{melody_path}"
        )
    return postjoin_gt, _canonical_note_sha256(postjoin_melody.melody)


def _seconds_to_ticks(value: float, ticks_per_beat: int) -> int:
    return round(value * BEATS_PER_SECOND * ticks_per_beat)


def _build_note_track(
    name: str,
    notes: Sequence[NoteValue],
    *,
    program: int,
    channel: int,
    ticks_per_beat: int,
) -> mido.MidiTrack:
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name=name, time=0))
    track.append(
        mido.Message(
            "program_change",
            program=max(0, min(127, int(program))),
            channel=channel,
            time=0,
        )
    )
    events: list[tuple[int, int, int, int]] = []
    for note in notes:
        start_tick = _seconds_to_ticks(note.start, ticks_per_beat)
        end_tick = _seconds_to_ticks(note.end, ticks_per_beat)
        end_tick = max(start_tick + 1, end_tick)
        events.append((start_tick, 1, note.pitch, note.velocity))
        events.append((end_tick, 0, note.pitch, 0))
    events.sort(key=lambda event: (event[0], event[1], event[2]))
    previous_tick = 0
    for tick, is_on, pitch, velocity in events:
        delta = tick - previous_tick
        previous_tick = tick
        track.append(
            mido.Message(
                "note_on" if is_on else "note_off",
                note=pitch,
                velocity=velocity,
                channel=channel,
                time=delta,
            )
        )
    track.append(mido.MetaMessage("end_of_track", time=0))
    return track


def write_window_midi(window: WindowMidi, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ticks_per_beat = 480
    midi = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)
    metadata = mido.MidiTrack()
    metadata.append(mido.MetaMessage("track_name", name="Metadata", time=0))
    metadata.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(BPM), time=0))
    metadata.append(
        mido.MetaMessage(
            "time_signature", numerator=4, denominator=4, clocks_per_click=24, time=0
        )
    )
    metadata.append(mido.MetaMessage("end_of_track", time=0))
    midi.tracks.append(metadata)
    midi.tracks.append(
        _build_note_track(
            "Melody",
            window.melody,
            program=window.melody_program,
            channel=0,
            ticks_per_beat=ticks_per_beat,
        )
    )
    midi.tracks.append(
        _build_note_track(
            "Accompaniment",
            window.accompaniment,
            program=window.accompaniment_program,
            channel=1,
            ticks_per_beat=ticks_per_beat,
        )
    )
    midi.save(path)


def write_metric_groundtruth(window: WindowMidi, path: Path) -> None:
    """Write accompaniment-only GT in the legacy metric evaluator contract."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ticks_per_beat = 480
    midi = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)
    metadata = mido.MidiTrack()
    metadata.append(mido.MetaMessage("track_name", name="Metadata", time=0))
    metadata.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(BPM), time=0))
    metadata.append(
        mido.MetaMessage(
            "time_signature", numerator=4, denominator=4, clocks_per_click=24, time=0
        )
    )
    metadata.append(mido.MetaMessage("end_of_track", time=0))
    midi.tracks.append(metadata)
    midi.tracks.append(
        _build_note_track(
            "Piano",
            window.accompaniment,
            program=window.accompaniment_program,
            channel=0,
            ticks_per_beat=ticks_per_beat,
        )
    )
    midi.save(path)


def _read_metric_groundtruth(path: Path) -> tuple[NoteValue, ...]:
    midi = _load_midi(path)
    piano = [
        instrument for instrument in midi.instruments if instrument.name == "Piano"
    ]
    if len(piano) != 1:
        raise PreparationError(
            f"metric-ready GT requires exactly one track named 'Piano': {path}"
        )
    if any(instrument.name != "Piano" for instrument in midi.instruments):
        raise PreparationError(
            f"metric-ready GT must not contain Melody or other instruments: {path}"
        )
    notes: list[NoteValue] = []
    for note in piano[0].notes:
        if (
            note.start < -TIME_EPSILON_S
            or note.end > WINDOW_DURATION_S + TIME_EPSILON_S
        ):
            raise PreparationError(
                f"metric-ready GT note lies outside [0, 12]s: {path}"
            )
        if note.end <= note.start:
            raise PreparationError(
                f"metric-ready GT has non-positive note duration: {path}"
            )
        notes.append(
            NoteValue(
                pitch=int(note.pitch),
                velocity=int(note.velocity),
                start=max(0.0, float(note.start)),
                end=min(WINDOW_DURATION_S, float(note.end)),
            )
        )
    if not notes:
        raise PreparationError(f"metric-ready GT Piano track is empty: {path}")
    return tuple(sorted(notes, key=lambda n: (n.start, n.pitch, n.end, n.velocity)))


def _read_offline_postjoin_gt(path: Path) -> tuple[NoteValue, ...]:
    """Read either a full named GT pair or accompaniment-only Piano GT."""
    midi = _load_midi(path)
    piano = [
        instrument for instrument in midi.instruments if instrument.name == "Piano"
    ]
    if piano:
        return _read_metric_groundtruth(path)
    return _extract_window(
        path,
        already_postjoin=True,
        require_accompaniment_track=True,
        require_accompaniment_notes=True,
    ).accompaniment


def _empty_audit(trial: Trial, cohort_piece: CohortPiece) -> dict[str, Any]:
    basename = f"piece-{_slug(trial.piece_id)}__seed-{_slug(trial.seed)}.mid"
    system_dir = _slug(trial.system_id)
    source_gt = (
        cohort_piece.gt_midi
        if trial.source_kind == "realtime_session"
        else trial.gt_midi
    )
    source_gt_sha256 = (
        cohort_piece.gt_midi_sha256
        if trial.source_kind == "realtime_session"
        else _existing_file_sha256(source_gt)
    )
    return {
        "source_kind": trial.source_kind,
        "system_scope": trial.system_scope,
        "piece_id": trial.piece_id,
        "seed": trial.seed,
        "system_id": trial.system_id,
        "source_status": trial.run_status,
        "source_generated_midi": None,
        "source_gt_midi": str(source_gt) if source_gt is not None else None,
        "cohort_full_gt_midi": str(cohort_piece.gt_midi),
        "source_melody_midi": str(cohort_piece.melody_midi),
        "all_trials_generated_midi": (f"{system_dir}/all_trials/generated/{basename}"),
        "all_trials_metric_gt_midi": (
            f"{system_dir}/all_trials/groundtruth/{basename}"
        ),
        "valid_only_generated_midi": None,
        "valid_only_metric_gt_midi": None,
        "melody_input_sha256": cohort_piece.melody_midi_sha256,
        "melody_hash_kind": HASH_KIND,
        "source_gt_sha256": source_gt_sha256,
        "cohort_full_gt_sha256": cohort_piece.gt_midi_sha256,
        "cohort_postjoin_melody_note_count": len(cohort_piece.postjoin_gt.melody),
        "cohort_postjoin_melody_sha256": cohort_piece.postjoin_melody_sha256,
        "cohort_source_npz": (
            str(cohort_piece.source_npz)
            if cohort_piece.source_npz is not None
            else None
        ),
        "cohort_source_npz_sha256": cohort_piece.source_npz_sha256,
        "trial_source_npz_sha256": trial.source_npz_sha256,
        "offline_gt_roundtrip_exact": None,
        "generated_sha256": None,
        "metric_gt_sha256": None,
        "generated_mel_note_count": None,
        "generated_acc_note_count": None,
        "source_gt_mel_note_count": None,
        "source_gt_acc_note_count": None,
        "metric_gt_acc_note_count": None,
        "valid_output": None,
        "window_start_beat": WINDOW_START_BEAT,
        "window_end_beat_exclusive": WINDOW_END_BEAT,
        "window_start_s": WINDOW_START_S,
        "window_end_s": WINDOW_END_S,
        "preparation_status": "blocked",
        "reason": trial.failure_reason,
    }


def _write_audit(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def prepare_matched_music_eval(
    *,
    cohort_manifest: Path,
    output_dir: Path,
    realtime_manifest: Path | None = None,
    offline_manifest: Path | None = None,
    expected_piece_count: int = DEFAULT_EXPECTED_PIECE_COUNT,
    expected_seeds: Sequence[str] = DEFAULT_EXPECTED_SEEDS,
) -> dict[str, Any]:
    """Validate and prepare matched post-join MIDI pairs.

    A blocked batch writes audit metadata but does not publish generated or
    groundtruth directories, then raises :class:`PreparationBlockedError`.
    """
    if realtime_manifest is None and offline_manifest is None:
        raise PreparationError("at least one input manifest is required")
    expected_seeds = tuple(str(seed) for seed in expected_seeds)
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise PreparationError(f"output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    cohort = load_cohort_manifest(cohort_manifest)
    trials: list[Trial] = []
    if realtime_manifest is not None:
        trials.extend(load_realtime_manifest(realtime_manifest))
    if offline_manifest is not None:
        trials.extend(load_offline_manifest(offline_manifest))
    trials = _validate_grid(
        trials,
        cohort,
        expected_piece_count=expected_piece_count,
        expected_seeds=expected_seeds,
    )

    staging = output_dir / f".staging-{uuid.uuid4().hex}"
    staging.mkdir()
    for system_id in sorted({trial.system_id for trial in trials}):
        system_dir = staging / _slug(system_id)
        for collection in ("all_trials", "valid_only"):
            (system_dir / collection / "generated").mkdir(parents=True)
            (system_dir / collection / "groundtruth").mkdir(parents=True)
    audit_rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    target_keys: set[tuple[str, str]] = set()
    source_paths: set[str] = set()

    try:
        for trial in trials:
            cohort_piece = cohort[trial.piece_id]
            audit = _empty_audit(trial, cohort_piece)
            if trial.run_status != "complete":
                reason = (
                    f"{trial.system_id}/{trial.piece_id}/seed {trial.seed}: "
                    f"{trial.run_status}: {trial.failure_reason}"
                )
                blockers.append(reason)
                audit_rows.append(audit)
                continue

            try:
                target_key = (
                    trial.system_id,
                    Path(audit["all_trials_generated_midi"]).name,
                )
                if target_key in target_keys:
                    raise PreparationError(f"duplicate output basename: {target_key}")
                target_keys.add(target_key)

                cohort_gt = cohort_piece.postjoin_gt

                if trial.source_kind == "realtime_session":
                    if trial.melody_input_sha256 != cohort_piece.melody_midi_sha256:
                        raise PreparationError(
                            "realtime manifest input hash does not match cohort "
                            f"melody MIDI file SHA256 ({HASH_KIND})"
                        )
                    if trial.session_dir is None or not trial.session_dir.is_dir():
                        raise PreparationError(
                            f"session_dir not found: {trial.session_dir}"
                        )
                    source_generated = trial.session_dir / "combined.mid"
                    generated = _extract_window(
                        source_generated,
                        already_postjoin=False,
                        require_accompaniment_track=False,
                        require_accompaniment_notes=False,
                    )
                else:
                    if trial.generated_midi is None or trial.gt_midi is None:
                        raise PreparationError("complete offline row lacks MIDI paths")
                    if trial.source_npz_sha256 is None:
                        raise PreparationError(
                            "offline source_npz_sha256 is required for canonical identity"
                        )
                    if trial.source_npz_sha256 != cohort_piece.source_npz_sha256:
                        raise PreparationError(
                            "offline source_npz_sha256 does not match cohort canonical "
                            "source NPZ"
                        )
                    source_generated = trial.generated_midi
                    provided_gt = _read_offline_postjoin_gt(trial.gt_midi)
                    audit["offline_gt_roundtrip_exact"] = _canonical_note_geometry(
                        provided_gt
                    ) == _canonical_note_geometry(cohort_gt.accompaniment)
                    generated = _extract_window(
                        source_generated,
                        already_postjoin=True,
                        require_accompaniment_track=False,
                        require_accompaniment_notes=False,
                    )

                source_key = str(source_generated.resolve()).casefold()
                if source_key in source_paths:
                    raise PreparationError(
                        f"generated source MIDI is reused by another trial: {source_generated}"
                    )
                source_paths.add(source_key)

                generated_target = staging / audit["all_trials_generated_midi"]
                gt_target = staging / audit["all_trials_metric_gt_midi"]
                write_window_midi(generated, generated_target)
                write_metric_groundtruth(cohort_gt, gt_target)

                generated_check = _extract_window(
                    generated_target,
                    already_postjoin=True,
                    require_accompaniment_track=False,
                    require_accompaniment_notes=False,
                )
                gt_check = _read_metric_groundtruth(gt_target)
                if _canonical_notes(generated_check) != _canonical_notes(generated):
                    raise PreparationError(
                        "written generated MIDI failed round-trip check"
                    )
                if _canonical_note_values(gt_check) != _canonical_note_values(
                    cohort_gt.accompaniment
                ):
                    raise PreparationError(
                        "written groundtruth MIDI failed round-trip check"
                    )

                valid_output = bool(generated.accompaniment)
                if valid_output:
                    basename = generated_target.name
                    system_dir = _slug(trial.system_id)
                    valid_generated = (
                        staging / system_dir / "valid_only" / "generated" / basename
                    )
                    valid_gt = (
                        staging / system_dir / "valid_only" / "groundtruth" / basename
                    )
                    valid_generated.parent.mkdir(parents=True, exist_ok=True)
                    valid_gt.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(generated_target, valid_generated)
                    shutil.copy2(gt_target, valid_gt)
                    audit["valid_only_generated_midi"] = str(
                        valid_generated.relative_to(staging)
                    ).replace("\\", "/")
                    audit["valid_only_metric_gt_midi"] = str(
                        valid_gt.relative_to(staging)
                    ).replace("\\", "/")

                audit.update(
                    {
                        "source_generated_midi": str(source_generated.resolve()),
                        "source_gt_midi": str(
                            (trial.gt_midi or cohort_piece.gt_midi).resolve()
                        ),
                        "generated_sha256": file_sha256(generated_target),
                        "metric_gt_sha256": file_sha256(gt_target),
                        "generated_mel_note_count": len(generated.melody),
                        "generated_acc_note_count": len(generated.accompaniment),
                        "source_gt_mel_note_count": len(cohort_gt.melody),
                        "source_gt_acc_note_count": len(cohort_gt.accompaniment),
                        "metric_gt_acc_note_count": len(gt_check),
                        "valid_output": valid_output,
                        "preparation_status": "prepared",
                        "reason": "",
                    }
                )
            except PreparationError as exc:
                audit["reason"] = str(exc)
                blockers.append(
                    f"{trial.system_id}/{trial.piece_id}/seed {trial.seed}: {exc}"
                )
            audit_rows.append(audit)

        status = "blocked" if blockers else "success"
        if blockers:
            for row in audit_rows:
                if row["preparation_status"] == "prepared":
                    row["preparation_status"] = "validated_not_published"
                for field in PUBLISHED_PATH_FIELDS:
                    row[field] = None
        systems = [
            {
                "system_id": system_id,
                "source_kind": next(
                    trial.source_kind
                    for trial in trials
                    if trial.system_id == system_id
                ),
                "system_scope": next(
                    trial.system_scope
                    for trial in trials
                    if trial.system_id == system_id
                ),
            }
            for system_id in sorted({trial.system_id for trial in trials})
        ]
        system_summaries: dict[str, dict[str, Any]] = {}
        for system in systems:
            system_id = system["system_id"]
            rows = [row for row in audit_rows if row["system_id"] == system_id]
            planned_count = len(rows)
            complete_count = sum(row["source_status"] == "complete" for row in rows)
            prepared_count = sum(
                row["preparation_status"] == "prepared" for row in rows
            )
            validated_count = sum(
                row["preparation_status"] in {"prepared", "validated_not_published"}
                for row in rows
            )
            valid_count = sum(row["valid_output"] is True for row in rows)
            system_summaries[system_id] = {
                "planned_trial_count": planned_count,
                "complete_source_trial_count": complete_count,
                "validated_trial_count": validated_count,
                "prepared_trial_count": prepared_count,
                "valid_output_count": valid_count,
                "valid_output_rate": (
                    valid_count / planned_count if planned_count else None
                ),
                "valid_output_denominator": "all planned matched trials",
                "music_metrics_scope": "conditional_on_valid_output",
            }
        manifest = {
            "schema_version": 1,
            "preparation_status": status,
            "tool_scope": "music_quality_input_preparation",
            "produces_system_metrics": False,
            "melody_hash_kind": HASH_KIND,
            "cohort_manifest": str(cohort_manifest.resolve()),
            "window": {
                "bpm": BPM,
                "start_beat": WINDOW_START_BEAT,
                "end_beat_exclusive": WINDOW_END_BEAT,
                "source_time_range_s": [WINDOW_START_S, WINDOW_END_S],
                "prepared_time_range_s": [0.0, WINDOW_DURATION_S],
                "boundary_policy": "clip notes to the window, then shift to zero",
            },
            "expected_grid": {
                "piece_count": expected_piece_count,
                "seeds": list(expected_seeds),
                "systems": [item["system_id"] for item in systems],
            },
            "systems": systems,
            "system_summaries": system_summaries,
            "blockers": blockers,
            "trials": audit_rows,
        }

        if not blockers:
            for child in staging.iterdir():
                child.rename(output_dir / child.name)
        shutil.rmtree(staging, ignore_errors=True)
        _write_audit(output_dir / "audit.csv", audit_rows)
        _write_json(output_dir / "prepared_manifest.json", manifest)
        if blockers:
            raise PreparationBlockedError(
                f"preparation blocked by {len(blockers)} trial/grid issue(s); "
                f"see {output_dir / 'audit.csv'}"
            )
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _parse_expected_seeds(value: str) -> tuple[str, ...]:
    seeds = tuple(part.strip() for part in value.split(",") if part.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("expected seeds cannot be empty")
    if len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("expected seeds must be unique")
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare matched post-join MIDI pairs for music metrics."
    )
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--realtime-manifest", type=Path)
    parser.add_argument("--offline-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-piece-count", type=int, default=DEFAULT_EXPECTED_PIECE_COUNT
    )
    parser.add_argument(
        "--expected-seeds",
        type=_parse_expected_seeds,
        default=DEFAULT_EXPECTED_SEEDS,
        help="Comma-separated seed IDs (default: 0,1,2).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.realtime_manifest is None and args.offline_manifest is None:
        parser.error(
            "at least one of --realtime-manifest or --offline-manifest is required"
        )
    try:
        manifest = prepare_matched_music_eval(
            cohort_manifest=args.cohort_manifest,
            realtime_manifest=args.realtime_manifest,
            offline_manifest=args.offline_manifest,
            output_dir=args.output_dir,
            expected_piece_count=args.expected_piece_count,
            expected_seeds=args.expected_seeds,
        )
    except PreparationBlockedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except PreparationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"Prepared {len(manifest['trials'])} matched music-quality trial(s) "
        f"in {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
