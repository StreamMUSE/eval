"""Materialize the common-valid subset of prepared matched music trials."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED_AUDIT_FIELDS = {
    "piece_id",
    "seed",
    "system_id",
    "source_status",
    "preparation_status",
    "valid_output",
    "valid_only_generated_midi",
    "valid_only_metric_gt_midi",
    "generated_sha256",
    "metric_gt_sha256",
}
CSV_FIELDS = (
    "piece_id",
    "seed",
    "system_id",
    "basename",
    "source_generated_midi",
    "source_metric_gt_midi",
    "generated_sha256",
    "metric_gt_sha256",
    "common_generated_midi",
    "common_metric_gt_midi",
)
PUBLISH_RENAME_MAX_ATTEMPTS = 5
PUBLISH_RENAME_INITIAL_BACKOFF_SECONDS = 0.05


class CommonValidMaterializationError(ValueError):
    """Raised when the audit or its prepared artifacts violate the contract."""


@dataclass(frozen=True)
class AuditTrial:
    piece_id: str
    seed: str
    system_id: str
    valid_output: bool
    generated_path: Path | None
    metric_gt_path: Path | None
    generated_sha256: str | None
    metric_gt_sha256: str | None
    basename: str | None
    line_number: int


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_bool(value: str, context: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise CommonValidMaterializationError(
        f"{context} must be exactly true or false, found {value!r}"
    )


def _strict_sha256(value: str, context: str) -> str:
    normalized = value.strip()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise CommonValidMaterializationError(
            f"{context} must be 64 lowercase hexadecimal characters"
        )
    return normalized


def _system_slug(system_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", system_id).strip("._-")
    if not slug:
        raise CommonValidMaterializationError(
            f"system ID {system_id!r} cannot form a directory name"
        )
    return slug


def normalize_system_ids(system_ids: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(system_id.strip() for system_id in system_ids)
    if not normalized or any(not system_id for system_id in normalized):
        raise CommonValidMaterializationError(
            "at least one non-empty --system-id is required"
        )
    if len(normalized) != len(set(normalized)):
        raise CommonValidMaterializationError("system IDs must be unique")
    ordered = tuple(sorted(normalized))
    slugs = [_system_slug(system_id) for system_id in ordered]
    if len(slugs) != len(set(slugs)):
        raise CommonValidMaterializationError(
            "system IDs collide after conversion to output directory names"
        )
    return ordered


def _resolve_recorded_path(raw_path: str, audit_path: Path, context: str) -> Path:
    value = raw_path.strip()
    if not value:
        raise CommonValidMaterializationError(f"{context} is empty")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = audit_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise CommonValidMaterializationError(
            f"{context} does not exist or is not a file: {path}"
        )
    return path


def _validate_source_hash(path: Path, expected: str, context: str) -> None:
    actual = file_sha256(path)
    if actual != expected:
        raise CommonValidMaterializationError(
            f"{context} SHA256 mismatch: expected {expected}, found {actual}"
        )


def load_common_valid_trials(
    audit_path: Path,
    system_ids: Sequence[str],
) -> tuple[
    dict[str, dict[tuple[str, str], AuditTrial]],
    tuple[tuple[str, str], ...],
]:
    """Load and strictly validate a preparation audit and its valid artifacts."""

    audit_path = audit_path.expanduser().resolve()
    ordered_system_ids = normalize_system_ids(system_ids)
    selected_systems = set(ordered_system_ids)
    rows_by_system: dict[str, dict[tuple[str, str], AuditTrial]] = {
        system_id: {} for system_id in ordered_system_ids
    }
    audit_systems: set[str] = set()

    try:
        handle = audit_path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise CommonValidMaterializationError(
            f"cannot read audit CSV {audit_path}: {exc}"
        ) from exc

    try:
        with handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise CommonValidMaterializationError(
                    f"audit CSV has no header: {audit_path}"
                )
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise CommonValidMaterializationError(
                    f"audit CSV has duplicate columns: {audit_path}"
                )
            missing_fields = sorted(REQUIRED_AUDIT_FIELDS - set(reader.fieldnames))
            if missing_fields:
                raise CommonValidMaterializationError(
                    "audit CSV is missing required columns: "
                    + ", ".join(missing_fields)
                )

            for line_number, row in enumerate(reader, start=2):
                system_id = str(row.get("system_id") or "").strip()
                if not system_id:
                    raise CommonValidMaterializationError(
                        f"audit line {line_number} has an empty system_id"
                    )
                audit_systems.add(system_id)
                if system_id not in selected_systems:
                    continue

                piece_id = str(row.get("piece_id") or "").strip()
                seed = str(row.get("seed") or "").strip()
                if not piece_id or not seed:
                    raise CommonValidMaterializationError(
                        f"audit line {line_number} has empty piece_id or seed"
                    )
                if str(row.get("source_status") or "").strip() != "complete":
                    raise CommonValidMaterializationError(
                        f"audit line {line_number} source_status is not complete"
                    )
                if str(row.get("preparation_status") or "").strip() != "prepared":
                    raise CommonValidMaterializationError(
                        f"audit line {line_number} preparation_status is not prepared"
                    )

                key = (piece_id, seed)
                if key in rows_by_system[system_id]:
                    raise CommonValidMaterializationError(
                        f"duplicate audit key for {system_id}/{piece_id}/seed {seed}"
                    )
                valid_output = _strict_bool(
                    str(row.get("valid_output") or ""),
                    f"audit line {line_number} valid_output",
                )

                generated_path: Path | None = None
                metric_gt_path: Path | None = None
                generated_sha256: str | None = None
                metric_gt_sha256: str | None = None
                basename: str | None = None
                raw_generated_path = str(
                    row.get("valid_only_generated_midi") or ""
                ).strip()
                raw_gt_path = str(
                    row.get("valid_only_metric_gt_midi") or ""
                ).strip()

                if valid_output:
                    generated_path = _resolve_recorded_path(
                        raw_generated_path,
                        audit_path,
                        f"audit line {line_number} valid_only_generated_midi",
                    )
                    metric_gt_path = _resolve_recorded_path(
                        raw_gt_path,
                        audit_path,
                        f"audit line {line_number} valid_only_metric_gt_midi",
                    )
                    if generated_path.name != metric_gt_path.name:
                        raise CommonValidMaterializationError(
                            f"audit line {line_number} generated/groundtruth "
                            "basenames differ"
                        )
                    basename = generated_path.name
                    generated_sha256 = _strict_sha256(
                        str(row.get("generated_sha256") or ""),
                        f"audit line {line_number} generated_sha256",
                    )
                    metric_gt_sha256 = _strict_sha256(
                        str(row.get("metric_gt_sha256") or ""),
                        f"audit line {line_number} metric_gt_sha256",
                    )
                    _validate_source_hash(
                        generated_path,
                        generated_sha256,
                        f"audit line {line_number} generated MIDI",
                    )
                    _validate_source_hash(
                        metric_gt_path,
                        metric_gt_sha256,
                        f"audit line {line_number} metric GT MIDI",
                    )
                elif raw_generated_path or raw_gt_path:
                    raise CommonValidMaterializationError(
                        f"audit line {line_number} is invalid-output but records "
                        "a valid_only path"
                    )

                rows_by_system[system_id][key] = AuditTrial(
                    piece_id=piece_id,
                    seed=seed,
                    system_id=system_id,
                    valid_output=valid_output,
                    generated_path=generated_path,
                    metric_gt_path=metric_gt_path,
                    generated_sha256=generated_sha256,
                    metric_gt_sha256=metric_gt_sha256,
                    basename=basename,
                    line_number=line_number,
                )
    except csv.Error as exc:
        raise CommonValidMaterializationError(
            f"invalid audit CSV {audit_path}: {exc}"
        ) from exc

    if audit_systems != selected_systems:
        missing = sorted(audit_systems - selected_systems)
        unexpected = sorted(selected_systems - audit_systems)
        raise CommonValidMaterializationError(
            "explicit system IDs must exactly match the audit system set; "
            f"unselected_in_audit={missing}, absent_from_audit={unexpected}"
        )

    reference_keys: set[tuple[str, str]] | None = None
    for system_id in ordered_system_ids:
        keys = set(rows_by_system[system_id])
        if reference_keys is None:
            reference_keys = keys
        elif keys != reference_keys:
            missing = sorted(reference_keys - keys)
            extra = sorted(keys - reference_keys)
            raise CommonValidMaterializationError(
                f"system {system_id!r} does not share the complete audit key grid; "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
    assert reference_keys is not None

    common_keys = tuple(
        key
        for key in sorted(reference_keys)
        if all(rows_by_system[system_id][key].valid_output for system_id in ordered_system_ids)
    )

    basenames_by_system: dict[str, set[str]] = {
        system_id: set() for system_id in ordered_system_ids
    }
    for key in common_keys:
        rows = [rows_by_system[system_id][key] for system_id in ordered_system_ids]
        basenames = {row.basename for row in rows}
        if len(basenames) != 1:
            raise CommonValidMaterializationError(
                f"common key {key!r} has inconsistent basenames across systems: "
                f"{sorted(str(value) for value in basenames)}"
            )
        gt_hashes = {row.metric_gt_sha256 for row in rows}
        if len(gt_hashes) != 1:
            raise CommonValidMaterializationError(
                f"common key {key!r} has inconsistent metric GT hashes across systems"
            )
        basename = rows[0].basename
        assert basename is not None
        for system_id in ordered_system_ids:
            if basename in basenames_by_system[system_id]:
                raise CommonValidMaterializationError(
                    f"system {system_id!r} maps multiple common keys to {basename!r}"
                )
            basenames_by_system[system_id].add(basename)

    return rows_by_system, common_keys


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=CSV_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _publish_staging_directory(staging: Path, output_dir: Path) -> None:
    delay = PUBLISH_RENAME_INITIAL_BACKOFF_SECONDS
    for attempt in range(PUBLISH_RENAME_MAX_ATTEMPTS):
        if output_dir.exists():
            raise CommonValidMaterializationError(
                f"output directory appeared during publication: {output_dir}"
            )
        try:
            staging.rename(output_dir)
            return
        except PermissionError:
            if attempt + 1 == PUBLISH_RENAME_MAX_ATTEMPTS:
                raise
            time.sleep(delay)
            delay *= 2


def materialize_common_valid_music_eval(
    *,
    audit_path: Path,
    system_ids: Sequence[str],
    output_dir: Path,
) -> dict[str, Any]:
    """Publish basename-matched common-valid MIDI directories atomically."""

    audit_path = audit_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    ordered_system_ids = normalize_system_ids(system_ids)
    if output_dir.exists():
        raise CommonValidMaterializationError(
            f"output directory already exists; refusing to overwrite: {output_dir}"
        )

    rows_by_system, common_keys = load_common_valid_trials(
        audit_path,
        ordered_system_ids,
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()

    flat_rows: list[dict[str, Any]] = []
    system_summaries: list[dict[str, Any]] = []
    try:
        for system_id in ordered_system_ids:
            system_dir = _system_slug(system_id)
            generated_dir = staging / system_dir / "generated"
            groundtruth_dir = staging / system_dir / "groundtruth"
            generated_dir.mkdir(parents=True)
            groundtruth_dir.mkdir(parents=True)

            for piece_id, seed in common_keys:
                trial = rows_by_system[system_id][(piece_id, seed)]
                assert trial.generated_path is not None
                assert trial.metric_gt_path is not None
                assert trial.generated_sha256 is not None
                assert trial.metric_gt_sha256 is not None
                assert trial.basename is not None

                generated_target = generated_dir / trial.basename
                gt_target = groundtruth_dir / trial.basename
                shutil.copy2(trial.generated_path, generated_target)
                shutil.copy2(trial.metric_gt_path, gt_target)
                _validate_source_hash(
                    generated_target,
                    trial.generated_sha256,
                    f"copied generated MIDI for {system_id}/{piece_id}/seed {seed}",
                )
                _validate_source_hash(
                    gt_target,
                    trial.metric_gt_sha256,
                    f"copied metric GT MIDI for {system_id}/{piece_id}/seed {seed}",
                )
                flat_rows.append(
                    {
                        "piece_id": piece_id,
                        "seed": seed,
                        "system_id": system_id,
                        "basename": trial.basename,
                        "source_generated_midi": str(trial.generated_path),
                        "source_metric_gt_midi": str(trial.metric_gt_path),
                        "generated_sha256": trial.generated_sha256,
                        "metric_gt_sha256": trial.metric_gt_sha256,
                        "common_generated_midi": (
                            f"{system_dir}/generated/{trial.basename}"
                        ),
                        "common_metric_gt_midi": (
                            f"{system_dir}/groundtruth/{trial.basename}"
                        ),
                    }
                )

            valid_count = sum(
                trial.valid_output for trial in rows_by_system[system_id].values()
            )
            system_summaries.append(
                {
                    "system_id": system_id,
                    "directory": system_dir,
                    "audit_trial_count": len(rows_by_system[system_id]),
                    "valid_output_count": valid_count,
                    "common_valid_count": len(common_keys),
                    "generated_count": len(list(generated_dir.iterdir())),
                    "groundtruth_count": len(list(groundtruth_dir.iterdir())),
                }
            )

        flat_rows.sort(
            key=lambda row: (row["piece_id"], row["seed"], row["system_id"])
        )
        manifest = {
            "schema_version": 1,
            "audit_path": str(audit_path),
            "audit_sha256": file_sha256(audit_path),
            "system_ids": list(ordered_system_ids),
            "key_fields": ["piece_id", "seed"],
            "common_valid_key_count": len(common_keys),
            "common_valid_keys": [
                {"piece_id": piece_id, "seed": seed}
                for piece_id, seed in common_keys
            ],
            "systems": system_summaries,
            "trials": flat_rows,
        }
        _write_json(staging / "manifest.json", manifest)
        _write_csv(staging / "manifest.csv", flat_rows)

        _publish_staging_directory(staging, output_dir)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the exact common-valid intersection from a matched "
            "music preparation audit."
        )
    )
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument(
        "--system-id",
        action="append",
        default=[],
        help="Repeat once for every system in the audit.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = materialize_common_valid_music_eval(
            audit_path=args.audit,
            system_ids=args.system_id,
            output_dir=args.output_dir,
        )
    except CommonValidMaterializationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"Published {manifest['common_valid_key_count']} common-valid key(s) "
        f"for {len(manifest['system_ids'])} system(s) to "
        f"{args.output_dir.expanduser().resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
