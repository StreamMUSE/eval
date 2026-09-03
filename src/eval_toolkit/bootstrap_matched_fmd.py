"""Bootstrap CLAMP2 Frechet Music Distance for common-valid matched systems.

The official ``frechet-music-distance`` point estimate uses
``np.cov(rowvar=False)`` and therefore the unbiased ``n - 1`` covariance.
For centered feature matrices ``Xc`` and ``Yc`` with ``n`` and ``m`` rows,
write ``A = Xc / sqrt(n - 1)`` and ``B = Yc / sqrt(m - 1)``. Then the
covariances are ``A.T @ A`` and ``B.T @ B``. The non-zero eigenvalues of
``(A.T @ A) @ (B.T @ B)`` match those of ``(A @ B.T) @ (A @ B.T).T`` by the
standard ``AB``/``BA`` eigenvalue equivalence, so
``tr(sqrt(cov_x @ cov_y)) = ||A @ B.T||_*``. Substituting gives

    FMD = ||mu_x - mu_y||^2
        + ||Xc||_F^2 / (n - 1)
        + ||Yc||_F^2 / (m - 1)
        - 2 * ||Xc @ Yc.T||_* / sqrt((n - 1) * (m - 1)).

This module evaluates that expression from sample-space Gram submatrices.
Bootstrap repeats are handled by repeated sample indices in the submatrices,
which preserves the exact covariance convention without forming 768x768
matrix square roots.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_BOOTSTRAP_REPLICATES = 10_000
DEFAULT_BOOTSTRAP_SEED = 0
DEFAULT_FEATURE_EXTRACTOR = "clamp2"
DEFAULT_GAUSSIAN_ESTIMATOR = "mle"
DEFAULT_VALIDATION_ATOL = 1e-6
DEFAULT_VALIDATION_RTOL = 1e-6
OUTPUT_SCHEMA_VERSION = 1
FEATURE_CACHE_SCHEMA_VERSION = 1
FEATURE_ROLES = ("generated", "groundtruth")


class BootstrapFMDValidationError(ValueError):
    """Raised when common-valid FMD inputs violate the formal contract."""


@dataclass(frozen=True)
class ManifestTrial:
    system_id: str
    piece_id: str
    seed: str
    basename: str
    generated_path: Path
    groundtruth_path: Path
    generated_relative_path: str
    groundtruth_relative_path: str
    generated_sha256: str
    groundtruth_sha256: str

    @property
    def key(self) -> tuple[str, str]:
        return self.piece_id, self.seed


@dataclass(frozen=True)
class CommonValidManifest:
    path: Path
    root: Path
    sha256: str
    system_ids: tuple[str, ...]
    key_order: tuple[tuple[str, str], ...]
    piece_order: tuple[str, ...]
    trials_by_system: dict[str, tuple[ManifestTrial, ...]]


@dataclass(frozen=True)
class FeatureRequest:
    system_id: str
    piece_id: str
    seed: str
    role: str
    relative_path: str
    absolute_path: Path
    sha256: str

    @property
    def payload(self) -> dict[str, str]:
        return {
            "system_id": self.system_id,
            "piece_id": self.piece_id,
            "seed": self.seed,
            "role": self.role,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
        }

    @property
    def key_payload(self) -> dict[str, str]:
        return {
            "system_id": self.system_id,
            "piece_id": self.piece_id,
            "seed": self.seed,
            "role": self.role,
        }

    @property
    def entry_id(self) -> str:
        return hashlib.sha256(_canonical_json_bytes(self.payload)).hexdigest()


@dataclass(frozen=True)
class FeatureCacheResult:
    vectors_by_entry_id: dict[str, np.ndarray]
    provenance: dict[str, Any]
    cache_path: Path
    cache_sha256: str
    entry_count: int
    unique_midi_hash_count: int
    cache_hits: int
    cache_misses: int
    extracted_midi_count: int


@dataclass(frozen=True)
class SampleSpaceFMD:
    x_gram: np.ndarray
    y_gram: np.ndarray
    cross_gram: np.ndarray


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise BootstrapFMDValidationError(f"cannot read JSON {path}: {exc}") from exc


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _string_field(row: Mapping[str, Any], field: str, context: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise BootstrapFMDValidationError(f"{context}.{field} must be a non-empty string")
    return value.strip()


def _strict_sha256(value: Any, context: str) -> str:
    if not isinstance(value, str):
        raise BootstrapFMDValidationError(
            f"{context} must be 64 lowercase hexadecimal characters"
        )
    normalized = value.strip()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise BootstrapFMDValidationError(
            f"{context} must be 64 lowercase hexadecimal characters"
        )
    return normalized


def _finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BootstrapFMDValidationError(f"{context} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise BootstrapFMDValidationError(f"{context} must be a finite number")
    return number


def _strict_integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BootstrapFMDValidationError(f"{context} must be an integer")
    return value


def _resolve_manifest_relative_path(
    raw_path: str,
    manifest_root: Path,
    context: str,
) -> tuple[Path, str]:
    raw_path = raw_path.strip()
    if not raw_path:
        raise BootstrapFMDValidationError(f"{context} is empty")
    relative = Path(raw_path)
    if relative.is_absolute():
        raise BootstrapFMDValidationError(f"{context} must be relative to manifest.json")
    resolved = (manifest_root / relative).resolve()
    try:
        resolved.relative_to(manifest_root)
    except ValueError as exc:
        raise BootstrapFMDValidationError(
            f"{context} escapes the manifest directory: {raw_path!r}"
        ) from exc
    if not resolved.is_file():
        raise BootstrapFMDValidationError(
            f"{context} does not exist or is not a file: {resolved}"
        )
    return resolved, relative.as_posix()


def _parse_common_key(raw: Any, context: str) -> tuple[str, str]:
    if not isinstance(raw, Mapping):
        raise BootstrapFMDValidationError(f"{context} must be an object")
    piece_id = _string_field(raw, "piece_id", context)
    seed = _string_field(raw, "seed", context)
    return piece_id, seed


def load_common_valid_manifest(manifest_path: Path) -> CommonValidManifest:
    """Load and strictly validate a materialized common-valid manifest."""

    manifest_path = manifest_path.expanduser().resolve()
    manifest_root = manifest_path.parent.resolve()
    data = _read_json(manifest_path)
    if not isinstance(data, Mapping):
        raise BootstrapFMDValidationError("manifest root must be an object")
    if _strict_integer(data.get("schema_version"), "manifest.schema_version") != 1:
        raise BootstrapFMDValidationError("manifest.schema_version must be exactly 1")

    raw_system_ids = data.get("system_ids")
    if not isinstance(raw_system_ids, list):
        raise BootstrapFMDValidationError("manifest.system_ids must be a list")
    manifest_system_ids = tuple(
        str(system_id).strip()
        for system_id in raw_system_ids
        if isinstance(system_id, str) and system_id.strip()
    )
    if len(manifest_system_ids) != len(raw_system_ids) or not manifest_system_ids:
        raise BootstrapFMDValidationError(
            "manifest.system_ids must contain non-empty strings"
        )
    if len(manifest_system_ids) != len(set(manifest_system_ids)):
        raise BootstrapFMDValidationError("manifest.system_ids must be unique")

    raw_trials = data.get("trials")
    if not isinstance(raw_trials, list) or not raw_trials:
        raise BootstrapFMDValidationError("manifest.trials must be a non-empty list")

    rows_by_system: dict[str, dict[tuple[str, str], ManifestTrial]] = {}
    for index, raw_trial in enumerate(raw_trials):
        context = f"manifest.trials[{index}]"
        if not isinstance(raw_trial, Mapping):
            raise BootstrapFMDValidationError(f"{context} must be an object")
        system_id = _string_field(raw_trial, "system_id", context)
        piece_id = _string_field(raw_trial, "piece_id", context)
        seed = _string_field(raw_trial, "seed", context)
        generated_path, generated_relative = _resolve_manifest_relative_path(
            _string_field(raw_trial, "common_generated_midi", context),
            manifest_root,
            f"{context}.common_generated_midi",
        )
        groundtruth_path, groundtruth_relative = _resolve_manifest_relative_path(
            _string_field(raw_trial, "common_metric_gt_midi", context),
            manifest_root,
            f"{context}.common_metric_gt_midi",
        )
        if generated_path.name != groundtruth_path.name:
            raise BootstrapFMDValidationError(
                f"{context} generated/GT basenames differ: "
                f"{generated_path.name!r} != {groundtruth_path.name!r}"
            )
        basename = generated_path.name
        raw_basename = raw_trial.get("basename")
        if raw_basename is not None and raw_basename != basename:
            raise BootstrapFMDValidationError(
                f"{context}.basename does not match common MIDI basename"
            )
        generated_sha256 = _strict_sha256(
            raw_trial.get("generated_sha256"),
            f"{context}.generated_sha256",
        )
        groundtruth_sha256 = _strict_sha256(
            raw_trial.get("metric_gt_sha256"),
            f"{context}.metric_gt_sha256",
        )
        actual_generated_sha256 = file_sha256(generated_path)
        if actual_generated_sha256 != generated_sha256:
            raise BootstrapFMDValidationError(
                f"{context} generated MIDI SHA256 mismatch: expected "
                f"{generated_sha256}, found {actual_generated_sha256}"
            )
        actual_groundtruth_sha256 = file_sha256(groundtruth_path)
        if actual_groundtruth_sha256 != groundtruth_sha256:
            raise BootstrapFMDValidationError(
                f"{context} metric GT MIDI SHA256 mismatch: expected "
                f"{groundtruth_sha256}, found {actual_groundtruth_sha256}"
            )

        key = (piece_id, seed)
        system_rows = rows_by_system.setdefault(system_id, {})
        if key in system_rows:
            raise BootstrapFMDValidationError(
                f"duplicate common-valid key for {system_id}/{piece_id}/seed {seed}"
            )
        system_rows[key] = ManifestTrial(
            system_id=system_id,
            piece_id=piece_id,
            seed=seed,
            basename=basename,
            generated_path=generated_path,
            groundtruth_path=groundtruth_path,
            generated_relative_path=generated_relative,
            groundtruth_relative_path=groundtruth_relative,
            generated_sha256=generated_sha256,
            groundtruth_sha256=groundtruth_sha256,
        )

    trial_system_ids = set(rows_by_system)
    if trial_system_ids != set(manifest_system_ids):
        raise BootstrapFMDValidationError(
            "manifest.system_ids must exactly match trial systems; "
            f"declared={sorted(manifest_system_ids)}, "
            f"trials={sorted(trial_system_ids)}"
        )

    ordered_system_ids = tuple(sorted(manifest_system_ids))
    reference_keys: set[tuple[str, str]] | None = None
    for system_id in ordered_system_ids:
        keys = set(rows_by_system[system_id])
        if reference_keys is None:
            reference_keys = keys
        elif keys != reference_keys:
            missing = sorted(reference_keys - keys)
            extra = sorted(keys - reference_keys)
            raise BootstrapFMDValidationError(
                f"system {system_id!r} does not share the exact common key set; "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
    assert reference_keys is not None

    raw_common_keys = data.get("common_valid_keys")
    if not isinstance(raw_common_keys, list):
        raise BootstrapFMDValidationError("manifest.common_valid_keys must be a list")
    manifest_keys = tuple(
        _parse_common_key(raw_key, f"manifest.common_valid_keys[{index}]")
        for index, raw_key in enumerate(raw_common_keys)
    )
    if len(manifest_keys) != len(set(manifest_keys)):
        raise BootstrapFMDValidationError("manifest.common_valid_keys contains duplicates")
    declared_key_count = _strict_integer(
        data.get("common_valid_key_count"),
        "manifest.common_valid_key_count",
    )
    if declared_key_count != len(manifest_keys):
        raise BootstrapFMDValidationError(
            "manifest.common_valid_key_count does not match common_valid_keys"
        )
    if set(manifest_keys) != reference_keys:
        missing = sorted(reference_keys - set(manifest_keys))
        extra = sorted(set(manifest_keys) - reference_keys)
        raise BootstrapFMDValidationError(
            "manifest.common_valid_keys does not match trial keys; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    for key in sorted(reference_keys):
        rows = [rows_by_system[system_id][key] for system_id in ordered_system_ids]
        basenames = {row.basename for row in rows}
        if len(basenames) != 1:
            raise BootstrapFMDValidationError(
                f"common key {key!r} has inconsistent basenames across systems"
            )
        groundtruth_hashes = {row.groundtruth_sha256 for row in rows}
        if len(groundtruth_hashes) != 1:
            raise BootstrapFMDValidationError(
                f"common key {key!r} has inconsistent metric GT hashes across systems"
            )

    key_order = tuple(sorted(reference_keys))
    piece_order = tuple(sorted({piece_id for piece_id, _seed in key_order}))
    trials_by_system = {
        system_id: tuple(
            rows_by_system[system_id][key]
            for key in key_order
        )
        for system_id in ordered_system_ids
    }
    return CommonValidManifest(
        path=manifest_path,
        root=manifest_root,
        sha256=file_sha256(manifest_path),
        system_ids=ordered_system_ids,
        key_order=key_order,
        piece_order=piece_order,
        trials_by_system=trials_by_system,
    )


def _module_file(module_name: str) -> str | None:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        return None
    return str(Path(module_file).resolve())


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _extractor_identity(
    feature_extractor: object | None,
    *,
    injected: bool,
) -> dict[str, Any]:
    if feature_extractor is not None:
        extractor_class = feature_extractor.__class__
        try:
            source_file = inspect.getfile(extractor_class)
        except TypeError:
            source_file = None
        return {
            "name": DEFAULT_FEATURE_EXTRACTOR,
            "class_path": (
                f"{extractor_class.__module__}.{extractor_class.__qualname__}"
            ),
            "module_file": str(Path(source_file).resolve()) if source_file else None,
            "injected": injected,
        }
    try:
        from frechet_music_distance.models.clamp2.clamp2_extractor import (  # type: ignore
            CLaMP2Extractor,
        )
    except Exception:
        return {
            "name": DEFAULT_FEATURE_EXTRACTOR,
            "class_path": None,
            "module_file": _module_file("frechet_music_distance.models.clamp2"),
            "injected": False,
        }
    try:
        source_file = inspect.getfile(CLaMP2Extractor)
    except TypeError:
        source_file = None
    return {
        "name": DEFAULT_FEATURE_EXTRACTOR,
        "class_path": f"{CLaMP2Extractor.__module__}.{CLaMP2Extractor.__qualname__}",
        "module_file": str(Path(source_file).resolve()) if source_file else None,
        "injected": False,
    }


def _checkpoint_identity(*, check_hash: bool) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "name": None,
        "path": None,
        "url": None,
        "exists": None,
        "sha256": None,
        "status": "unavailable",
    }
    try:
        from frechet_music_distance.models.clamp2 import config  # type: ignore
    except Exception as exc:
        identity["error"] = f"{type(exc).__name__}: {exc}"
        return identity
    raw_path = getattr(config, "CLAMP2_WEIGHTS_PATH", None)
    if raw_path is not None:
        checkpoint_path = Path(raw_path).expanduser().resolve()
        identity["path"] = str(checkpoint_path)
        identity["name"] = checkpoint_path.name
        identity["exists"] = checkpoint_path.is_file()
        if check_hash and checkpoint_path.is_file():
            identity["sha256"] = file_sha256(checkpoint_path)
            identity["status"] = "hashed"
        elif checkpoint_path.is_file():
            identity["status"] = "present_not_hashed"
        else:
            identity["status"] = "missing"
    url = getattr(config, "CLAMP2_WEIGHTS_URL", None)
    if isinstance(url, str):
        identity["url"] = url
    return identity


def build_cache_provenance(
    feature_extractor: object | None = None,
    *,
    caller_injected: bool | None = None,
) -> dict[str, Any]:
    """Return extractor/package/checkpoint provenance for cache auditing."""

    if caller_injected is None:
        caller_injected = feature_extractor is not None
    return {
        "frechet_music_distance": {
            "package_version": _distribution_version("frechet-music-distance"),
            "module_file": _module_file("frechet_music_distance"),
        },
        "feature_extractor": _extractor_identity(
            feature_extractor,
            injected=caller_injected,
        ),
        "checkpoint": _checkpoint_identity(check_hash=not caller_injected),
        "gaussian_estimator": {
            "name": DEFAULT_GAUSSIAN_ESTIMATOR,
            "covariance": "np.cov(rowvar=False), unbiased n-1 normalization",
        },
    }


def _nested_value(root: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = root
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _validate_cache_compatibility(
    cached: Mapping[str, Any],
    current: Mapping[str, Any],
    cache_path: Path,
) -> None:
    cached_name = _nested_value(cached, ("feature_extractor", "name"))
    current_name = _nested_value(current, ("feature_extractor", "name"))
    if cached_name != DEFAULT_FEATURE_EXTRACTOR or current_name != DEFAULT_FEATURE_EXTRACTOR:
        raise BootstrapFMDValidationError(
            f"feature cache {cache_path} is not a CLAMP2 cache"
        )
    comparisons = (
        (
            "frechet-music-distance package version",
            ("frechet_music_distance", "package_version"),
        ),
        ("extractor class", ("feature_extractor", "class_path")),
        ("checkpoint SHA256", ("checkpoint", "sha256")),
    )
    for label, path in comparisons:
        cached_value = _nested_value(cached, path)
        current_value = _nested_value(current, path)
        if cached_value is not None and current_value is not None:
            if cached_value != current_value:
                raise BootstrapFMDValidationError(
                    f"feature cache {cache_path} {label} mismatch: "
                    f"{cached_value!r} != {current_value!r}"
                )


def _load_cached_feature_entries(
    cache_path: Path,
    requests: Sequence[FeatureRequest],
    current_provenance: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any] | None]:
    if not cache_path.exists():
        return {}, None
    cache_data = _read_json(cache_path)
    if not isinstance(cache_data, Mapping):
        raise BootstrapFMDValidationError(f"feature cache root must be an object: {cache_path}")
    if cache_data.get("schema_version") != FEATURE_CACHE_SCHEMA_VERSION:
        raise BootstrapFMDValidationError(
            f"feature cache schema_version must be {FEATURE_CACHE_SCHEMA_VERSION}: "
            f"{cache_path}"
        )
    cached_provenance = cache_data.get("provenance")
    if not isinstance(cached_provenance, Mapping):
        raise BootstrapFMDValidationError(f"feature cache provenance is required: {cache_path}")
    _validate_cache_compatibility(cached_provenance, current_provenance, cache_path)
    raw_features = cache_data.get("features")
    if not isinstance(raw_features, Mapping):
        raise BootstrapFMDValidationError(f"feature cache features must be an object: {cache_path}")

    cached_vectors: dict[str, np.ndarray] = {}
    requests_by_id = {request.entry_id: request for request in requests}
    for entry_id, raw_entry in raw_features.items():
        if entry_id not in requests_by_id:
            continue
        context = f"{cache_path}: features[{entry_id!r}]"
        if not isinstance(raw_entry, Mapping):
            raise BootstrapFMDValidationError(f"{context} must be an object")
        request = requests_by_id[entry_id]
        if raw_entry.get("key") != request.key_payload:
            raise BootstrapFMDValidationError(f"{context}.key does not match manifest")
        if raw_entry.get("path") != request.relative_path:
            raise BootstrapFMDValidationError(f"{context}.path does not match manifest")
        if raw_entry.get("sha256") != request.sha256:
            raise BootstrapFMDValidationError(f"{context}.sha256 does not match manifest")
        vector = _coerce_feature_vector(raw_entry.get("vector"), f"{context}.vector")
        cached_vectors[entry_id] = vector
    return cached_vectors, dict(cached_provenance)


def _coerce_feature_vector(value: Any, context: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 1 or array.size == 0:
        raise BootstrapFMDValidationError(f"{context} must be a non-empty 1D vector")
    if not np.isfinite(array).all():
        raise BootstrapFMDValidationError(f"{context} must contain only finite numbers")
    return np.asarray(array, dtype=np.float64)


def _make_clamp2_extractor(verbose: bool) -> object:
    try:
        from frechet_music_distance import FrechetMusicDistance  # type: ignore
    except ImportError as exc:
        raise BootstrapFMDValidationError(
            "frechet-music-distance package is required for CLAMP2 extraction"
        ) from exc
    metric = FrechetMusicDistance(
        feature_extractor=DEFAULT_FEATURE_EXTRACTOR,
        gaussian_estimator=DEFAULT_GAUSSIAN_ESTIMATOR,
        verbose=verbose,
    )
    extractor = getattr(metric, "_feature_extractor", None)
    if extractor is None or not hasattr(extractor, "extract_feature"):
        raise BootstrapFMDValidationError(
            "frechet-music-distance did not expose a CLAMP2 extract_feature hook"
        )
    return extractor


def _extract_feature_vector(
    feature_extractor: object,
    path: Path,
    context: str,
) -> np.ndarray:
    try:
        raw_vector = feature_extractor.extract_feature(path)  # type: ignore[attr-defined]
    except Exception as exc:
        raise BootstrapFMDValidationError(
            f"failed to extract CLAMP2 feature for {context}: {exc}"
        ) from exc
    return _coerce_feature_vector(raw_vector, f"{context} feature")


def _feature_requests(manifest: CommonValidManifest) -> tuple[FeatureRequest, ...]:
    requests: list[FeatureRequest] = []
    for system_id in manifest.system_ids:
        for trial in manifest.trials_by_system[system_id]:
            requests.append(
                FeatureRequest(
                    system_id=trial.system_id,
                    piece_id=trial.piece_id,
                    seed=trial.seed,
                    role="generated",
                    relative_path=trial.generated_relative_path,
                    absolute_path=trial.generated_path,
                    sha256=trial.generated_sha256,
                )
            )
            requests.append(
                FeatureRequest(
                    system_id=trial.system_id,
                    piece_id=trial.piece_id,
                    seed=trial.seed,
                    role="groundtruth",
                    relative_path=trial.groundtruth_relative_path,
                    absolute_path=trial.groundtruth_path,
                    sha256=trial.groundtruth_sha256,
                )
            )
    return tuple(requests)


def load_or_extract_feature_cache(
    manifest: CommonValidManifest,
    cache_path: Path,
    *,
    feature_extractor: object | None = None,
    cache_provenance: Mapping[str, Any] | None = None,
    verbose_extractor: bool = False,
) -> FeatureCacheResult:
    """Load cached CLAMP2 vectors and extract missing MIDI hashes once."""

    cache_path = cache_path.expanduser().resolve()
    requests = _feature_requests(manifest)
    caller_injected_extractor = feature_extractor is not None
    current_provenance = (
        dict(cache_provenance)
        if cache_provenance is not None
        else build_cache_provenance(
            feature_extractor,
            caller_injected=caller_injected_extractor,
        )
    )
    cached_vectors, cached_provenance = _load_cached_feature_entries(
        cache_path,
        requests,
        current_provenance,
    )
    vectors_by_hash: dict[str, np.ndarray] = {}
    for request in requests:
        vector = cached_vectors.get(request.entry_id)
        if vector is None:
            continue
        existing = vectors_by_hash.get(request.sha256)
        if existing is not None and not np.array_equal(existing, vector):
            raise BootstrapFMDValidationError(
                f"feature cache has inconsistent vectors for SHA256 {request.sha256}"
            )
        vectors_by_hash[request.sha256] = vector

    missing_by_hash: dict[str, FeatureRequest] = {}
    for request in requests:
        if request.sha256 not in vectors_by_hash:
            missing_by_hash.setdefault(request.sha256, request)

    if missing_by_hash:
        if feature_extractor is None:
            feature_extractor = _make_clamp2_extractor(verbose_extractor)
        current_provenance = (
            dict(cache_provenance)
            if cache_provenance is not None
            else build_cache_provenance(
                feature_extractor,
                caller_injected=caller_injected_extractor,
            )
        )
        if cached_provenance is not None:
            _validate_cache_compatibility(cached_provenance, current_provenance, cache_path)
        for sha256, request in sorted(missing_by_hash.items()):
            vectors_by_hash[sha256] = _extract_feature_vector(
                feature_extractor,
                request.absolute_path,
                (
                    f"{request.system_id}/{request.piece_id}/seed "
                    f"{request.seed}/{request.role}"
                ),
            )
    else:
        if cached_provenance is not None:
            current_provenance = cached_provenance

    vectors_by_entry_id = {
        request.entry_id: vectors_by_hash[request.sha256]
        for request in requests
    }
    dimensions = {vector.shape[0] for vector in vectors_by_entry_id.values()}
    if len(dimensions) != 1:
        raise BootstrapFMDValidationError(
            f"CLAMP2 feature dimensions differ across MIDI files: {sorted(dimensions)}"
        )

    feature_entries = {
        request.entry_id: {
            "key": request.key_payload,
            "path": request.relative_path,
            "sha256": request.sha256,
            "vector": vectors_by_entry_id[request.entry_id].tolist(),
        }
        for request in sorted(requests, key=lambda item: item.entry_id)
    }
    cache_payload = {
        "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
        "producer": "eval_toolkit.bootstrap_matched_fmd",
        "manifest": {
            "path": str(manifest.path),
            "sha256": manifest.sha256,
        },
        "entry_count": len(feature_entries),
        "unique_midi_hash_count": len(vectors_by_hash),
        "provenance": current_provenance,
        "features": feature_entries,
    }
    _write_json_atomic(cache_path, cache_payload)

    return FeatureCacheResult(
        vectors_by_entry_id=vectors_by_entry_id,
        provenance=current_provenance,
        cache_path=cache_path,
        cache_sha256=file_sha256(cache_path),
        entry_count=len(feature_entries),
        unique_midi_hash_count=len(vectors_by_hash),
        cache_hits=len(cached_vectors),
        cache_misses=len(requests) - len(cached_vectors),
        extracted_midi_count=len(missing_by_hash),
    )


def precompute_sample_space_fmd(x_features: Any, y_features: Any) -> SampleSpaceFMD:
    x = np.asarray(x_features, dtype=np.float64)
    y = np.asarray(y_features, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2:
        raise BootstrapFMDValidationError("feature matrices must be 2D")
    if x.shape[1] != y.shape[1]:
        raise BootstrapFMDValidationError(
            f"feature dimensions differ: {x.shape[1]} != {y.shape[1]}"
        )
    if x.shape[0] < 2 or y.shape[0] < 2:
        raise BootstrapFMDValidationError("FMD requires at least two samples per side")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise BootstrapFMDValidationError("feature matrices must be finite")
    return SampleSpaceFMD(
        x_gram=x @ x.T,
        y_gram=y @ y.T,
        cross_gram=x @ y.T,
    )


def _validate_indices(indices: Any, upper_bound: int, context: str) -> np.ndarray:
    array = np.asarray(indices, dtype=np.int64)
    if array.ndim != 1 or array.size < 2:
        raise BootstrapFMDValidationError(f"{context} must contain at least two indices")
    if np.any(array < 0) or np.any(array >= upper_bound):
        raise BootstrapFMDValidationError(f"{context} contains out-of-range indices")
    return array


def frechet_distance_from_precomputed(
    precomputed: SampleSpaceFMD,
    x_indices: Any | None = None,
    y_indices: Any | None = None,
) -> float:
    """Compute FMD from centered sample-space Gram submatrices.

    Repeated indices are intentional for bootstrap resamples. The selected
    Gram submatrices therefore contain repeated rows and columns, matching the
    sample that would be produced by materializing the repeated feature rows.
    """

    if x_indices is None:
        x_indices = np.arange(precomputed.x_gram.shape[0], dtype=np.int64)
    if y_indices is None:
        y_indices = np.arange(precomputed.y_gram.shape[0], dtype=np.int64)
    x_indices = _validate_indices(
        x_indices,
        precomputed.x_gram.shape[0],
        "x_indices",
    )
    y_indices = _validate_indices(
        y_indices,
        precomputed.y_gram.shape[0],
        "y_indices",
    )
    n = int(x_indices.size)
    m = int(y_indices.size)

    kxx = precomputed.x_gram[np.ix_(x_indices, x_indices)]
    kyy = precomputed.y_gram[np.ix_(y_indices, y_indices)]
    kxy = precomputed.cross_gram[np.ix_(x_indices, y_indices)]

    x_pair_sum = float(kxx.sum())
    y_pair_sum = float(kyy.sum())
    cross_pair_sum = float(kxy.sum())
    mean_difference_squared = (
        x_pair_sum / (n * n)
        + y_pair_sum / (m * m)
        - 2.0 * cross_pair_sum / (n * m)
    )

    x_centered_ss = float(np.trace(kxx)) - x_pair_sum / n
    y_centered_ss = float(np.trace(kyy)) - y_pair_sum / m
    centered_cross = (
        kxy
        - kxy.mean(axis=1, keepdims=True)
        - kxy.mean(axis=0, keepdims=True)
        + kxy.mean()
    )
    nuclear_norm = float(np.linalg.svd(centered_cross, compute_uv=False).sum())
    score = (
        mean_difference_squared
        + x_centered_ss / (n - 1)
        + y_centered_ss / (m - 1)
        - 2.0 * nuclear_norm / math.sqrt((n - 1) * (m - 1))
    )
    return float(score)


def frechet_distance_low_rank(
    x_features: Any,
    y_features: Any,
    *,
    x_indices: Any | None = None,
    y_indices: Any | None = None,
) -> float:
    """Compute unbiased-covariance FMD without a feature-dimensional sqrtm."""

    return frechet_distance_from_precomputed(
        precompute_sample_space_fmd(x_features, y_features),
        x_indices=x_indices,
        y_indices=y_indices,
    )


def _piece_sample_indices(
    key_order: Sequence[tuple[str, str]],
    piece_order: Sequence[str],
) -> dict[str, np.ndarray]:
    indices_by_piece: dict[str, list[int]] = {piece_id: [] for piece_id in piece_order}
    for index, (piece_id, _seed) in enumerate(key_order):
        indices_by_piece[piece_id].append(index)
    return {
        piece_id: np.asarray(indices, dtype=np.int64)
        for piece_id, indices in indices_by_piece.items()
    }


def _sample_indices_for_piece_draw(
    draw: np.ndarray,
    piece_order: Sequence[str],
    indices_by_piece: Mapping[str, np.ndarray],
) -> np.ndarray:
    pieces = [piece_order[int(piece_index)] for piece_index in draw]
    return np.concatenate([indices_by_piece[piece_id] for piece_id in pieces])


def _system_feature_matrices(
    manifest: CommonValidManifest,
    feature_cache: FeatureCacheResult,
    system_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    generated_rows: list[np.ndarray] = []
    groundtruth_rows: list[np.ndarray] = []
    for trial in manifest.trials_by_system[system_id]:
        generated_request = FeatureRequest(
            system_id=trial.system_id,
            piece_id=trial.piece_id,
            seed=trial.seed,
            role="generated",
            relative_path=trial.generated_relative_path,
            absolute_path=trial.generated_path,
            sha256=trial.generated_sha256,
        )
        groundtruth_request = FeatureRequest(
            system_id=trial.system_id,
            piece_id=trial.piece_id,
            seed=trial.seed,
            role="groundtruth",
            relative_path=trial.groundtruth_relative_path,
            absolute_path=trial.groundtruth_path,
            sha256=trial.groundtruth_sha256,
        )
        generated_rows.append(feature_cache.vectors_by_entry_id[generated_request.entry_id])
        groundtruth_rows.append(
            feature_cache.vectors_by_entry_id[groundtruth_request.entry_id]
        )
    return np.vstack(generated_rows), np.vstack(groundtruth_rows)


def _percentile_ci(values: np.ndarray) -> tuple[float, float]:
    return (
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    )


def parse_expected_point_mapping(values: Sequence[str]) -> dict[str, float]:
    mappings: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise BootstrapFMDValidationError(
                f"--expected-point requires SYSTEM_ID=VALUE, found {value!r}"
            )
        system_id, raw_expected = value.split("=", 1)
        system_id = system_id.strip()
        if not system_id:
            raise BootstrapFMDValidationError("--expected-point system ID is empty")
        if system_id in mappings:
            raise BootstrapFMDValidationError(
                f"duplicate --expected-point for system {system_id!r}"
            )
        try:
            expected = float(raw_expected)
        except ValueError as exc:
            raise BootstrapFMDValidationError(
                f"--expected-point for {system_id!r} must be numeric"
            ) from exc
        mappings[system_id] = _finite_number(expected, f"--expected-point {system_id}")
    return mappings


def _validation_report(
    estimates: Mapping[str, float],
    expected_points: Mapping[str, float],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if atol < 0 or rtol < 0:
        raise BootstrapFMDValidationError("validation tolerances must be non-negative")
    system_ids = tuple(sorted(estimates))
    if not expected_points:
        return {
            "status": "not_supplied",
            "source": "--expected-point",
            "atol": atol,
            "rtol": rtol,
            "systems": {
                system_id: {
                    "status": "not_supplied",
                    "estimate": estimates[system_id],
                    "expected_point": None,
                    "absolute_error": None,
                    "tolerance": None,
                }
                for system_id in system_ids
            },
        }

    expected_system_ids = set(expected_points)
    actual_system_ids = set(system_ids)
    if expected_system_ids != actual_system_ids:
        missing = sorted(actual_system_ids - expected_system_ids)
        extra = sorted(expected_system_ids - actual_system_ids)
        raise BootstrapFMDValidationError(
            "if any --expected-point is supplied, exactly one is required for "
            f"every system; missing={missing}, extra={extra}"
        )

    systems: dict[str, Any] = {}
    failures: list[str] = []
    for system_id in system_ids:
        estimate = float(estimates[system_id])
        expected = float(expected_points[system_id])
        absolute_error = abs(estimate - expected)
        tolerance = atol + rtol * abs(expected)
        passed = absolute_error <= tolerance
        if not passed:
            failures.append(
                f"{system_id}: estimate={estimate}, expected={expected}, "
                f"absolute_error={absolute_error}, tolerance={tolerance}"
            )
        systems[system_id] = {
            "status": "passed" if passed else "failed",
            "estimate": estimate,
            "expected_point": expected,
            "absolute_error": absolute_error,
            "tolerance": tolerance,
        }
    if failures:
        raise BootstrapFMDValidationError(
            "low-rank FMD validation failed against supplied official "
            "frechet_music_distance point estimate(s): "
            + "; ".join(failures)
        )
    return {
        "status": "passed",
        "source": "--expected-point",
        "atol": atol,
        "rtol": rtol,
        "systems": systems,
    }


def _validate_expected_point_coverage(
    expected_points: Mapping[str, float],
    system_ids: Sequence[str],
    *,
    require_expected_points: bool,
) -> None:
    if not expected_points:
        if require_expected_points:
            raise BootstrapFMDValidationError(
                "output_json publication requires --expected-point for every system"
            )
        return
    expected_system_ids = set(expected_points)
    actual_system_ids = set(system_ids)
    if expected_system_ids != actual_system_ids:
        missing = sorted(actual_system_ids - expected_system_ids)
        extra = sorted(expected_system_ids - actual_system_ids)
        raise BootstrapFMDValidationError(
            "if any --expected-point is supplied, exactly one is required for "
            f"every system; missing={missing}, extra={extra}"
        )


def bootstrap_matched_fmd(
    *,
    manifest_path: Path,
    output_json: Path | None = None,
    feature_cache_path: Path | None = None,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    expected_points: Mapping[str, float] | None = None,
    validation_atol: float = DEFAULT_VALIDATION_ATOL,
    validation_rtol: float = DEFAULT_VALIDATION_RTOL,
    feature_extractor: object | None = None,
    cache_provenance: Mapping[str, Any] | None = None,
    verbose_extractor: bool = False,
) -> dict[str, Any]:
    bootstrap_replicates = _strict_integer(
        bootstrap_replicates,
        "bootstrap_replicates",
    )
    bootstrap_seed = _strict_integer(bootstrap_seed, "bootstrap_seed")
    validation_atol = _finite_number(validation_atol, "validation_atol")
    validation_rtol = _finite_number(validation_rtol, "validation_rtol")
    if bootstrap_replicates <= 0:
        raise BootstrapFMDValidationError("bootstrap_replicates must be positive")
    manifest = load_common_valid_manifest(manifest_path)
    if len(manifest.key_order) < 2:
        raise BootstrapFMDValidationError(
            "common-valid manifest must contain at least two keys for FMD"
        )
    expected_points = expected_points or {}
    _validate_expected_point_coverage(
        expected_points,
        manifest.system_ids,
        require_expected_points=output_json is not None,
    )
    if feature_cache_path is None:
        feature_cache_path = manifest.root / "clamp2_feature_cache.json"

    feature_cache = load_or_extract_feature_cache(
        manifest,
        feature_cache_path,
        feature_extractor=feature_extractor,
        cache_provenance=cache_provenance,
        verbose_extractor=verbose_extractor,
    )

    rng = np.random.default_rng(bootstrap_seed)
    draws = rng.integers(
        0,
        len(manifest.piece_order),
        size=(bootstrap_replicates, len(manifest.piece_order)),
        dtype=np.int64,
    )
    draw_hash = hashlib.sha256(draws.tobytes(order="C")).hexdigest()
    indices_by_piece = _piece_sample_indices(manifest.key_order, manifest.piece_order)
    full_indices = np.arange(len(manifest.key_order), dtype=np.int64)

    estimates: dict[str, float] = {}
    system_results: dict[str, Any] = {}
    for system_id in manifest.system_ids:
        generated_features, groundtruth_features = _system_feature_matrices(
            manifest,
            feature_cache,
            system_id,
        )
        precomputed = precompute_sample_space_fmd(
            generated_features,
            groundtruth_features,
        )
        estimate = frechet_distance_from_precomputed(
            precomputed,
            x_indices=full_indices,
            y_indices=full_indices,
        )
        replicate_values = np.empty(bootstrap_replicates, dtype=np.float64)
        for replicate_index, draw in enumerate(draws):
            sample_indices = _sample_indices_for_piece_draw(
                draw,
                manifest.piece_order,
                indices_by_piece,
            )
            replicate_values[replicate_index] = frechet_distance_from_precomputed(
                precomputed,
                x_indices=sample_indices,
                y_indices=sample_indices,
            )
        ci_low, ci_high = _percentile_ci(replicate_values)
        estimates[system_id] = estimate
        system_results[system_id] = {
            "estimate": estimate,
            "ci": {
                "low": ci_low,
                "high": ci_high,
                "confidence_level": 0.95,
                "method": "percentile",
            },
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "scope": {
                "dataset": "common_valid",
                "metric": "CLAMP2 Frechet Music Distance",
                "comparison": "common_generated_midi_vs_common_metric_gt_midi",
                "cluster": "piece_id",
                "preserve_all_seeds_per_sampled_piece": True,
                "covariance": "np.cov(rowvar=False), unbiased n-1",
            },
            "sample_count": {
                "generated": int(generated_features.shape[0]),
                "groundtruth": int(groundtruth_features.shape[0]),
            },
            "feature_dimension": int(generated_features.shape[1]),
            "replicate_estimates_sha256": hashlib.sha256(
                replicate_values.tobytes(order="C")
            ).hexdigest(),
        }

    validation = _validation_report(
        estimates,
        expected_points or {},
        atol=validation_atol,
        rtol=validation_rtol,
    )
    for system_id in manifest.system_ids:
        system_results[system_id]["validation"] = validation["systems"][system_id]

    cache_provenance_summary = {
        "feature_cache_path": str(feature_cache.cache_path),
        "feature_cache_sha256": feature_cache.cache_sha256,
        "entry_count": feature_cache.entry_count,
        "unique_midi_hash_count": feature_cache.unique_midi_hash_count,
        "cache_hits": feature_cache.cache_hits,
        "cache_misses": feature_cache.cache_misses,
        "extracted_midi_count": feature_cache.extracted_midi_count,
        "provenance": feature_cache.provenance,
    }
    summary = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "manifest_path": str(manifest.path),
        "manifest_sha256": manifest.sha256,
        "system_ids": list(manifest.system_ids),
        "scope": {
            "dataset": "common_valid",
            "key_fields": ["piece_id", "seed"],
            "common_valid_key_count": len(manifest.key_order),
            "piece_count": len(manifest.piece_order),
            "feature_extractor": DEFAULT_FEATURE_EXTRACTOR,
            "gaussian_estimator": DEFAULT_GAUSSIAN_ESTIMATOR,
            "official_point_estimate_compatibility": (
                "frechet_music_distance score(reference_dir, test_dir) "
                "with np.cov(rowvar=False)"
            ),
        },
        "bootstrap": {
            "method": "matched_piece_cluster_percentile",
            "cluster": "piece_id",
            "preserve_all_seeds_per_sampled_piece": True,
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "percentiles": [2.5, 97.5],
            "piece_order": list(manifest.piece_order),
            "shared_draw_matrix_sha256": draw_hash,
        },
        "validation": validation,
        "cache_provenance": cache_provenance_summary,
        "systems": system_results,
    }
    if output_json is not None:
        _write_json_atomic(output_json, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute percentile-bootstrap CLAMP2 FMD confidence intervals "
            "from common_valid/manifest.json."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("common_valid") / "manifest.json",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=None,
        help="Auditable CLAMP2 feature cache JSON; defaults next to the manifest.",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--expected-point",
        action="append",
        default=[],
        metavar="SYSTEM_ID=VALUE",
        help=(
            "Official frechet_music_distance point estimate. If supplied once, "
            "it must be supplied exactly once for every system."
        ),
    )
    parser.add_argument("--validation-atol", type=float, default=DEFAULT_VALIDATION_ATOL)
    parser.add_argument("--validation-rtol", type=float, default=DEFAULT_VALIDATION_RTOL)
    parser.add_argument("--verbose-extractor", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        expected_points = parse_expected_point_mapping(args.expected_point)
        summary = bootstrap_matched_fmd(
            manifest_path=args.manifest,
            output_json=args.output_json,
            feature_cache_path=args.feature_cache,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
            expected_points=expected_points,
            validation_atol=args.validation_atol,
            validation_rtol=args.validation_rtol,
            verbose_extractor=args.verbose_extractor,
        )
    except BootstrapFMDValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"Wrote CLAMP2 FMD bootstrap CIs for {len(summary['systems'])} system(s) "
        f"to {args.output_json.expanduser().resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
