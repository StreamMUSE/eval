from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from unittest import mock

import numpy as np
import pretty_midi
import scipy.linalg

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluate_accompaniment_metrics import (  # noqa: E402
    _build_accompaniment_only_midi as official_generated_accompaniment_midi,
)
from evaluate_accompaniment_metrics import (  # noqa: E402
    _build_ground_truth_accompaniment_midi as official_groundtruth_accompaniment_midi,
)
from eval_toolkit.bootstrap_matched_fmd import (
    BootstrapFMDValidationError,
    FEATURE_CACHE_SCHEMA_VERSION,
    FEATURE_PREPARATION_CONTRACT,
    bootstrap_matched_fmd,
    frechet_distance_from_precomputed,
    frechet_distance_low_rank,
    load_common_valid_manifest,
    main,
    precompute_sample_space_fmd,
)


TEST_PROVENANCE = {
    "frechet_music_distance": {
        "package_version": "test",
        "module_file": None,
    },
    "feature_extractor": {
        "name": "clamp2",
        "class_path": "tests.test_bootstrap_matched_fmd.FakeExtractor",
        "module_file": None,
        "injected": True,
    },
    "checkpoint": {
        "name": "fake-checkpoint",
        "path": None,
        "url": None,
        "exists": True,
        "sha256": "f" * 64,
        "status": "test",
    },
    "gaussian_estimator": {
        "name": "mle",
        "covariance": "np.cov(rowvar=False), unbiased n-1 normalization",
    },
}


def _vector_from_sha256(value: str) -> np.ndarray:
    words = [int(value[index:index + 8], 16) for index in range(0, 32, 8)]
    return np.asarray([(word % 1000) / 97.0 for word in words], dtype=np.float64)


def _midi_summary(path: Path) -> list[dict[str, object]]:
    midi = pretty_midi.PrettyMIDI(str(path))
    return [
        {
            "name": instrument.name,
            "program": instrument.program,
            "is_drum": instrument.is_drum,
            "notes": [
                (
                    note.pitch,
                    round(note.start, 6),
                    round(note.end, 6),
                    note.velocity,
                )
                for note in instrument.notes
            ],
        }
        for instrument in midi.instruments
    ]


class FakeExtractor:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.paths: list[Path] = []
        self.track_summaries: dict[str, list[dict[str, object]]] = {}

    def extract_feature(self, path: Path) -> np.ndarray:
        path = Path(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.calls.append(digest)
        self.paths.append(path)
        self.track_summaries[digest] = _midi_summary(path)
        return _vector_from_sha256(digest)[None, :]


class FailingExtractor:
    def extract_feature(self, path: Path) -> np.ndarray:
        raise AssertionError(f"unexpected extraction for {path}")


def _direct_sqrtm_fmd(x_features: np.ndarray, y_features: np.ndarray) -> float:
    mean_x = np.mean(x_features, axis=0)
    mean_y = np.mean(y_features, axis=0)
    cov_x = np.cov(x_features, rowvar=False)
    cov_y = np.cov(y_features, rowvar=False)
    covmean = scipy.linalg.sqrtm(cov_x.dot(cov_y))
    if np.iscomplexobj(covmean):
        if not np.allclose(covmean.imag, 0.0, atol=1e-8):
            raise AssertionError("sqrtm produced a non-negligible imaginary part")
        covmean = covmean.real
    diff = mean_x - mean_y
    return float(diff.dot(diff) + np.trace(cov_x) + np.trace(cov_y) - 2.0 * np.trace(covmean))


class LowRankFMDTest(unittest.TestCase):
    def test_low_rank_formula_matches_direct_sqrtm(self) -> None:
        rng = np.random.default_rng(1234)
        x = rng.normal(size=(9, 4))
        y = rng.normal(size=(11, 4)) + 0.25
        cases = (
            ("full", None, None),
            (
                "repeated-bootstrap-indices",
                np.asarray([0, 2, 2, 5, 7, 7, 1], dtype=np.int64),
                np.asarray([3, 3, 8, 1, 4, 4, 2, 10], dtype=np.int64),
            ),
            (
                "unequal-sample-counts",
                np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64),
                np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64),
            ),
        )

        precomputed = precompute_sample_space_fmd(x, y)
        for name, x_indices, y_indices in cases:
            with self.subTest(name=name):
                selected_x = x if x_indices is None else x[x_indices]
                selected_y = y if y_indices is None else y[y_indices]
                direct = _direct_sqrtm_fmd(selected_x, selected_y)
                low_rank = frechet_distance_low_rank(
                    x,
                    y,
                    x_indices=x_indices,
                    y_indices=y_indices,
                )
                precomputed_score = frechet_distance_from_precomputed(
                    precomputed,
                    x_indices=x_indices,
                    y_indices=y_indices,
                )
                self.assertLess(abs(low_rank - direct), 1e-8)
                self.assertLess(abs(precomputed_score - direct), 1e-8)

    def test_low_rank_matches_package_mle_compute_fmd_for_identical_matrix(
        self,
    ) -> None:
        from frechet_music_distance import FrechetMusicDistance
        from frechet_music_distance.gaussian_estimators import MaxLikelihoodEstimator

        class IdentityExtractor:
            pass

        rng = np.random.default_rng(20260203)
        features = rng.normal(size=(7, 11))
        metric = FrechetMusicDistance(
            feature_extractor=IdentityExtractor(),
            gaussian_estimator=MaxLikelihoodEstimator(),
            verbose=False,
        )
        mean, covariance = metric._gaussian_estimator.estimate_parameters(features)

        self.assertTrue(np.array_equal(mean, np.mean(features, axis=0)))
        self.assertTrue(np.array_equal(covariance, np.cov(features, rowvar=False)))

        package_score = metric._compute_fmd(mean, mean, covariance, covariance)
        low_rank = frechet_distance_low_rank(features, features)

        self.assertLess(abs(package_score - low_rank), 6e-5)
        self.assertLess(abs(low_rank), 6e-5)


class BootstrapMatchedFMDTest(unittest.TestCase):
    SYSTEMS = ("system_a", "system_b")
    PIECES = ("piece_1", "piece_2")
    SEEDS = ("0", "1")

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @classmethod
    def _instrument(
        cls,
        *,
        name: str,
        program: int,
        notes: list[tuple[int, float, float]],
        is_drum: bool = False,
    ) -> pretty_midi.Instrument:
        instrument = pretty_midi.Instrument(
            program=program,
            is_drum=is_drum,
            name=name,
        )
        instrument.notes = [
            pretty_midi.Note(
                velocity=80,
                pitch=pitch,
                start=start,
                end=end,
            )
            for pitch, start, end in notes
        ]
        return instrument

    @classmethod
    def _write_generated_midi(
        cls,
        path: Path,
        *,
        system_id: str,
        piece_id: str,
        seed: str,
    ) -> None:
        system_offset = cls.SYSTEMS.index(system_id)
        piece_offset = cls.PIECES.index(piece_id)
        seed_offset = cls.SEEDS.index(seed)
        midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        midi.instruments = [
            cls._instrument(
                name="MeLoDy",
                program=40,
                notes=[(72 + piece_offset * 4 + seed_offset, 0.0, 1.0)],
            ),
            cls._instrument(
                name="Accompaniment",
                program=0,
                notes=[
                    (
                        48 + system_offset * 16 + piece_offset * 4 + seed_offset,
                        1.0,
                        2.0,
                    )
                ],
            ),
            cls._instrument(
                name="Drums",
                program=0,
                notes=[(36, 1.5, 1.75)],
                is_drum=True,
            ),
        ]
        midi.write(str(path))

    @classmethod
    def _write_groundtruth_midi(
        cls,
        path: Path,
        *,
        piece_id: str,
        seed: str,
    ) -> None:
        piece_offset = cls.PIECES.index(piece_id)
        midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        if seed == "0":
            midi.instruments = [
                cls._instrument(
                    name="Piano",
                    program=0,
                    notes=[(50 + piece_offset, 1.0, 2.0)],
                ),
                cls._instrument(
                    name="Strings",
                    program=48,
                    notes=[(62 + piece_offset, 1.25, 1.75)],
                ),
                cls._instrument(
                    name="Drums",
                    program=0,
                    notes=[(38, 1.5, 1.75)],
                    is_drum=True,
                ),
            ]
        else:
            midi.instruments = [
                cls._instrument(
                    name="Strings",
                    program=48,
                    notes=[(55 + piece_offset, 1.0, 2.0)],
                ),
                cls._instrument(
                    name="Guitar",
                    program=24,
                    notes=[(67 + piece_offset, 1.5, 2.25)],
                ),
                cls._instrument(
                    name="Drums",
                    program=0,
                    notes=[(38, 1.5, 1.75)],
                    is_drum=True,
                ),
            ]
        midi.write(str(path))

    def _write_manifest(
        self,
        *,
        mutate: Callable[[dict[str, object], Path], None] | None = None,
    ) -> Path:
        common_valid = self.root / "common_valid"
        rows: list[dict[str, str]] = []
        keys = [(piece_id, seed) for piece_id in self.PIECES for seed in self.SEEDS]
        for system_id in self.SYSTEMS:
            for piece_id, seed in keys:
                basename = f"{piece_id}__seed-{seed}.mid"
                generated_relative = f"{system_id}/generated/{basename}"
                groundtruth_relative = f"{system_id}/groundtruth/{basename}"
                generated = common_valid / generated_relative
                groundtruth = common_valid / groundtruth_relative
                generated.parent.mkdir(parents=True, exist_ok=True)
                groundtruth.parent.mkdir(parents=True, exist_ok=True)
                self._write_generated_midi(
                    generated,
                    system_id=system_id,
                    piece_id=piece_id,
                    seed=seed,
                )
                self._write_groundtruth_midi(
                    groundtruth,
                    piece_id=piece_id,
                    seed=seed,
                )
                rows.append(
                    {
                        "piece_id": piece_id,
                        "seed": seed,
                        "system_id": system_id,
                        "basename": basename,
                        "source_generated_midi": "",
                        "source_metric_gt_midi": "",
                        "generated_sha256": self._sha256(generated),
                        "metric_gt_sha256": self._sha256(groundtruth),
                        "common_generated_midi": generated_relative,
                        "common_metric_gt_midi": groundtruth_relative,
                    }
                )
        payload: dict[str, object] = {
            "schema_version": 1,
            "audit_path": str(self.root / "audit.csv"),
            "audit_sha256": "a" * 64,
            "system_ids": list(self.SYSTEMS),
            "key_fields": ["piece_id", "seed"],
            "common_valid_key_count": len(keys),
            "common_valid_keys": [
                {"piece_id": piece_id, "seed": seed}
                for piece_id, seed in keys
            ],
            "systems": [],
            "trials": rows,
        }
        if mutate is not None:
            mutate(payload, common_valid)
        manifest = common_valid / "manifest.json"
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        return manifest

    def _manifest_rows(self, manifest: Path) -> list[dict[str, str]]:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        return payload["trials"]

    @staticmethod
    def _official_prepared_digest_and_summary(
        source: Path,
        role: str,
        destination: Path,
    ) -> tuple[str, list[dict[str, object]]]:
        midi = pretty_midi.PrettyMIDI(str(source))
        if role == "generated":
            prepared = official_generated_accompaniment_midi(
                midi,
                ("melody",),
                (),
                (),
                False,
                False,
            )
        else:
            prepared = official_groundtruth_accompaniment_midi(midi, False)
        prepared.write(str(destination))
        return (
            hashlib.sha256(destination.read_bytes()).hexdigest(),
            _midi_summary(destination),
        )

    def _official_prepared_records(
        self,
        manifest: Path,
    ) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            for index, row in enumerate(self._manifest_rows(manifest)):
                for role, path_field, hash_field in (
                    ("generated", "common_generated_midi", "generated_sha256"),
                    ("groundtruth", "common_metric_gt_midi", "metric_gt_sha256"),
                ):
                    source = manifest.parent / row[path_field]
                    destination = temporary / f"{index}-{role}.mid"
                    digest, summary = self._official_prepared_digest_and_summary(
                        source,
                        role,
                        destination,
                    )
                    records.append(
                        {
                            "system_id": row["system_id"],
                            "piece_id": row["piece_id"],
                            "seed": row["seed"],
                            "role": role,
                            "source_path": row[path_field],
                            "source_sha256": row[hash_field],
                            "digest": digest,
                            "summary": summary,
                        }
                    )
        return records

    def test_deterministic_clustered_ci_and_feature_cache_reuse(self) -> None:
        manifest = self._write_manifest()
        cache = self.root / "features.json"
        output = self.root / "fmd.json"
        extractor = FakeExtractor()

        first = bootstrap_matched_fmd(
            manifest_path=manifest,
            feature_cache_path=cache,
            bootstrap_replicates=40,
            bootstrap_seed=5,
            feature_extractor=extractor,
            cache_provenance=TEST_PROVENANCE,
        )

        self.assertFalse(output.exists())
        self.assertTrue(cache.is_file())
        official_records = self._official_prepared_records(manifest)
        expected_hash_counts = Counter(
            {
                record["digest"]: 1
                for record in official_records
            }
        )
        self.assertEqual(Counter(extractor.calls), expected_hash_counts)
        self.assertEqual(first["cache_provenance"]["extracted_midi_count"], 12)
        self.assertEqual(
            first["cache_provenance"]["unique_prepared_midi_hash_count"],
            12,
        )
        draws = np.random.default_rng(5).integers(
            0,
            len(self.PIECES),
            size=(40, len(self.PIECES)),
            dtype=np.int64,
        )
        self.assertEqual(
            first["bootstrap"]["shared_draw_matrix_sha256"],
            hashlib.sha256(draws.tobytes(order="C")).hexdigest(),
        )
        self.assertEqual(first["bootstrap"]["replicates"], 40)
        self.assertEqual(first["bootstrap"]["seed"], 5)
        self.assertTrue(
            first["bootstrap"]["preserve_all_seeds_per_sampled_piece"]
        )
        for system_id in self.SYSTEMS:
            result = first["systems"][system_id]
            self.assertEqual(result["replicates"], 40)
            self.assertEqual(result["seed"], 5)
            self.assertEqual(result["ci"]["confidence_level"], 0.95)
            self.assertLessEqual(result["ci"]["low"], result["ci"]["high"])
            self.assertEqual(result["validation"]["status"], "not_supplied")

        expected = {
            system_id: first["systems"][system_id]["estimate"]
            for system_id in self.SYSTEMS
        }
        published = bootstrap_matched_fmd(
            manifest_path=manifest,
            output_json=output,
            feature_cache_path=cache,
            bootstrap_replicates=40,
            bootstrap_seed=5,
            expected_points=expected,
            feature_extractor=FailingExtractor(),
            cache_provenance=TEST_PROVENANCE,
        )
        self.assertTrue(output.is_file())
        for system_id in self.SYSTEMS:
            self.assertEqual(
                published["systems"][system_id]["validation"]["status"],
                "passed",
            )

        second = bootstrap_matched_fmd(
            manifest_path=manifest,
            output_json=self.root / "fmd-second.json",
            feature_cache_path=cache,
            bootstrap_replicates=40,
            bootstrap_seed=5,
            expected_points=expected,
            feature_extractor=FailingExtractor(),
            cache_provenance=TEST_PROVENANCE,
        )

        self.assertEqual(published["bootstrap"], second["bootstrap"])
        self.assertEqual(published["systems"], second["systems"])
        self.assertEqual(published["cache_provenance"]["cache_hits"], 16)
        self.assertEqual(second["cache_provenance"]["cache_hits"], 16)
        self.assertEqual(second["cache_provenance"]["extracted_midi_count"], 0)

    def test_extractor_receives_exact_official_accompaniment_only_midis(
        self,
    ) -> None:
        manifest = self._write_manifest()
        extractor = FakeExtractor()

        bootstrap_matched_fmd(
            manifest_path=manifest,
            feature_cache_path=self.root / "features.json",
            bootstrap_replicates=8,
            bootstrap_seed=11,
            feature_extractor=extractor,
            cache_provenance=TEST_PROVENANCE,
        )

        official_records = self._official_prepared_records(manifest)
        summaries_by_digest = {
            record["digest"]: record["summary"]
            for record in official_records
        }
        self.assertEqual(
            Counter(extractor.calls),
            Counter({digest: 1 for digest in summaries_by_digest}),
        )
        self.assertEqual(extractor.track_summaries, summaries_by_digest)
        self.assertTrue(all(path.suffix == ".mid" for path in extractor.paths))

        for record in official_records:
            names = [
                instrument["name"]
                for instrument in record["summary"]  # type: ignore[index]
            ]
            drums = [
                instrument["is_drum"]
                for instrument in record["summary"]  # type: ignore[index]
            ]
            with self.subTest(
                role=record["role"],
                seed=record["seed"],
                source=record["source_path"],
            ):
                self.assertFalse(any(drums))
                if record["role"] == "generated":
                    self.assertEqual(names, ["Accompaniment"])
                elif record["seed"] == "0":
                    self.assertEqual(names, ["Piano"])
                else:
                    self.assertEqual(names, ["Strings", "Guitar"])

    def test_cache_records_prepared_metadata_and_rejects_stale_cache(self) -> None:
        manifest = self._write_manifest()
        cache = self.root / "features.json"

        bootstrap_matched_fmd(
            manifest_path=manifest,
            feature_cache_path=cache,
            bootstrap_replicates=8,
            bootstrap_seed=13,
            feature_extractor=FakeExtractor(),
            cache_provenance=TEST_PROVENANCE,
        )

        cache_data = json.loads(cache.read_text(encoding="utf-8"))
        self.assertEqual(cache_data["schema_version"], FEATURE_CACHE_SCHEMA_VERSION)
        self.assertEqual(cache_data["preparation"], FEATURE_PREPARATION_CONTRACT)
        self.assertEqual(
            cache_data["provenance"]["preparation"],
            FEATURE_PREPARATION_CONTRACT,
        )
        self.assertEqual(cache_data["entry_count"], 16)
        self.assertEqual(cache_data["unique_prepared_midi_hash_count"], 12)
        self.assertEqual(len(cache_data["features"]), 16)
        for entry in cache_data["features"].values():
            self.assertEqual(set(entry["source"]), {"path", "sha256"})
            self.assertEqual(entry["preparation"]["role"], entry["key"]["role"])
            self.assertIn("role_contract", entry["preparation"])
            self.assertRegex(entry["prepared_midi"]["sha256"], r"^[0-9a-f]{64}$")
            self.assertIn("vector", entry)

        stale_cache = json.loads(json.dumps(cache_data))
        stale_cache["schema_version"] = 1
        cache.write_text(json.dumps(stale_cache), encoding="utf-8")
        with self.assertRaisesRegex(BootstrapFMDValidationError, "schema_version"):
            bootstrap_matched_fmd(
                manifest_path=manifest,
                feature_cache_path=cache,
                bootstrap_replicates=8,
                bootstrap_seed=13,
                feature_extractor=FailingExtractor(),
                cache_provenance=TEST_PROVENANCE,
            )

        incompatible_cache = json.loads(json.dumps(cache_data))
        incompatible_cache["preparation"]["generated"]["melody_track_names"] = [
            "melody",
            "lead",
        ]
        cache.write_text(json.dumps(incompatible_cache), encoding="utf-8")
        with self.assertRaisesRegex(BootstrapFMDValidationError, "preparation"):
            bootstrap_matched_fmd(
                manifest_path=manifest,
                feature_cache_path=cache,
                bootstrap_replicates=8,
                bootstrap_seed=13,
                feature_extractor=FailingExtractor(),
                cache_provenance=TEST_PROVENANCE,
            )

    def test_expected_point_validation_passes_requires_complete_and_fails_closed(
        self,
    ) -> None:
        manifest = self._write_manifest()
        cache = self.root / "features.json"
        unvalidated_output = self.root / "unvalidated.json"
        unvalidated_cache = self.root / "unvalidated-cache.json"
        with self.assertRaisesRegex(BootstrapFMDValidationError, "requires"):
            bootstrap_matched_fmd(
                manifest_path=manifest,
                output_json=unvalidated_output,
                feature_cache_path=unvalidated_cache,
                bootstrap_replicates=10,
                bootstrap_seed=7,
                feature_extractor=FailingExtractor(),
                cache_provenance=TEST_PROVENANCE,
            )
        self.assertFalse(unvalidated_output.exists())
        self.assertFalse(unvalidated_cache.exists())

        incomplete_output = self.root / "incomplete-expected.json"
        incomplete_cache = self.root / "incomplete-cache.json"
        with self.assertRaisesRegex(BootstrapFMDValidationError, "exactly one"):
            bootstrap_matched_fmd(
                manifest_path=manifest,
                output_json=incomplete_output,
                feature_cache_path=incomplete_cache,
                bootstrap_replicates=10,
                bootstrap_seed=7,
                expected_points={self.SYSTEMS[0]: 0.0},
                feature_extractor=FailingExtractor(),
                cache_provenance=TEST_PROVENANCE,
            )
        self.assertFalse(incomplete_output.exists())
        self.assertFalse(incomplete_cache.exists())

        baseline = bootstrap_matched_fmd(
            manifest_path=manifest,
            feature_cache_path=cache,
            bootstrap_replicates=10,
            bootstrap_seed=7,
            feature_extractor=FakeExtractor(),
            cache_provenance=TEST_PROVENANCE,
        )
        expected = {
            system_id: baseline["systems"][system_id]["estimate"]
            for system_id in self.SYSTEMS
        }

        passed = bootstrap_matched_fmd(
            manifest_path=manifest,
            output_json=self.root / "validated.json",
            feature_cache_path=cache,
            bootstrap_replicates=10,
            bootstrap_seed=7,
            expected_points=expected,
            feature_extractor=FailingExtractor(),
            cache_provenance=TEST_PROVENANCE,
        )
        self.assertEqual(passed["validation"]["status"], "passed")

        bad_expected = dict(expected)
        bad_expected[self.SYSTEMS[0]] += 1.0
        bad_output = self.root / "bad-expected.json"
        with self.assertRaisesRegex(BootstrapFMDValidationError, "validation failed"):
            bootstrap_matched_fmd(
                manifest_path=manifest,
                output_json=bad_output,
                feature_cache_path=cache,
                bootstrap_replicates=10,
                bootstrap_seed=7,
                expected_points=bad_expected,
                validation_atol=0.0,
                validation_rtol=0.0,
                feature_extractor=FailingExtractor(),
                cache_provenance=TEST_PROVENANCE,
            )
        self.assertFalse(bad_output.exists())

    def test_cli_refuses_unvalidated_publish_before_cache_or_extraction(self) -> None:
        manifest = self._write_manifest()
        output = self.root / "cli-unvalidated.json"
        cache = self.root / "cli-cache.json"

        exit_code = main(
            [
                "--manifest",
                str(manifest),
                "--output-json",
                str(output),
                "--feature-cache",
                str(cache),
                "--bootstrap-replicates",
                "10",
            ]
        )

        self.assertEqual(exit_code, 2)
        self.assertFalse(output.exists())
        self.assertFalse(cache.exists())

    def test_default_extractor_provenance_hashes_checkpoint_when_obtainable(
        self,
    ) -> None:
        manifest = self._write_manifest()
        checkpoint_calls: list[bool] = []
        identity_calls: list[bool] = []

        def checkpoint_identity(*, check_hash: bool) -> dict[str, object]:
            checkpoint_calls.append(check_hash)
            return {
                "name": "weights.pth",
                "path": str(self.root / "weights.pth"),
                "url": "https://example.invalid/weights.pth",
                "exists": True,
                "sha256": "c" * 64 if check_hash else None,
                "status": "hashed" if check_hash else "present_not_hashed",
            }

        def extractor_identity(
            feature_extractor: object | None,
            *,
            injected: bool,
        ) -> dict[str, object]:
            identity_calls.append(injected)
            class_path = (
                "default.CLaMP2Extractor"
                if feature_extractor is None
                else "default.CreatedCLaMP2Extractor"
            )
            return {
                "name": "clamp2",
                "class_path": class_path,
                "module_file": None,
                "injected": injected,
            }

        with mock.patch(
            "eval_toolkit.bootstrap_matched_fmd._make_clamp2_extractor",
            return_value=FakeExtractor(),
        ), mock.patch(
            "eval_toolkit.bootstrap_matched_fmd._checkpoint_identity",
            side_effect=checkpoint_identity,
        ), mock.patch(
            "eval_toolkit.bootstrap_matched_fmd._extractor_identity",
            side_effect=extractor_identity,
        ), mock.patch(
            "eval_toolkit.bootstrap_matched_fmd._distribution_version",
            return_value="1.0.0-test",
        ), mock.patch(
            "eval_toolkit.bootstrap_matched_fmd._module_file",
            return_value=None,
        ):
            summary = bootstrap_matched_fmd(
                manifest_path=manifest,
                feature_cache_path=self.root / "default-cache.json",
                bootstrap_replicates=5,
                bootstrap_seed=3,
            )

        provenance = summary["cache_provenance"]["provenance"]
        self.assertFalse(provenance["feature_extractor"]["injected"])
        self.assertEqual(provenance["checkpoint"]["status"], "hashed")
        self.assertEqual(provenance["checkpoint"]["sha256"], "c" * 64)
        self.assertTrue(checkpoint_calls)
        self.assertTrue(all(checkpoint_calls))
        self.assertTrue(identity_calls)
        self.assertFalse(any(identity_calls))

    def test_manifest_validation_is_strict(self) -> None:
        def remove_one_system_key(payload: dict[str, object], _root: Path) -> None:
            payload["trials"] = [
                row
                for row in payload["trials"]  # type: ignore[index]
                if not (
                    row["system_id"] == self.SYSTEMS[1]
                    and row["piece_id"] == self.PIECES[1]
                    and row["seed"] == self.SEEDS[1]
                )
            ]

        def corrupt_hash(payload: dict[str, object], _root: Path) -> None:
            payload["trials"][0]["generated_sha256"] = "0" * 64  # type: ignore[index]

        def mismatch_basename(payload: dict[str, object], root: Path) -> None:
            row = payload["trials"][0]  # type: ignore[index]
            original = root / row["common_metric_gt_midi"]
            replacement = original.with_name("different.mid")
            replacement.write_bytes(original.read_bytes())
            row["common_metric_gt_midi"] = str(replacement.relative_to(root))

        def use_absolute_path(payload: dict[str, object], root: Path) -> None:
            row = payload["trials"][0]  # type: ignore[index]
            row["common_generated_midi"] = str((root / row["common_generated_midi"]).resolve())

        cases = (
            ("key-set", remove_one_system_key, "exact common key set"),
            ("hash", corrupt_hash, "SHA256 mismatch"),
            ("basename", mismatch_basename, "basenames differ"),
            ("relative", use_absolute_path, "must be relative"),
        )
        for name, mutate, message in cases:
            with self.subTest(name=name):
                manifest = self._write_manifest(mutate=mutate)
                with self.assertRaisesRegex(BootstrapFMDValidationError, message):
                    load_common_valid_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
