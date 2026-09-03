from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np
import scipy.linalg

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval_toolkit.bootstrap_matched_fmd import (
    BootstrapFMDValidationError,
    bootstrap_matched_fmd,
    frechet_distance_from_precomputed,
    frechet_distance_low_rank,
    load_common_valid_manifest,
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


class FakeExtractor:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def extract_feature(self, path: Path) -> np.ndarray:
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        self.calls.append(digest)
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
                generated.write_bytes(f"generated:{system_id}:{piece_id}:{seed}".encode())
                groundtruth.write_bytes(f"groundtruth:{piece_id}:{seed}".encode())
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

    def _unique_hashes(self, manifest: Path) -> set[str]:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        hashes: set[str] = set()
        for row in payload["trials"]:
            hashes.add(row["generated_sha256"])
            hashes.add(row["metric_gt_sha256"])
        return hashes

    def test_deterministic_clustered_ci_and_feature_cache_reuse(self) -> None:
        manifest = self._write_manifest()
        cache = self.root / "features.json"
        output = self.root / "fmd.json"
        extractor = FakeExtractor()

        first = bootstrap_matched_fmd(
            manifest_path=manifest,
            output_json=output,
            feature_cache_path=cache,
            bootstrap_replicates=40,
            bootstrap_seed=5,
            feature_extractor=extractor,
            cache_provenance=TEST_PROVENANCE,
        )

        self.assertTrue(output.is_file())
        self.assertTrue(cache.is_file())
        expected_hash_counts = Counter({digest: 1 for digest in self._unique_hashes(manifest)})
        self.assertEqual(Counter(extractor.calls), expected_hash_counts)
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

        second = bootstrap_matched_fmd(
            manifest_path=manifest,
            output_json=self.root / "fmd-second.json",
            feature_cache_path=cache,
            bootstrap_replicates=40,
            bootstrap_seed=5,
            feature_extractor=FailingExtractor(),
            cache_provenance=TEST_PROVENANCE,
        )

        self.assertEqual(first["bootstrap"], second["bootstrap"])
        self.assertEqual(first["systems"], second["systems"])
        self.assertEqual(second["cache_provenance"]["cache_hits"], 16)
        self.assertEqual(second["cache_provenance"]["extracted_midi_count"], 0)

    def test_expected_point_validation_passes_requires_complete_and_fails_closed(
        self,
    ) -> None:
        manifest = self._write_manifest()
        cache = self.root / "features.json"
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

        missing_output = self.root / "missing-expected.json"
        with self.assertRaisesRegex(BootstrapFMDValidationError, "exactly one"):
            bootstrap_matched_fmd(
                manifest_path=manifest,
                output_json=missing_output,
                feature_cache_path=cache,
                bootstrap_replicates=10,
                bootstrap_seed=7,
                expected_points={self.SYSTEMS[0]: expected[self.SYSTEMS[0]]},
                feature_extractor=FailingExtractor(),
                cache_provenance=TEST_PROVENANCE,
            )
        self.assertFalse(missing_output.exists())

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
