from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval_toolkit.summarize_matched_music_metrics import (
    MusicSummaryValidationError,
    load_matched_audit,
    main,
    parse_metrics_mapping,
    summarize_matched_music_metrics,
)


class SummarizeMatchedMusicMetricsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_audit(
        self,
        valid_by_system: dict[str, dict[tuple[str, str], bool]],
        *,
        name: str = "audit.csv",
        extra_rows: list[dict[str, str]] | None = None,
    ) -> Path:
        path = self.root / name
        rows = []
        for system_id, values in valid_by_system.items():
            for (piece_id, seed), valid_output in values.items():
                rows.append(
                    {
                        "piece_id": piece_id,
                        "seed": seed,
                        "system_id": system_id,
                        "source_status": "complete",
                        "preparation_status": "prepared",
                        "valid_output": str(valid_output),
                    }
                )
        rows.extend(extra_rows or [])
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "piece_id",
                    "seed",
                    "system_id",
                    "source_status",
                    "preparation_status",
                    "valid_output",
                ),
            )
            writer.writeheader()
            writer.writerows(rows)
        return path

    @staticmethod
    def _detail(
        piece_id: str,
        seed: str,
        value: float,
    ) -> dict[str, object]:
        return {
            "piece": f"piece-{piece_id}__seed-{seed}",
            "pitch_jsd": value,
            "onset_jsd": value + 0.1,
            "duration_jsd": value + 0.2,
            "harmonicity": {
                "consonant_ratio": 0.8 - value / 10.0,
                "unsupported_ratio": 0.1 + value / 20.0,
            },
        }

    def _write_metrics(
        self,
        path: Path,
        details: list[dict[str, object]],
        *,
        pairs: int | None = None,
        summary_count: int | None = None,
        fmd: float | None = None,
    ) -> Path:
        count = len(details) if summary_count is None else summary_count
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "meta": {"pairs": len(details) if pairs is None else pairs},
                    "summary": {
                        "accompaniment_vs_groundtruth": {
                            "pitch_jsd": {"count": count},
                            "onset_jsd": {"count": count},
                            "duration_jsd": {"count": count},
                            "frechet_music_distance": fmd,
                        },
                        "melody_relationship": {
                            "harmonicity": {
                                "consonant_ratio": {"count": count},
                                "unsupported_ratio": {"count": count},
                            }
                        },
                    },
                    "details": details,
                }
            ),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def _grid(
        pieces: tuple[str, ...] = ("p1", "p2"),
        seeds: tuple[str, ...] = ("0", "1"),
        *,
        default: bool = True,
    ) -> dict[tuple[str, str], bool]:
        return {(piece, seed): default for piece in pieces for seed in seeds}

    def _valid_details(
        self,
        grid: dict[tuple[str, str], bool],
        values: dict[tuple[str, str], float] | None = None,
    ) -> list[dict[str, object]]:
        values = values or {}
        return [
            self._detail(piece, seed, values.get((piece, seed), 0.2))
            for (piece, seed), valid in grid.items()
            if valid
        ]

    def test_conditional_means_vor_fmd_and_flat_cli_output(self) -> None:
        system_a = self._grid(default=False)
        system_a[("p1", "0")] = True
        system_a[("p2", "0")] = True
        system_a[("p2", "1")] = True
        system_b = self._grid(default=True)
        audit = self._write_audit({"A": system_a, "B": system_b})
        values_a = {
            ("p1", "0"): 0.1,
            ("p2", "0"): 0.3,
            ("p2", "1"): 0.5,
        }
        metrics_a = self._write_metrics(
            self.root / "A.json",
            self._valid_details(system_a, values_a),
            fmd=1.25,
        )
        metrics_b = self._write_metrics(
            self.root / "B.json",
            self._valid_details(system_b),
            fmd=2.5,
        )
        output_json = self.root / "summary.json"
        output_csv = self.root / "summary.csv"
        exit_code = main(
            [
                "--audit",
                str(audit),
                "--metrics",
                f"A={metrics_a}",
                "--metrics",
                f"B={metrics_b}",
                "--output-json",
                str(output_json),
                "--output-csv",
                str(output_csv),
                "--expected-piece-count",
                "2",
                "--expected-seeds",
                "0,1",
                "--bootstrap-replicates",
                "200",
                "--bootstrap-seed",
                "7",
            ]
        )
        self.assertEqual(exit_code, 0)
        summary = json.loads(output_json.read_text(encoding="utf-8"))
        metrics = summary["systems"]["A"]["metrics"]
        self.assertEqual(metrics["valid_output_rate"]["numerator"], 3)
        self.assertEqual(metrics["valid_output_rate"]["denominator"], 4)
        self.assertAlmostEqual(metrics["valid_output_rate"]["estimate"], 0.75)
        self.assertAlmostEqual(metrics["pitch_jsd"]["numerator"], 0.9)
        self.assertEqual(metrics["pitch_jsd"]["denominator"], 3)
        self.assertAlmostEqual(metrics["pitch_jsd"]["estimate"], 0.3)
        self.assertEqual(metrics["pitch_jsd"]["scope"], "conditional_on_valid_output")
        fmd = metrics["fmd"]
        self.assertEqual(fmd["estimate"], 1.25)
        self.assertIsNone(fmd["ci_low"])
        self.assertIsNone(fmd["ci_high"])
        self.assertEqual(fmd["ci_status"], "not_computed")
        self.assertIsNone(fmd["numerator"])
        self.assertEqual(fmd["denominator"], 3)
        self.assertEqual(fmd["scope"], "dataset_level_valid_output")

        with output_csv.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 14)
        self.assertEqual(len({(row["system_id"], row["metric"]) for row in rows}), 14)
        fmd_row = next(
            row for row in rows if row["system_id"] == "A" and row["metric"] == "fmd"
        )
        self.assertEqual(fmd_row["ci_low"], "")
        self.assertEqual(fmd_row["ci_high"], "")

    def test_empty_valid_system_is_legal_and_conditional_metrics_are_null(self) -> None:
        empty = self._grid(default=False)
        audit = self._write_audit({"empty": empty})
        metrics_path = self._write_metrics(self.root / "empty.json", [])
        summary = summarize_matched_music_metrics(
            audit_path=audit,
            metrics_paths={"empty": metrics_path},
            expected_piece_count=2,
            expected_seeds=("0", "1"),
            bootstrap_replicates=50,
            bootstrap_seed=0,
        )
        metrics = summary["systems"]["empty"]["metrics"]
        vor = metrics["valid_output_rate"]
        self.assertEqual(vor["estimate"], 0.0)
        self.assertEqual(vor["ci_low"], 0.0)
        self.assertEqual(vor["ci_high"], 0.0)
        self.assertEqual(vor["numerator"], 0)
        self.assertEqual(vor["denominator"], 4)
        for metric in ("pitch_jsd", "onset_jsd", "duration_jsd", "cr", "ur"):
            record = metrics[metric]
            self.assertIsNone(record["estimate"])
            self.assertIsNone(record["ci_low"])
            self.assertEqual(record["ci_status"], "not_computed")
            self.assertEqual(record["numerator"], 0)
            self.assertEqual(record["denominator"], 0)

    def test_bootstrap_is_deterministic_shared_and_system_order_independent(
        self,
    ) -> None:
        system_a = self._grid(default=False)
        system_a[("p1", "0")] = True
        system_b = self._grid(default=False)
        system_b[("p2", "1")] = True
        audit = self._write_audit({"A": system_a, "B": system_b})
        metrics_a = self._write_metrics(
            self.root / "A.json", self._valid_details(system_a)
        )
        metrics_b = self._write_metrics(
            self.root / "B.json", self._valid_details(system_b)
        )

        kwargs = {
            "audit_path": audit,
            "expected_piece_count": 2,
            "expected_seeds": ("0", "1"),
            "bootstrap_replicates": 100,
            "bootstrap_seed": 19,
        }
        first = summarize_matched_music_metrics(
            metrics_paths={"A": metrics_a, "B": metrics_b}, **kwargs
        )
        second = summarize_matched_music_metrics(
            metrics_paths={"B": metrics_b, "A": metrics_a}, **kwargs
        )
        self.assertEqual(first["bootstrap"], second["bootstrap"])
        self.assertEqual(first["systems"], second["systems"])
        self.assertRegex(
            first["bootstrap"]["shared_draw_matrix_sha256"], r"^[0-9a-f]{64}$"
        )
        third = summarize_matched_music_metrics(
            metrics_paths={"A": metrics_a, "B": metrics_b}, **kwargs
        )
        self.assertEqual(first, third)

    def test_audit_grid_and_schema_are_strict(self) -> None:
        valid = self._grid()
        base_rows = {
            "A": valid,
            "B": valid,
        }
        duplicate_row = {
            "piece_id": "p1",
            "seed": "0",
            "system_id": "A",
            "source_status": "complete",
            "preparation_status": "prepared",
            "valid_output": "True",
        }
        duplicate = self._write_audit(
            base_rows, name="duplicate.csv", extra_rows=[duplicate_row]
        )
        with self.assertRaisesRegex(MusicSummaryValidationError, "duplicate audit key"):
            load_matched_audit(
                duplicate,
                ("A", "B"),
                expected_piece_count=2,
                expected_seeds=("0", "1"),
            )

        missing_grid = {system: dict(values) for system, values in base_rows.items()}
        missing_grid["A"].pop(("p2", "1"))
        missing = self._write_audit(missing_grid, name="missing.csv")
        with self.assertRaisesRegex(MusicSummaryValidationError, "grid mismatch"):
            load_matched_audit(
                missing,
                ("A", "B"),
                expected_piece_count=2,
                expected_seeds=("0", "1"),
            )

        invalid_bool = self._write_audit(base_rows, name="invalid_bool.csv")
        rows = list(csv.DictReader(invalid_bool.open(encoding="utf-8")))
        rows[0]["valid_output"] = "1"
        with invalid_bool.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(
            MusicSummaryValidationError, "exactly true or false"
        ):
            load_matched_audit(
                invalid_bool,
                ("A", "B"),
                expected_piece_count=2,
                expected_seeds=("0", "1"),
            )

        extra_seed = {system: dict(values) for system, values in base_rows.items()}
        extra_seed["A"][("p1", "2")] = True
        extra = self._write_audit(extra_seed, name="extra.csv")
        with self.assertRaisesRegex(MusicSummaryValidationError, "seeds differ"):
            load_matched_audit(
                extra,
                ("A", "B"),
                expected_piece_count=2,
                expected_seeds=("0", "1"),
            )

    def test_detail_alignment_and_reported_counts_are_strict(self) -> None:
        grid = self._grid(default=False)
        grid[("p1", "0")] = True
        audit = self._write_audit({"A": grid})
        valid_detail = self._detail("p1", "0", 0.2)
        cases = (
            (
                "missing",
                [],
                {},
                "missing valid trials",
            ),
            (
                "invalid_present",
                [valid_detail, self._detail("p2", "0", 0.3)],
                {},
                "invalid-output trial appears",
            ),
            (
                "extra",
                [valid_detail, self._detail("unknown", "0", 0.3)],
                {},
                "unexpected metrics detail",
            ),
            (
                "duplicate",
                [valid_detail, dict(valid_detail)],
                {},
                "duplicate metrics detail",
            ),
            (
                "pairs",
                [valid_detail],
                {"pairs": 2},
                "meta.pairs=2",
            ),
            (
                "summary_count",
                [valid_detail],
                {"summary_count": 2},
                "summary pitch_jsd count 2",
            ),
            (
                "nonfinite",
                [{**valid_detail, "pitch_jsd": float("nan")}],
                {},
                "must be a finite number",
            ),
        )
        for name, details, options, message in cases:
            with self.subTest(name=name):
                metrics = self._write_metrics(
                    self.root / f"{name}.json", details, **options
                )
                with self.assertRaisesRegex(MusicSummaryValidationError, message):
                    summarize_matched_music_metrics(
                        audit_path=audit,
                        metrics_paths={"A": metrics},
                        expected_piece_count=2,
                        expected_seeds=("0", "1"),
                        bootstrap_replicates=10,
                    )

    def test_non_null_fmd_with_zero_valid_details_is_rejected(self) -> None:
        grid = self._grid(default=False)
        audit = self._write_audit({"A": grid})
        metrics = self._write_metrics(self.root / "fmd.json", [], fmd=1.0)
        with self.assertRaisesRegex(MusicSummaryValidationError, "zero valid details"):
            summarize_matched_music_metrics(
                audit_path=audit,
                metrics_paths={"A": metrics},
                expected_piece_count=2,
                expected_seeds=("0", "1"),
                bootstrap_replicates=10,
            )

    def test_duplicate_metrics_mapping_is_rejected(self) -> None:
        with self.assertRaisesRegex(MusicSummaryValidationError, "duplicate"):
            parse_metrics_mapping(["A=one.json", "A=two.json"])


if __name__ == "__main__":
    unittest.main()
