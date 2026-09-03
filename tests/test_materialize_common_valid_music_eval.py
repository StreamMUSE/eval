from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval_toolkit.materialize_common_valid_music_eval import (
    CommonValidMaterializationError,
    materialize_common_valid_music_eval,
)


class MaterializeCommonValidMusicEvalTest(unittest.TestCase):
    SYSTEMS = ("streammuse_v1_standard", "streammuse_v2_prompt_continuation")
    KEYS = (("piece-a", "0"), ("piece-a", "1"), ("piece-b", "0"))

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _write_audit(
        self,
        *,
        invalid: set[tuple[str, str, str]] | None = None,
        mutate: Callable[[dict[str, str]], None] | None = None,
    ) -> Path:
        invalid = invalid or set()
        rows: list[dict[str, str]] = []
        for system_id in self.SYSTEMS:
            for piece_id, seed in self.KEYS:
                valid = (system_id, piece_id, seed) not in invalid
                basename = f"piece-{piece_id}__seed-{seed}.mid"
                generated = (
                    self.root
                    / system_id
                    / "valid_only"
                    / "generated"
                    / basename
                )
                gt = (
                    self.root
                    / system_id
                    / "valid_only"
                    / "groundtruth"
                    / basename
                )
                generated.parent.mkdir(parents=True, exist_ok=True)
                gt.parent.mkdir(parents=True, exist_ok=True)
                generated.write_bytes(
                    f"generated:{system_id}:{piece_id}:{seed}".encode()
                )
                gt.write_bytes(f"gt:{piece_id}:{seed}".encode())
                row = {
                    "piece_id": piece_id,
                    "seed": seed,
                    "system_id": system_id,
                    "source_status": "complete",
                    "preparation_status": "prepared",
                    "valid_output": str(valid),
                    "valid_only_generated_midi": (
                        str(generated.relative_to(self.root)) if valid else ""
                    ),
                    "valid_only_metric_gt_midi": (
                        str(gt.relative_to(self.root)) if valid else ""
                    ),
                    "generated_sha256": self._sha256(generated),
                    "metric_gt_sha256": self._sha256(gt),
                }
                if mutate is not None:
                    mutate(row)
                rows.append(row)

        audit = self.root / "audit.csv"
        with audit.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return audit

    def test_materializes_exact_intersection_with_deterministic_manifests(self) -> None:
        audit = self._write_audit(
            invalid={(self.SYSTEMS[1], "piece-a", "1")}
        )
        first = self.root / "common-first"
        second = self.root / "common-second"

        manifest = materialize_common_valid_music_eval(
            audit_path=audit,
            system_ids=reversed(self.SYSTEMS),
            output_dir=first,
        )
        materialize_common_valid_music_eval(
            audit_path=audit,
            system_ids=self.SYSTEMS,
            output_dir=second,
        )

        self.assertEqual(manifest["common_valid_key_count"], 2)
        self.assertEqual(
            manifest["common_valid_keys"],
            [
                {"piece_id": "piece-a", "seed": "0"},
                {"piece_id": "piece-b", "seed": "0"},
            ],
        )
        self.assertEqual(
            (first / "manifest.json").read_bytes(),
            (second / "manifest.json").read_bytes(),
        )
        self.assertEqual(
            (first / "manifest.csv").read_bytes(),
            (second / "manifest.csv").read_bytes(),
        )
        for system_id in self.SYSTEMS:
            generated = list((first / system_id / "generated").glob("*.mid"))
            groundtruth = list(
                (first / system_id / "groundtruth").glob("*.mid")
            )
            self.assertEqual(len(generated), 2)
            self.assertEqual(len(groundtruth), 2)
            self.assertEqual(
                {path.name for path in generated},
                {path.name for path in groundtruth},
            )
        with (first / "manifest.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 4)

    def test_rejects_system_key_basename_and_hash_contract_violations(self) -> None:
        audit = self._write_audit()
        with self.subTest("system set"):
            with self.assertRaisesRegex(
                CommonValidMaterializationError, "exactly match"
            ):
                materialize_common_valid_music_eval(
                    audit_path=audit,
                    system_ids=(self.SYSTEMS[0],),
                    output_dir=self.root / "bad-system-set",
                )

        with audit.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        with self.subTest("duplicate key"):
            duplicate_audit = self.root / "duplicate.csv"
            with duplicate_audit.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
                writer.writeheader()
                writer.writerows([*rows, rows[0]])
            with self.assertRaisesRegex(
                CommonValidMaterializationError, "duplicate audit key"
            ):
                materialize_common_valid_music_eval(
                    audit_path=duplicate_audit,
                    system_ids=self.SYSTEMS,
                    output_dir=self.root / "bad-duplicate",
                )

        with self.subTest("basename"):
            basename_audit = self._write_audit(
                mutate=lambda row: row.update(
                    {
                        "valid_only_metric_gt_midi": (
                            row["valid_only_metric_gt_midi"] + ".different"
                            if row["system_id"] == self.SYSTEMS[0]
                            and row["piece_id"] == "piece-a"
                            and row["seed"] == "0"
                            else row["valid_only_metric_gt_midi"]
                        )
                    }
                )
            )
            source_gt = self.root / rows[0]["valid_only_metric_gt_midi"]
            mismatched_gt = Path(str(source_gt) + ".different")
            mismatched_gt.write_bytes(source_gt.read_bytes())
            with self.assertRaisesRegex(
                CommonValidMaterializationError, "basenames differ"
            ):
                materialize_common_valid_music_eval(
                    audit_path=basename_audit,
                    system_ids=self.SYSTEMS,
                    output_dir=self.root / "bad-basename",
                )

        with self.subTest("hash"):
            hash_audit = self._write_audit(
                mutate=lambda row: row.update(
                    {
                        "generated_sha256": "0" * 64
                        if row["system_id"] == self.SYSTEMS[0]
                        and row["piece_id"] == "piece-a"
                        and row["seed"] == "0"
                        else row["generated_sha256"]
                    }
                )
            )
            with self.assertRaisesRegex(
                CommonValidMaterializationError, "SHA256 mismatch"
            ):
                materialize_common_valid_music_eval(
                    audit_path=hash_audit,
                    system_ids=self.SYSTEMS,
                    output_dir=self.root / "bad-hash",
                )

    def test_staging_cleanup_and_existing_output_are_non_destructive(self) -> None:
        audit = self._write_audit()
        existing = self.root / "existing"
        existing.mkdir()
        sentinel = existing / "sentinel.txt"
        sentinel.write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(
            CommonValidMaterializationError, "refusing to overwrite"
        ):
            materialize_common_valid_music_eval(
                audit_path=audit,
                system_ids=self.SYSTEMS,
                output_dir=existing,
            )
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

        output = self.root / "copy-failure"
        with mock.patch(
            "eval_toolkit.materialize_common_valid_music_eval.shutil.copy2",
            side_effect=OSError("injected copy failure"),
        ):
            with self.assertRaisesRegex(OSError, "injected copy failure"):
                materialize_common_valid_music_eval(
                    audit_path=audit,
                    system_ids=self.SYSTEMS,
                    output_dir=output,
                )
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob(".copy-failure.staging-*")), [])


if __name__ == "__main__":
    unittest.main()
