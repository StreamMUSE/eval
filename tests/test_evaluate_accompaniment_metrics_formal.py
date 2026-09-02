from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pretty_midi

from evaluate_accompaniment_metrics import _build_parser, evaluate_pair, main


class EvaluateAccompanimentMetricsFormalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _write_generated(
        path: Path,
        *,
        accompaniment: list[tuple[int, float, float]],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        melody = pretty_midi.Instrument(program=0, name="Melody")
        melody.notes.append(
            pretty_midi.Note(velocity=80, pitch=60, start=1.0, end=2.0)
        )
        acc = pretty_midi.Instrument(program=0, name="Accompaniment")
        acc.notes.extend(
            pretty_midi.Note(velocity=70, pitch=pitch, start=start, end=end)
            for pitch, start, end in accompaniment
        )
        midi.instruments.extend((melody, acc))
        midi.write(str(path))

    @staticmethod
    def _write_groundtruth(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        piano = pretty_midi.Instrument(program=0, name="Piano")
        piano.notes.extend(
            (
                pretty_midi.Note(velocity=70, pitch=48, start=1.0, end=1.5),
                pretty_midi.Note(velocity=70, pitch=52, start=8.0, end=9.5),
            )
        )
        midi.instruments.append(piano)
        midi.write(str(path))

    def _formal_args(self, generated_dir: Path, groundtruth_dir: Path):
        return _build_parser().parse_args(
            [
                "--generated-dir",
                str(generated_dir),
                "--groundtruth-dir",
                str(groundtruth_dir),
                "--evaluation-duration-seconds",
                "12.0",
                "--duration-histogram-upper-bound-seconds",
                "12.0",
            ]
        )

    def test_same_gt_histograms_are_identical_across_generated_systems(self) -> None:
        generated_a = self.root / "system-a" / "piece.mid"
        generated_b = self.root / "system-b" / "piece.mid"
        groundtruth = self.root / "gt" / "piece.mid"
        self._write_generated(
            generated_a,
            accompaniment=[(48, 1.0, 1.25), (55, 3.0, 3.25)],
        )
        self._write_generated(
            generated_b,
            accompaniment=[(36, 0.0, 7.0), (72, 11.0, 12.0)],
        )
        self._write_groundtruth(groundtruth)

        result_a = evaluate_pair(
            generated_a,
            groundtruth,
            self._formal_args(generated_a.parent, groundtruth.parent),
        )
        result_b = evaluate_pair(
            generated_b,
            groundtruth,
            self._formal_args(generated_b.parent, groundtruth.parent),
        )

        for field in (
            "onset_ground_truth",
            "duration_ground_truth",
            "onset_edges",
            "duration_edges",
        ):
            self.assertEqual(
                result_a["histograms"][field], result_b["histograms"][field]
            )
        self.assertEqual(result_a["histograms"]["onset_edges"][0], 0.0)
        self.assertEqual(result_a["histograms"]["onset_edges"][-1], 12.0)
        self.assertEqual(result_a["histograms"]["duration_edges"][-1], 12.0)

    def test_output_json_records_formal_config_reproducibility_and_hashes(self) -> None:
        generated_dir = self.root / "generated"
        groundtruth_dir = self.root / "groundtruth"
        generated = generated_dir / "piece.mid"
        groundtruth = groundtruth_dir / "piece.mid"
        output = self.root / "metrics.json"
        self._write_generated(generated, accompaniment=[(48, 1.0, 2.0)])
        self._write_groundtruth(groundtruth)

        argv = [
            "evaluate_accompaniment_metrics.py",
            "--generated-dir",
            str(generated_dir),
            "--groundtruth-dir",
            str(groundtruth_dir),
            "--evaluation-duration-seconds",
            "12.0",
            "--duration-histogram-upper-bound-seconds",
            "12.0",
            "--output-json",
            str(output),
        ]
        with patch.object(sys, "argv", argv), redirect_stdout(StringIO()):
            main()

        payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(payload["meta"]["schema_version"], 2)
        self.assertEqual(
            payload["meta"]["metric_config"]["histogram_mode"],
            "fixed_seconds",
        )
        self.assertIn("cli_config", payload["meta"])
        self.assertRegex(
            payload["meta"]["reproducibility"]["code"]["evaluator_sha256"],
            r"^[0-9a-f]{64}$",
        )
        detail = payload["details"][0]
        self.assertEqual(
            detail["generated_sha256"],
            hashlib.sha256(generated.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            detail["metric_gt_sha256"],
            hashlib.sha256(groundtruth.read_bytes()).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
