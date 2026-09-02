from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import mido
import pretty_midi

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval_toolkit.prepare_matched_music_eval import (
    PreparationBlockedError,
    PreparationError,
    load_cohort_manifest,
    main,
    prepare_matched_music_eval,
)


class PrepareMatchedMusicEvalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _write_midi(
        path: Path,
        *,
        melody: list[tuple[int, float, float]] | None,
        accompaniment: list[tuple[int, float, float]] | None,
        melody_name: str = "Melody",
        accompaniment_name: str = "Accompaniment",
        bpm: float = 120.0,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        if melody is not None:
            instrument = pretty_midi.Instrument(program=0, name=melody_name)
            instrument.notes.extend(
                pretty_midi.Note(velocity=80, pitch=pitch, start=start, end=end)
                for pitch, start, end in melody
            )
            midi.instruments.append(instrument)
        if accompaniment is not None:
            instrument = pretty_midi.Instrument(program=1, name=accompaniment_name)
            instrument.notes.extend(
                pretty_midi.Note(velocity=70, pitch=pitch, start=start, end=end)
                for pitch, start, end in accompaniment
            )
            midi.instruments.append(instrument)
        midi.write(str(path))

    def _cohort(
        self,
        *,
        piece_id: str = "piece-a",
        gt_accompaniment: list[tuple[int, float, float]] | None = None,
        declared_hash: str | None = None,
        declared_gt_hash: str | None = None,
    ) -> tuple[Path, Path, Path]:
        melody = self.root / piece_id / "melody_120bpm.mid"
        gt = self.root / piece_id / "gt_120bpm.mid"
        source_npz = self.root / piece_id / "source.npz"
        source_npz.parent.mkdir(parents=True, exist_ok=True)
        source_npz.write_bytes(f"canonical-npz:{piece_id}".encode())
        self._write_midi(
            melody,
            melody=[(60, 3.0, 5.0), (62, 7.0, 8.0), (64, 15.5, 17.0)],
            accompaniment=None,
        )
        self._write_midi(
            gt,
            melody=[(60, 3.0, 5.0), (62, 7.0, 8.0), (64, 15.5, 17.0)],
            accompaniment=(
                [(48, 3.5, 4.5), (50, 8.0, 9.0), (52, 15.0, 17.0)]
                if gt_accompaniment is None
                else gt_accompaniment
            ),
        )
        manifest = self.root / f"cohort_{piece_id}.json"
        manifest.write_text(
            json.dumps(
                {
                    "samples": [
                        {
                            "piece_id": piece_id,
                            "melody_midi": str(melody.relative_to(self.root)),
                            "gt_midi": str(gt.relative_to(self.root)),
                            "source_npz": str(source_npz.relative_to(self.root)),
                            "melody_midi_sha256": declared_hash or self._sha256(melody),
                            "gt_midi_sha256": declared_gt_hash or self._sha256(gt),
                            "source_npz_sha256": self._sha256(source_npz),
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return manifest, melody, gt

    @staticmethod
    def _cohort_source_npz_sha256(cohort_manifest: Path) -> str:
        data = json.loads(cohort_manifest.read_text(encoding="utf-8"))
        return data["samples"][0]["source_npz_sha256"]

    @staticmethod
    def _write_postjoin_gt(source: Path, target: Path) -> None:
        source_midi = pretty_midi.PrettyMIDI(str(source))
        target_midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        for source_track in source_midi.instruments:
            target_track = pretty_midi.Instrument(
                program=source_track.program,
                is_drum=source_track.is_drum,
                name=source_track.name,
            )
            for note in source_track.notes:
                if note.end <= 4.0 or note.start >= 16.0:
                    continue
                target_track.notes.append(
                    pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=max(note.start, 4.0) - 4.0,
                        end=min(note.end, 16.0) - 4.0,
                    )
                )
            target_midi.instruments.append(target_track)
        target.parent.mkdir(parents=True, exist_ok=True)
        target_midi.write(str(target))

    def _write_realtime_manifest(
        self,
        rows: list[dict[str, str]],
        *,
        name: str = "realtime.csv",
    ) -> Path:
        path = self.root / name
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def _realtime_row(
        self,
        *,
        piece_id: str,
        session: Path,
        melody_hash: str,
        seed: str = "0",
        system_id: str = "streaming-v2",
        status_field: str = "run_status",
        hash_field: str = "melody_input_sha256",
        status: str = "complete",
        reason: str = "",
    ) -> dict[str, str]:
        return {
            "piece_id": piece_id,
            "seed": seed,
            "system_id": system_id,
            "session_dir": str(session),
            status_field: status,
            hash_field: melody_hash,
            "failure_reason": reason,
        }

    def test_realtime_crop_clips_boundaries_and_writes_metric_ready_pairs(self) -> None:
        cohort, melody, _ = self._cohort()
        session = self.root / "session"
        self._write_midi(
            session / "combined.mid",
            melody=[
                (55, 1.0, 2.0),
                (60, 3.0, 5.0),
                (62, 8.0, 9.0),
                (64, 15.5, 17.0),
            ],
            accompaniment=[
                (43, 2.0, 3.0),
                (48, 3.5, 4.5),
                (50, 9.0, 10.0),
                (52, 15.0, 17.0),
            ],
        )
        realtime = self._write_realtime_manifest(
            [
                self._realtime_row(
                    piece_id="piece-a",
                    session=session,
                    melody_hash=self._sha256(melody),
                )
            ]
        )
        output = self.root / "output"
        manifest = prepare_matched_music_eval(
            cohort_manifest=cohort,
            realtime_manifest=realtime,
            output_dir=output,
            expected_piece_count=1,
            expected_seeds=("0",),
        )

        basename = "piece-piece-a__seed-0.mid"
        generated = output / "streaming-v2" / "all_trials" / "generated" / basename
        groundtruth = output / "streaming-v2" / "all_trials" / "groundtruth" / basename
        valid_generated = (
            output / "streaming-v2" / "valid_only" / "generated" / basename
        )
        valid_groundtruth = (
            output / "streaming-v2" / "valid_only" / "groundtruth" / basename
        )
        self.assertTrue(generated.is_file())
        self.assertTrue(groundtruth.is_file())
        self.assertTrue(valid_generated.is_file())
        self.assertTrue(valid_groundtruth.is_file())
        self.assertEqual(generated.name, groundtruth.name)

        generated_midi = pretty_midi.PrettyMIDI(str(generated))
        generated_tracks = {track.name: track for track in generated_midi.instruments}
        self.assertEqual(set(generated_tracks), {"Melody", "Accompaniment"})
        melody_notes = generated_tracks["Melody"].notes
        self.assertEqual({note.pitch for note in melody_notes}, {60, 62, 64})
        clipped_start = next(note for note in melody_notes if note.pitch == 60)
        clipped_end = next(note for note in melody_notes if note.pitch == 64)
        self.assertAlmostEqual(clipped_start.start, 0.0, places=5)
        self.assertAlmostEqual(clipped_start.end, 1.0, places=5)
        self.assertAlmostEqual(clipped_end.start, 11.5, places=5)
        self.assertAlmostEqual(clipped_end.end, 12.0, places=5)
        all_notes = [
            note for track in generated_midi.instruments for note in track.notes
        ]
        self.assertTrue(all(0.0 <= note.start < note.end <= 12.0 for note in all_notes))

        gt_midi = pretty_midi.PrettyMIDI(str(groundtruth))
        self.assertEqual([track.name for track in gt_midi.instruments], ["Piano"])
        self.assertTrue(gt_midi.instruments[0].notes)
        self.assertNotIn("Melody", [track.name for track in gt_midi.instruments])

        summary = manifest["system_summaries"]["streaming-v2"]
        self.assertEqual(summary["planned_trial_count"], 1)
        self.assertEqual(summary["valid_output_count"], 1)
        self.assertEqual(summary["valid_output_rate"], 1.0)
        self.assertEqual(summary["music_metrics_scope"], "conditional_on_valid_output")
        row = manifest["trials"][0]
        self.assertEqual(row["cohort_postjoin_melody_note_count"], 3)
        self.assertRegex(row["cohort_postjoin_melody_sha256"], r"^[0-9a-f]{64}$")

    def test_empty_generated_accompaniment_is_a_valid_preparation_result(self) -> None:
        cohort, melody, _ = self._cohort()
        session = self.root / "empty-session"
        self._write_midi(
            session / "combined.mid",
            melody=[(60, 5.0, 6.0)],
            accompaniment=None,
        )
        realtime = self._write_realtime_manifest(
            [
                self._realtime_row(
                    piece_id="piece-a",
                    session=session,
                    melody_hash=self._sha256(melody),
                )
            ]
        )
        output = self.root / "empty-output"
        result = prepare_matched_music_eval(
            cohort_manifest=cohort,
            realtime_manifest=realtime,
            output_dir=output,
            expected_piece_count=1,
            expected_seeds=("0",),
        )

        row = result["trials"][0]
        self.assertEqual(row["generated_acc_note_count"], 0)
        self.assertIs(row["valid_output"], False)
        self.assertTrue((output / row["all_trials_generated_midi"]).is_file())
        self.assertTrue((output / row["all_trials_metric_gt_midi"]).is_file())
        self.assertIsNone(row["valid_only_generated_midi"])
        self.assertIsNone(row["valid_only_metric_gt_midi"])
        valid_root = output / "streaming-v2" / "valid_only"
        self.assertTrue((valid_root / "generated").is_dir())
        self.assertTrue((valid_root / "groundtruth").is_dir())
        self.assertEqual(list((valid_root / "generated").iterdir()), [])
        self.assertEqual(list((valid_root / "groundtruth").iterdir()), [])
        summary = result["system_summaries"]["streaming-v2"]
        self.assertEqual(summary["planned_trial_count"], 1)
        self.assertEqual(summary["valid_output_count"], 0)
        self.assertEqual(summary["valid_output_rate"], 0.0)
        midi_tracks = [
            message.name
            for track in mido.MidiFile(output / row["all_trials_generated_midi"]).tracks
            for message in track
            if message.type == "track_name"
        ]
        self.assertIn("Melody", midi_tracks)

    def test_empty_gt_accompaniment_is_rejected_by_cohort_contract(self) -> None:
        cohort, melody, _ = self._cohort(gt_accompaniment=[])
        session = self.root / "session"
        self._write_midi(
            session / "combined.mid",
            melody=[(60, 5.0, 6.0)],
            accompaniment=[(48, 5.0, 6.0)],
        )
        realtime = self._write_realtime_manifest(
            [
                self._realtime_row(
                    piece_id="piece-a",
                    session=session,
                    melody_hash=self._sha256(melody),
                )
            ]
        )
        output = self.root / "blocked-gt"
        with self.assertRaisesRegex(PreparationError, "Accompaniment"):
            prepare_matched_music_eval(
                cohort_manifest=cohort,
                realtime_manifest=realtime,
                output_dir=output,
                expected_piece_count=1,
                expected_seeds=("0",),
            )
        self.assertFalse((output / "audit.csv").exists())
        self.assertFalse((output / "prepared_manifest.json").exists())
        self.assertFalse((output / "streaming-v2").exists())

    def test_missing_generated_melody_track_blocks(self) -> None:
        cohort, melody, _ = self._cohort()
        session = self.root / "acc-only-session"
        self._write_midi(
            session / "combined.mid",
            melody=None,
            accompaniment=[(48, 5.0, 6.0)],
        )
        realtime = self._write_realtime_manifest(
            [
                self._realtime_row(
                    piece_id="piece-a",
                    session=session,
                    melody_hash=self._sha256(melody),
                )
            ]
        )
        output = self.root / "missing-melody-output"
        with self.assertRaises(PreparationBlockedError):
            prepare_matched_music_eval(
                cohort_manifest=cohort,
                realtime_manifest=realtime,
                output_dir=output,
                expected_piece_count=1,
                expected_seeds=("0",),
            )
        data = json.loads((output / "prepared_manifest.json").read_text())
        self.assertIn("named Melody track", data["blockers"][0])

    def test_failed_and_inferred_missing_rows_block_without_silent_skip(self) -> None:
        cohort, melody, _ = self._cohort()
        rows = [
            self._realtime_row(
                piece_id="piece-a",
                session=self.root / "unused",
                melody_hash=self._sha256(melody),
                seed="0",
                status="failed",
                reason="model crashed",
            )
        ]
        realtime = self._write_realtime_manifest(rows)
        output = self.root / "blocked-grid"
        with self.assertRaises(PreparationBlockedError):
            prepare_matched_music_eval(
                cohort_manifest=cohort,
                realtime_manifest=realtime,
                output_dir=output,
                expected_piece_count=1,
                expected_seeds=("0", "1"),
            )
        with (output / "audit.csv").open(encoding="utf-8", newline="") as handle:
            audit = list(csv.DictReader(handle))
        self.assertEqual(len(audit), 2)
        self.assertEqual({row["source_status"] for row in audit}, {"failed", "missing"})
        self.assertFalse((output / "streaming-v2").exists())

    def test_status_hash_aliases_are_strict_and_hash_mismatch_blocks(self) -> None:
        cohort, melody, _ = self._cohort()
        session = self.root / "session"
        self._write_midi(
            session / "combined.mid",
            melody=[(60, 5.0, 6.0)],
            accompaniment=[(48, 5.0, 6.0)],
        )
        alias_manifest = self._write_realtime_manifest(
            [
                self._realtime_row(
                    piece_id="piece-a",
                    session=session,
                    melody_hash=self._sha256(melody),
                    status_field="status",
                    hash_field="hash",
                )
            ],
            name="aliases.csv",
        )
        self.assertEqual(
            main(
                [
                    "--cohort-manifest",
                    str(cohort),
                    "--realtime-manifest",
                    str(alias_manifest),
                    "--output-dir",
                    str(self.root / "alias-output"),
                    "--expected-piece-count",
                    "1",
                    "--expected-seeds",
                    "0",
                ]
            ),
            0,
        )

        ambiguous = self._realtime_row(
            piece_id="piece-a",
            session=session,
            melody_hash=self._sha256(melody),
        )
        ambiguous["status"] = "complete"
        ambiguous_manifest = self._write_realtime_manifest(
            [ambiguous], name="ambiguous.csv"
        )
        with self.assertRaisesRegex(PreparationError, "ambiguous aliases"):
            prepare_matched_music_eval(
                cohort_manifest=cohort,
                realtime_manifest=ambiguous_manifest,
                output_dir=self.root / "ambiguous-output",
                expected_piece_count=1,
                expected_seeds=("0",),
            )

        mismatch_manifest = self._write_realtime_manifest(
            [
                self._realtime_row(
                    piece_id="piece-a",
                    session=session,
                    melody_hash="0" * 64,
                )
            ],
            name="mismatch.csv",
        )
        with self.assertRaises(PreparationBlockedError):
            prepare_matched_music_eval(
                cohort_manifest=cohort,
                realtime_manifest=mismatch_manifest,
                output_dir=self.root / "mismatch-output",
                expected_piece_count=1,
                expected_seeds=("0",),
            )

    def test_duplicate_key_missing_file_and_default_grid_validation(self) -> None:
        cohort, melody, _ = self._cohort()
        row = self._realtime_row(
            piece_id="piece-a",
            session=self.root / "missing-session",
            melody_hash=self._sha256(melody),
        )
        duplicate = self._write_realtime_manifest([row, row], name="duplicate.csv")
        with self.assertRaisesRegex(PreparationError, "duplicate trial key"):
            prepare_matched_music_eval(
                cohort_manifest=cohort,
                realtime_manifest=duplicate,
                output_dir=self.root / "duplicate-output",
                expected_piece_count=1,
                expected_seeds=("0",),
            )

        one_row = self._write_realtime_manifest([row], name="one-row.csv")
        with self.assertRaisesRegex(PreparationError, "exactly 40 pieces"):
            prepare_matched_music_eval(
                cohort_manifest=cohort,
                realtime_manifest=one_row,
                output_dir=self.root / "default-grid-output",
            )
        with self.assertRaises(PreparationBlockedError):
            prepare_matched_music_eval(
                cohort_manifest=cohort,
                realtime_manifest=one_row,
                output_dir=self.root / "missing-file-output",
                expected_piece_count=1,
                expected_seeds=("0",),
            )

    def test_cohort_file_hash_is_not_confused_with_canonical_hash(self) -> None:
        cohort, _, _ = self._cohort(declared_hash="f" * 64)
        with self.assertRaisesRegex(PreparationError, "file hash mismatch"):
            prepare_matched_music_eval(
                cohort_manifest=cohort,
                realtime_manifest=self.root / "unused.csv",
                output_dir=self.root / "hash-output",
                expected_piece_count=1,
                expected_seeds=("0",),
            )

        gt_cohort, _, _ = self._cohort(
            piece_id="piece-gt-hash", declared_gt_hash="e" * 64
        )
        with self.assertRaisesRegex(PreparationError, "GT MIDI file hash mismatch"):
            prepare_matched_music_eval(
                cohort_manifest=gt_cohort,
                realtime_manifest=self.root / "unused-gt.csv",
                output_dir=self.root / "gt-hash-output",
                expected_piece_count=1,
                expected_seeds=("0",),
            )

    def test_cohort_requires_a_valid_source_npz_identity_hash(self) -> None:
        cases = (
            ("missing", None, "source_npz_sha256.*missing"),
            ("malformed", "abc", "64 lowercase hex digits"),
            ("mismatch", "0" * 64, "source NPZ file hash mismatch"),
        )
        for case, replacement, message in cases:
            with self.subTest(case=case):
                cohort, _, _ = self._cohort(piece_id=f"source-hash-{case}")
                data = json.loads(cohort.read_text(encoding="utf-8"))
                if replacement is None:
                    data["samples"][0].pop("source_npz_sha256")
                else:
                    data["samples"][0]["source_npz_sha256"] = replacement
                cohort.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaisesRegex(PreparationError, message):
                    load_cohort_manifest(cohort)

    def test_cohort_melody_content_contract_rejects_self_hashed_bad_files(
        self,
    ) -> None:
        cases = (
            ("unparseable", "cannot parse MIDI"),
            ("empty", "exactly one named Melody track"),
            ("contains_acc", "non-Melody music track"),
            ("mismatch", "does not match gt_midi Melody"),
        )
        for case, message in cases:
            with self.subTest(case=case):
                cohort, melody, _ = self._cohort(piece_id=f"bad-{case}")
                if case == "unparseable":
                    melody.write_bytes(b"not a MIDI file")
                elif case == "empty":
                    self._write_midi(
                        melody,
                        melody=[],
                        accompaniment=None,
                    )
                elif case == "contains_acc":
                    self._write_midi(
                        melody,
                        melody=[(60, 5.0, 6.0)],
                        accompaniment=[(48, 5.0, 6.0)],
                    )
                else:
                    self._write_midi(
                        melody,
                        melody=[(72, 5.0, 6.0)],
                        accompaniment=None,
                    )
                data = json.loads(cohort.read_text(encoding="utf-8"))
                data["samples"][0]["melody_midi_sha256"] = self._sha256(melody)
                cohort.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaisesRegex(PreparationError, message):
                    load_cohort_manifest(cohort)

    def test_offline_csv_and_json_are_not_shifted_twice(self) -> None:
        cohort, _, gt_full = self._cohort()
        gt_full_midi = pretty_midi.PrettyMIDI(str(gt_full))
        gt_melody = []
        gt_accompaniment = []
        for instrument in gt_full_midi.instruments:
            target = gt_melody if instrument.name == "Melody" else gt_accompaniment
            target.extend(
                (
                    note.pitch,
                    max(note.start, 4.0) - 4.0,
                    min(note.end, 16.0) - 4.0,
                )
                for note in instrument.notes
                if note.end > 4.0 and note.start < 16.0
            )
        postjoin_gt = self.root / "offline" / "postjoin_gt.mid"
        self._write_midi(
            postjoin_gt,
            melody=gt_melody,
            accompaniment=gt_accompaniment,
        )

        for extension in ("csv", "json"):
            with self.subTest(extension=extension):
                generated = self.root / extension / "generated.mid"
                self._write_midi(
                    generated,
                    melody=[(60, 1.0, 2.0)],
                    accompaniment=[(48, 2.0, 3.0)],
                )
                row = {
                    "piece_id": "piece-a",
                    "seed": "0",
                    "system_id": f"beat-offline-{extension}",
                    "postjoin_generated_midi": str(generated),
                    "postjoin_gt_midi": str(postjoin_gt),
                    "source_npz_sha256": self._cohort_source_npz_sha256(cohort),
                    "run_status": "complete",
                    "failure_reason": "",
                }
                offline = self.root / f"offline.{extension}"
                if extension == "csv":
                    with offline.open("w", encoding="utf-8", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(row))
                        writer.writeheader()
                        writer.writerow(row)
                else:
                    offline.write_text(json.dumps({"trials": [row]}), encoding="utf-8")
                output = self.root / f"offline-output-{extension}"
                result = prepare_matched_music_eval(
                    cohort_manifest=cohort,
                    offline_manifest=offline,
                    output_dir=output,
                    expected_piece_count=1,
                    expected_seeds=("0",),
                )
                prepared = pretty_midi.PrettyMIDI(
                    str(output / result["trials"][0]["all_trials_generated_midi"])
                )
                melody_note = next(
                    track.notes[0]
                    for track in prepared.instruments
                    if track.name == "Melody"
                )
                self.assertAlmostEqual(melody_note.start, 1.0, places=5)
                audit_row = result["trials"][0]
                self.assertEqual(
                    Path(audit_row["source_gt_midi"]), postjoin_gt.resolve()
                )
                self.assertEqual(
                    audit_row["source_gt_sha256"], self._sha256(postjoin_gt)
                )
                self.assertEqual(
                    Path(audit_row["cohort_full_gt_midi"]), gt_full.resolve()
                )
                self.assertEqual(
                    audit_row["cohort_full_gt_sha256"], self._sha256(gt_full)
                )
                self.assertEqual(
                    audit_row["trial_source_npz_sha256"],
                    self._cohort_source_npz_sha256(cohort),
                )
                self.assertEqual(
                    audit_row["cohort_source_npz_sha256"],
                    self._cohort_source_npz_sha256(cohort),
                )
                self.assertIs(audit_row["offline_gt_roundtrip_exact"], True)
                self.assertEqual(
                    result["systems"][0]["system_scope"], "music_quality_only"
                )
                self.assertFalse(result["produces_system_metrics"])

    def test_offline_gt_matching_ignores_only_velocity(self) -> None:
        cohort, _, gt_full = self._cohort()
        cases = (
            ("velocity", True),
            ("pitch", False),
            ("onset", False),
            ("duration", False),
        )
        for case, should_pass in cases:
            with self.subTest(case=case):
                case_root = self.root / f"offline-geometry-{case}"
                postjoin_gt = case_root / "postjoin_gt.mid"
                canonical_gt = case_root / "canonical_gt.mid"
                self._write_postjoin_gt(gt_full, postjoin_gt)
                self._write_postjoin_gt(gt_full, canonical_gt)
                midi = pretty_midi.PrettyMIDI(str(postjoin_gt))
                accompaniment = next(
                    track for track in midi.instruments if track.name == "Accompaniment"
                )
                note = accompaniment.notes[0]
                if case == "velocity":
                    for item in accompaniment.notes:
                        item.velocity = max(1, item.velocity - 23)
                elif case == "pitch":
                    note.pitch += 1
                elif case == "onset":
                    note.start += 0.125
                else:
                    note.end += 0.125
                midi.write(str(postjoin_gt))

                generated = case_root / "generated.mid"
                self._write_midi(
                    generated,
                    melody=[(60, 1.0, 2.0)],
                    accompaniment=[(48, 2.0, 3.0)],
                )
                offline = case_root / "offline.json"
                offline.write_text(
                    json.dumps(
                        [
                            {
                                "piece_id": "piece-a",
                                "seed": "0",
                                "system_id": "beat-offline",
                                "postjoin_generated_midi": str(generated),
                                "postjoin_gt_midi": str(postjoin_gt),
                                "source_npz_sha256": (
                                    self._cohort_source_npz_sha256(cohort)
                                ),
                                "run_status": "complete",
                                "failure_reason": "",
                            }
                        ]
                    ),
                    encoding="utf-8",
                )
                output = case_root / "output"
                result = prepare_matched_music_eval(
                    cohort_manifest=cohort,
                    offline_manifest=offline,
                    output_dir=output,
                    expected_piece_count=1,
                    expected_seeds=("0",),
                )
                self.assertEqual(result["preparation_status"], "success")
                audit_row = result["trials"][0]
                self.assertIs(audit_row["offline_gt_roundtrip_exact"], should_pass)

                canonical_midi = pretty_midi.PrettyMIDI(str(canonical_gt))
                canonical_acc = next(
                    track
                    for track in canonical_midi.instruments
                    if track.name == "Accompaniment"
                )
                metric_gt = pretty_midi.PrettyMIDI(
                    str(output / audit_row["all_trials_metric_gt_midi"])
                )
                self.assertEqual(
                    [track.name for track in metric_gt.instruments], ["Piano"]
                )
                expected_geometry = sorted(
                    (note.pitch, round(note.start, 6), round(note.end, 6))
                    for note in canonical_acc.notes
                )
                actual_geometry = sorted(
                    (note.pitch, round(note.start, 6), round(note.end, 6))
                    for note in metric_gt.instruments[0].notes
                )
                self.assertEqual(actual_geometry, expected_geometry)

    def test_offline_source_npz_identity_missing_or_mismatch_blocks(self) -> None:
        cohort, _, gt_full = self._cohort()
        expected_hash = self._cohort_source_npz_sha256(cohort)
        for case, source_hash, message in (
            ("missing", None, "source_npz_sha256 is required"),
            ("mismatch", "0" * 64, "does not match cohort"),
        ):
            with self.subTest(case=case):
                case_root = self.root / f"offline-source-{case}"
                generated = case_root / "generated.mid"
                postjoin_gt = case_root / "postjoin_gt.mid"
                self._write_midi(
                    generated,
                    melody=[(60, 1.0, 2.0)],
                    accompaniment=[(48, 2.0, 3.0)],
                )
                self._write_postjoin_gt(gt_full, postjoin_gt)
                row = {
                    "piece_id": "piece-a",
                    "seed": "0",
                    "system_id": "beat-offline",
                    "postjoin_generated_midi": str(generated),
                    "postjoin_gt_midi": str(postjoin_gt),
                    "run_status": "complete",
                    "failure_reason": "",
                }
                if source_hash is not None:
                    row["source_npz_sha256"] = source_hash
                manifest = case_root / "offline.json"
                manifest.write_text(json.dumps([row]), encoding="utf-8")
                output = case_root / "output"
                with self.assertRaises(PreparationBlockedError):
                    prepare_matched_music_eval(
                        cohort_manifest=cohort,
                        offline_manifest=manifest,
                        output_dir=output,
                        expected_piece_count=1,
                        expected_seeds=("0",),
                    )
                result = json.loads(
                    (output / "prepared_manifest.json").read_text(encoding="utf-8")
                )
                self.assertIn(message, result["blockers"][0])
                self.assertEqual(
                    result["trials"][0]["cohort_source_npz_sha256"],
                    expected_hash,
                )
                self.assertFalse((output / "beat-offline").exists())

    def test_offline_out_of_window_and_empty_melody_are_rejected(self) -> None:
        cohort, _, _ = self._cohort()
        generated = self.root / "bad-offline" / "generated.mid"
        gt = self.root / "bad-offline" / "gt.mid"
        self._write_midi(
            generated,
            melody=[(60, 11.5, 12.5)],
            accompaniment=[(48, 1.0, 2.0)],
        )
        self._write_midi(
            gt,
            melody=[(60, 0.0, 1.0)],
            accompaniment=[(48, 0.0, 1.0)],
        )
        row = {
            "piece_id": "piece-a",
            "seed": "0",
            "system_id": "offline",
            "postjoin_generated_midi": str(generated),
            "postjoin_gt_midi": str(gt),
            "source_npz_sha256": self._cohort_source_npz_sha256(cohort),
            "run_status": "complete",
            "failure_reason": "",
        }
        offline = self.root / "bad-offline.json"
        offline.write_text(json.dumps([row]), encoding="utf-8")
        with self.assertRaises(PreparationBlockedError):
            prepare_matched_music_eval(
                cohort_manifest=cohort,
                offline_manifest=offline,
                output_dir=self.root / "bad-offline-output",
                expected_piece_count=1,
                expected_seeds=("0",),
            )

    def test_failed_offline_row_does_not_claim_a_cohort_hash_for_missing_gt(
        self,
    ) -> None:
        cohort, _, gt_full = self._cohort()
        offline = self.root / "failed-offline.json"
        offline.write_text(
            json.dumps(
                [
                    {
                        "piece_id": "piece-a",
                        "seed": "0",
                        "system_id": "beat-offline",
                        "postjoin_generated_midi": "",
                        "postjoin_gt_midi": "",
                        "source_npz_sha256": self._cohort_source_npz_sha256(cohort),
                        "status": "failed",
                        "failure_reason": "offline inference failed",
                    }
                ]
            ),
            encoding="utf-8",
        )
        output = self.root / "failed-offline-output"
        with self.assertRaises(PreparationBlockedError):
            prepare_matched_music_eval(
                cohort_manifest=cohort,
                offline_manifest=offline,
                output_dir=output,
                expected_piece_count=1,
                expected_seeds=("0",),
            )
        data = json.loads((output / "prepared_manifest.json").read_text())
        row = data["trials"][0]
        self.assertIsNone(row["source_gt_midi"])
        self.assertIsNone(row["source_gt_sha256"])
        self.assertEqual(Path(row["cohort_full_gt_midi"]), gt_full.resolve())
        self.assertEqual(row["cohort_full_gt_sha256"], self._sha256(gt_full))
        for field in (
            "all_trials_generated_midi",
            "all_trials_metric_gt_midi",
            "valid_only_generated_midi",
            "valid_only_metric_gt_midi",
        ):
            self.assertIsNone(row[field])
        with (output / "audit.csv").open(encoding="utf-8", newline="") as handle:
            audit_row = next(csv.DictReader(handle))
        self.assertEqual(audit_row["all_trials_generated_midi"], "")
        self.assertEqual(audit_row["all_trials_metric_gt_midi"], "")
        self.assertEqual(audit_row["valid_only_generated_midi"], "")
        self.assertEqual(audit_row["valid_only_metric_gt_midi"], "")
        self.assertFalse((output / "beat-offline").exists())


if __name__ == "__main__":
    unittest.main()
