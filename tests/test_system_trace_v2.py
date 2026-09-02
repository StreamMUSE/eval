from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from eval_toolkit.system_trace_v2 import (
    MANIFEST_REQUIRED_FIELDS,
    TraceValidationError,
    build_summary,
    discover_sessions,
    evaluate_manifest,
    evaluate_session,
    evaluate_sessions,
    load_manifest,
    main,
)


def deadline(tick: int, nominal: float, due: float, **extra: object) -> dict:
    return {
        "schema_version": 2,
        "record_type": "frame_deadline",
        "tick": tick,
        "nominal_tick_time_s": nominal,
        "deadline_time_s": due,
        "condition": "prompt_continuation",
        **extra,
    }


def span(start: int, stop: int, available: float, **extra: object) -> dict:
    return {
        "schema_version": 2,
        "record_type": "availability_span",
        "start_tick": start,
        "end_tick_exclusive": stop,
        "availability_time_s": available,
        "condition": "prompt_continuation",
        **extra,
    }


class SystemTraceV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def make_session(
        self,
        records: list[dict],
        *,
        prompt_length_ticks: int = 2,
        name: str = "session_a",
        continuation_mode: str = "prompt_continuation",
    ) -> Path:
        session = self.root / name
        session.mkdir()
        (session / "session_config.json").write_text(
            json.dumps(
                {
                    "session_id": name,
                    "prompt_length_ticks": prompt_length_ticks,
                    "continuation_mode": continuation_mode,
                    "ticks_per_beat": 4,
                }
            ),
            encoding="utf-8",
        )
        (session / "system_trace.jsonl").write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )
        return session

    def write_manifest(
        self,
        rows: list[dict[str, str]],
        *,
        name: str = "manifest.csv",
        fields: tuple[str, ...] = MANIFEST_REQUIRED_FIELDS,
    ) -> Path:
        path = self.root / name
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in fields})
        return path

    def manifest_row(
        self,
        *,
        piece_id: str,
        seed: str,
        system_id: str,
        melody_hash: str,
        session_dir: Path | str = "",
        run_status: str = "complete",
        failure_reason: str = "",
    ) -> dict[str, str]:
        return {
            "piece_id": piece_id,
            "seed": seed,
            "system_id": system_id,
            "session_dir": str(session_dir),
            "run_status": run_status,
            "melody_input_sha256": melody_hash,
            "failure_reason": failure_reason,
        }

    def make_timed_session(
        self,
        *,
        name: str,
        system_id: str,
        availability_delay_s: float | None,
    ) -> Path:
        records = [
            deadline(2, 100.0, 100.1, condition=system_id),
            deadline(3, 100.1, 100.2, condition=system_id),
        ]
        if availability_delay_s is not None:
            records.append(
                span(
                    2,
                    4,
                    100.0 + availability_delay_s,
                    condition=system_id,
                )
            )
        return self.make_session(
            records,
            name=name,
            continuation_mode=system_id,
        )

    def test_all_frames_on_time(self) -> None:
        session = self.make_session(
            [
                deadline(2, 10.0, 10.1),
                deadline(3, 10.1, 10.2),
                deadline(4, 10.2, 10.3),
                span(2, 5, 10.05),
            ]
        )
        result, frames = evaluate_session(session)
        self.assertEqual(result["window_frames"], 3)
        self.assertEqual(result["delivered_frames"], 3)
        self.assertEqual(result["missing_frames"], 0)
        self.assertEqual(result["isr_f"], 1.0)
        self.assertEqual(result["delivery_rate"], 1.0)
        self.assertAlmostEqual(result["ttfp_ms"], 50.0)
        self.assertEqual(result["staleness_p50_ms"], 0.0)
        self.assertTrue(all(frame["on_time"] for frame in frames))

    def test_partial_late_and_staleness_percentiles(self) -> None:
        session = self.make_session(
            [
                deadline(2, 20.0, 20.1),
                deadline(3, 20.1, 20.2),
                span(2, 3, 20.2),
                span(3, 4, 20.15),
            ]
        )
        result, _ = evaluate_session(session)
        self.assertEqual(result["delivery_rate"], 1.0)
        self.assertEqual(result["isr_f"], 0.5)
        self.assertAlmostEqual(result["staleness_p50_ms"], 50.0)
        self.assertAlmostEqual(result["staleness_p95_ms"], 95.0)

    def test_missing_frame_is_not_infinite_staleness(self) -> None:
        session = self.make_session(
            [
                deadline(2, 30.0, 30.1),
                deadline(3, 30.1, 30.2),
                span(2, 3, 30.05),
            ]
        )
        result, frames = evaluate_session(session)
        self.assertEqual(result["delivered_frames"], 1)
        self.assertEqual(result["missing_frames"], 1)
        self.assertEqual(result["delivery_rate"], 0.5)
        self.assertEqual(result["isr_f"], 0.5)
        self.assertIsNone(frames[1]["availability_time_s"])
        self.assertIsNone(frames[1]["staleness_ms"])
        self.assertEqual(result["staleness_p95_ms"], 0.0)

    def test_past_frame_late_arrival_is_delivered_but_not_on_time(self) -> None:
        session = self.make_session([deadline(2, 40.0, 40.1), span(2, 3, 42.0)])
        result, frames = evaluate_session(session)
        self.assertEqual(result["delivery_rate"], 1.0)
        self.assertEqual(result["isr_f"], 0.0)
        self.assertAlmostEqual(frames[0]["staleness_ms"], 1900.0)

    def test_overlapping_spans_use_earliest_availability(self) -> None:
        session = self.make_session(
            [
                deadline(2, 50.0, 50.5),
                deadline(3, 50.5, 51.0),
                span(2, 4, 50.7),
                span(2, 3, 50.2),
            ]
        )
        result, frames = evaluate_session(session)
        self.assertEqual(frames[0]["availability_time_s"], 50.2)
        self.assertEqual(frames[1]["availability_time_s"], 50.7)
        self.assertEqual(result["isr_f"], 1.0)

    def test_schema_one_is_rejected_explicitly(self) -> None:
        session = self.make_session(
            [
                {
                    "schema_version": 1,
                    "tick": 2,
                    "decision": "note",
                    "emitted_model_note_on_count": 1,
                }
            ]
        )
        with self.assertRaisesRegex(
            TraceValidationError, "schema_version 1 is unsupported"
        ):
            evaluate_session(session)

    def test_schema_version_must_be_integer_two(self) -> None:
        session = self.make_session(
            [
                {
                    **deadline(2, 55.0, 55.1),
                    "schema_version": 2.0,
                }
            ],
            name="float_schema",
        )
        with self.assertRaisesRegex(TraceValidationError, "schema_version must be 2"):
            evaluate_session(session)

    def test_ttfp_is_clamped_at_zero_and_null_when_missing(self) -> None:
        early = self.make_session(
            [deadline(2, 60.0, 60.1), span(2, 3, 59.5)], name="early"
        )
        missing = self.make_session([deadline(2, 60.0, 60.1)], name="missing")
        early_result, _ = evaluate_session(early)
        missing_result, _ = evaluate_session(missing)
        self.assertEqual(early_result["ttfp_ms"], 0.0)
        self.assertIsNone(missing_result["ttfp_ms"])

    def test_window_must_have_exactly_one_deadline_per_tick(self) -> None:
        gap = self.make_session(
            [deadline(2, 70.0, 70.1), deadline(4, 70.2, 70.3)], name="gap"
        )
        with self.assertRaisesRegex(TraceValidationError, "not continuous"):
            evaluate_session(gap)

        duplicate = self.make_session(
            [deadline(2, 70.0, 70.1), deadline(2, 70.0, 70.1)],
            name="duplicate",
        )
        with self.assertRaisesRegex(TraceValidationError, "duplicate frame_deadline"):
            evaluate_session(duplicate)

    def test_invalid_span_and_non_finite_time_are_rejected(self) -> None:
        invalid_span = self.make_session(
            [deadline(2, 80.0, 80.1), span(3, 3, 80.0)], name="invalid_span"
        )
        with self.assertRaisesRegex(TraceValidationError, "must be greater"):
            evaluate_session(invalid_span)

        non_finite = self.make_session(
            [deadline(2, float("nan"), 80.1)], name="non_finite"
        )
        with self.assertRaisesRegex(TraceValidationError, "finite number"):
            evaluate_session(non_finite)

    def test_explicit_window_override_and_outputs(self) -> None:
        self.make_session(
            [
                deadline(0, 90.0, 90.1),
                deadline(1, 90.1, 90.2),
                deadline(2, 90.2, 90.3),
                span(0, 3, 90.05),
            ],
            prompt_length_ticks=99,
        )
        output = self.root / "output"
        sessions = discover_sessions([], [self.root])
        rows = evaluate_sessions(
            sessions,
            output,
            observation_tick=0,
            end_tick=2,
            write_per_frame=True,
        )
        self.assertEqual(rows[0]["window_frames"], 2)
        for name in (
            "per_session.csv",
            "per_session.json",
            "per_frame.csv",
            "summary.json",
        ):
            self.assertTrue((output / name).is_file())
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["statistics"], "descriptive_only")
        self.assertFalse(summary["bootstrap_confidence_intervals"]["computed"])

    def test_summary_keeps_standard_and_prompt_continuation_separate(self) -> None:
        prompt_session = self.make_session(
            [
                deadline(2, 100.0, 100.1),
                span(2, 3, 100.05),
            ],
            name="prompt_session",
            continuation_mode="prompt_continuation",
        )
        standard_session = self.make_session(
            [
                deadline(2, 100.0, 100.1, condition="standard"),
                span(2, 3, 100.2, condition="standard"),
            ],
            name="standard_session",
            continuation_mode="standard",
        )
        prompt_row, prompt_frames = evaluate_session(prompt_session)
        standard_row, standard_frames = evaluate_session(standard_session)

        summary = build_summary(
            [prompt_row, standard_row], prompt_frames + standard_frames
        )

        self.assertEqual(summary["overall"]["purpose"], "audit_only")
        self.assertEqual(summary["overall"]["session_count"], 2)
        self.assertEqual(len(summary["groups"]), 2)
        prompt_group = summary["groups"][
            "condition=prompt_continuation__continuation_mode=prompt_continuation"
        ]
        standard_group = summary["groups"][
            "condition=standard__continuation_mode=standard"
        ]
        self.assertEqual(prompt_group["session_count"], 1)
        self.assertEqual(standard_group["session_count"], 1)
        self.assertEqual(prompt_group["metrics"]["isr_f"]["mean"], 1.0)
        self.assertEqual(standard_group["metrics"]["isr_f"]["mean"], 0.0)

    def test_table_staleness_pools_frames_not_session_percentiles(self) -> None:
        three_frame_session = self.make_session(
            [
                deadline(2, 10.0, 10.1),
                deadline(3, 10.1, 10.2),
                deadline(4, 10.2, 10.3),
                span(2, 3, 10.1),
                span(3, 4, 10.2),
                span(4, 5, 10.4),
            ],
            name="three_frames",
        )
        one_frame_session = self.make_session(
            [
                deadline(2, 20.0, 20.1),
                span(2, 3, 21.1),
            ],
            name="one_frame",
        )
        first_row, first_frames = evaluate_session(three_frame_session)
        second_row, second_frames = evaluate_session(one_frame_session)

        summary = build_summary([first_row, second_row], first_frames + second_frames)
        group = summary["groups"][
            "condition=prompt_continuation__continuation_mode=prompt_continuation"
        ]

        self.assertEqual(group["metrics"]["staleness_p50_ms"]["p50"], 500.0)
        self.assertEqual(group["table_metrics"]["staleness_p50_ms"], 50.0)
        self.assertEqual(group["table_metrics"]["staleness_p95_ms"], 865.0)
        self.assertEqual(group["table_metrics"]["isr_f"], 0.5)
        self.assertEqual(group["table_metrics"]["delivery_rate"], 1.0)
        self.assertEqual(group["table_metrics"]["missing_frames"], 0)

    def test_manifest_requires_columns_and_formal_grid_size(self) -> None:
        missing_column = self.write_manifest(
            [],
            name="missing_column.csv",
            fields=MANIFEST_REQUIRED_FIELDS[:-1],
        )
        with self.assertRaisesRegex(TraceValidationError, "missing required columns"):
            load_manifest(missing_column)

        rows = [
            self.manifest_row(
                piece_id="piece_01",
                seed="0",
                system_id=system_id,
                melody_hash="a" * 64,
                session_dir=f"session_{system_id}",
            )
            for system_id in ("system_a", "system_b")
        ]
        undersized = self.write_manifest(rows, name="undersized.csv")
        with self.assertRaisesRegex(TraceValidationError, "exactly 40 pieces"):
            load_manifest(undersized)

    def test_manifest_validates_every_required_row_field(self) -> None:
        base_rows = [
            self.manifest_row(
                piece_id="piece_01",
                seed="0",
                system_id=system_id,
                melody_hash="a" * 64,
                session_dir=f"session_{system_id}",
            )
            for system_id in ("system_a", "system_b")
        ]
        cases = (
            ("piece_id", "", "piece_id.*empty"),
            ("seed", "", "seed.*empty"),
            ("system_id", "", "system_id.*empty"),
            ("session_dir", "", "complete manifest rows require session_dir"),
            ("run_status", "unknown", "run_status must be one of"),
            ("melody_input_sha256", "abc", "exactly 64 hex digits"),
        )
        for index, (field, value, message) in enumerate(cases):
            with self.subTest(field=field):
                rows = [dict(row) for row in base_rows]
                rows[0][field] = value
                manifest = self.write_manifest(rows, name=f"invalid_field_{index}.csv")
                with self.assertRaisesRegex(TraceValidationError, message):
                    load_manifest(
                        manifest,
                        expected_piece_count=1,
                        expected_seed_count=1,
                    )

        missing_reason_rows = [dict(row) for row in base_rows]
        missing_reason_rows[0]["run_status"] = "missing"
        missing_reason_rows[0]["session_dir"] = ""
        missing_reason = self.write_manifest(
            missing_reason_rows, name="missing_failure_reason.csv"
        )
        with self.assertRaisesRegex(TraceValidationError, "require failure_reason"):
            load_manifest(
                missing_reason,
                expected_piece_count=1,
                expected_seed_count=1,
            )

    def test_manifest_rejects_incomplete_grid_and_piece_hash_mismatch(self) -> None:
        incomplete_rows = []
        for piece_id, hash_char in (("piece_01", "a"), ("piece_02", "b")):
            for system_id in ("system_a", "system_b"):
                if (piece_id, system_id) == ("piece_02", "system_b"):
                    continue
                incomplete_rows.append(
                    self.manifest_row(
                        piece_id=piece_id,
                        seed="0",
                        system_id=system_id,
                        melody_hash=hash_char * 64,
                        session_dir=f"{piece_id}_{system_id}",
                    )
                )
        incomplete = self.write_manifest(incomplete_rows, name="incomplete.csv")
        with self.assertRaisesRegex(TraceValidationError, "matched grid is incomplete"):
            load_manifest(
                incomplete,
                expected_piece_count=2,
                expected_seed_count=1,
            )

        mismatched_rows = [
            self.manifest_row(
                piece_id="piece_01",
                seed="0",
                system_id="system_a",
                melody_hash="a" * 64,
                session_dir="hash_a",
            ),
            self.manifest_row(
                piece_id="piece_01",
                seed="0",
                system_id="system_b",
                melody_hash="b" * 64,
                session_dir="hash_b",
            ),
        ]
        mismatched = self.write_manifest(mismatched_rows, name="mismatched.csv")
        with self.assertRaisesRegex(TraceValidationError, "differs within pieces"):
            load_manifest(
                mismatched,
                expected_piece_count=1,
                expected_seed_count=1,
            )

    def test_manifest_failed_and_missing_rows_block_primary_ci(self) -> None:
        rows = []
        for piece_id, hash_char in (("piece_01", "a"), ("piece_02", "b")):
            complete = self.make_timed_session(
                name=f"{piece_id}_system_a",
                system_id="system_a",
                availability_delay_s=0.05,
            )
            rows.append(
                self.manifest_row(
                    piece_id=piece_id,
                    seed="0",
                    system_id="system_a",
                    melody_hash=hash_char * 64,
                    session_dir=complete,
                )
            )
            status = "failed" if piece_id == "piece_01" else "missing"
            rows.append(
                self.manifest_row(
                    piece_id=piece_id,
                    seed="0",
                    system_id="system_b",
                    melody_hash=hash_char * 64,
                    run_status=status,
                    failure_reason=f"planned {status} trial",
                )
            )

        manifest = self.write_manifest(rows)
        output = self.root / "blocked_output"
        summary = evaluate_manifest(
            manifest,
            output,
            observation_tick=2,
            end_tick=4,
            bootstrap_replicates=20,
            expected_piece_count=2,
            expected_seed_count=1,
        )

        self.assertFalse(summary["manifest"]["primary_ci_eligible"])
        self.assertFalse(summary["bootstrap_confidence_intervals"]["computed"])
        self.assertEqual(summary["manifest"]["manifest_status_counts"]["failed"], 1)
        self.assertEqual(summary["manifest"]["manifest_status_counts"]["missing"], 1)
        self.assertEqual(summary["paired_system_differences"], {})
        with (output / "manifest_audit.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            audit = list(csv.DictReader(handle))
        self.assertEqual(len(audit), 4)
        self.assertEqual(
            {row["evaluation_status"] for row in audit},
            {"evaluated", "failed", "missing"},
        )

    def test_manifest_complete_schema_one_is_audited_and_blocks_ci(self) -> None:
        valid = self.make_timed_session(
            name="valid_system_a",
            system_id="system_a",
            availability_delay_s=0.05,
        )
        invalid = self.make_session(
            [
                {
                    "schema_version": 1,
                    "tick": 2,
                    "decision": "note",
                }
            ],
            name="schema_one_system_b",
            continuation_mode="system_b",
        )
        rows = [
            self.manifest_row(
                piece_id="piece_01",
                seed="0",
                system_id="system_a",
                melody_hash="a" * 64,
                session_dir=valid,
            ),
            self.manifest_row(
                piece_id="piece_01",
                seed="0",
                system_id="system_b",
                melody_hash="a" * 64,
                session_dir=invalid,
            ),
        ]
        manifest = self.write_manifest(rows)
        output = self.root / "invalid_complete_output"
        summary = evaluate_manifest(
            manifest,
            output,
            observation_tick=2,
            end_tick=4,
            bootstrap_replicates=20,
            expected_piece_count=1,
            expected_seed_count=1,
        )

        self.assertFalse(summary["manifest"]["primary_ci_eligible"])
        self.assertEqual(
            summary["manifest"]["evaluation_status_counts"]["invalid_complete"],
            1,
        )
        with (output / "manifest_audit.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            audit = list(csv.DictReader(handle))
        invalid_row = next(
            row for row in audit if row["evaluation_status"] == "invalid_complete"
        )
        self.assertIn(
            "schema_version 1 is unsupported", invalid_row["evaluation_error"]
        )

    def test_piece_cluster_bootstrap_is_matched_and_reports_paired_differences(
        self,
    ) -> None:
        rows = []
        delays = {
            ("piece_01", "0", "system_a"): 0.05,
            ("piece_01", "1", "system_a"): 0.15,
            ("piece_02", "0", "system_a"): 0.25,
            ("piece_02", "1", "system_a"): 0.35,
            ("piece_01", "0", "system_b"): 0.15,
            ("piece_01", "1", "system_b"): 0.25,
            ("piece_02", "0", "system_b"): 0.35,
            ("piece_02", "1", "system_b"): 0.45,
        }
        for piece_id, hash_char in (("piece_01", "a"), ("piece_02", "b")):
            for seed in ("0", "1"):
                for system_id in ("system_a", "system_b"):
                    session = self.make_timed_session(
                        name=f"{piece_id}_{seed}_{system_id}",
                        system_id=system_id,
                        availability_delay_s=delays[(piece_id, seed, system_id)],
                    )
                    rows.append(
                        self.manifest_row(
                            piece_id=piece_id,
                            seed=seed,
                            system_id=system_id,
                            melody_hash=hash_char * 64,
                            session_dir=session,
                        )
                    )

        manifest = self.write_manifest(rows)
        output = self.root / "bootstrap_output"
        summary = evaluate_manifest(
            manifest,
            output,
            observation_tick=2,
            end_tick=4,
            bootstrap_replicates=200,
            bootstrap_seed=17,
            write_per_frame=True,
            expected_piece_count=2,
            expected_seed_count=2,
        )

        self.assertTrue(summary["manifest"]["primary_ci_eligible"])
        self.assertTrue(summary["bootstrap_confidence_intervals"]["computed"])
        self.assertEqual(set(summary["groups"]), {"system_a", "system_b"})
        self.assertEqual(
            summary["groups"]["system_a"]["table_metrics"]["ttfp_p50_ms"],
            200.0,
        )
        self.assertEqual(
            summary["groups"]["system_b"]["table_metrics"]["ttfp_p50_ms"],
            300.0,
        )
        system_ci = summary["groups"]["system_a"]["bootstrap_ci"]
        self.assertEqual(system_ci["ttfp_p50_ms"]["valid_replicates"], 200)
        self.assertLessEqual(
            system_ci["ttfp_p50_ms"]["ci95_low"],
            system_ci["ttfp_p50_ms"]["estimate"],
        )
        self.assertGreaterEqual(
            system_ci["ttfp_p50_ms"]["ci95_high"],
            system_ci["ttfp_p50_ms"]["estimate"],
        )
        paired = summary["paired_system_differences"]["system_a__minus__system_b"]
        paired_ttfp = paired["metrics"]["ttfp_p50_ms"]
        self.assertEqual(paired_ttfp["estimate"], -100.0)
        self.assertEqual(paired_ttfp["ci95_low"], -100.0)
        self.assertEqual(paired_ttfp["ci95_high"], -100.0)
        self.assertEqual(paired_ttfp["valid_replicates"], 200)
        self.assertTrue((output / "per_frame.csv").is_file())

    def test_manifest_cli_is_exclusive_and_requires_fixed_window(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as exclusive:
            main(
                [
                    "--manifest",
                    str(self.root / "manifest.csv"),
                    "--root",
                    str(self.root),
                    "--observation-tick",
                    "2",
                    "--end-tick",
                    "4",
                    "--output-dir",
                    str(self.root / "output"),
                ]
            )
        self.assertEqual(exclusive.exception.code, 2)
        self.assertIn("cannot be combined", stderr.getvalue())

        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as fixed_window:
            main(
                [
                    "--manifest",
                    str(self.root / "manifest.csv"),
                    "--observation-tick",
                    "2",
                    "--output-dir",
                    str(self.root / "output"),
                ]
            )
        self.assertEqual(fixed_window.exception.code, 2)
        self.assertIn("requires explicit", stderr.getvalue())

    def test_legacy_session_cli_remains_descriptive_only(self) -> None:
        session = self.make_timed_session(
            name="legacy_session",
            system_id="system_a",
            availability_delay_s=0.05,
        )
        output = self.root / "legacy_cli_output"
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            return_code = main(
                [
                    "--session-dir",
                    str(session),
                    "--observation-tick",
                    "2",
                    "--end-tick",
                    "4",
                    "--output-dir",
                    str(output),
                ]
            )
        self.assertEqual(return_code, 0)
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["metric_contract"], "matched_system_metrics_v2")
        self.assertEqual(summary["statistics"], "descriptive_only")
        self.assertFalse(summary["bootstrap_confidence_intervals"]["computed"])


if __name__ == "__main__":
    unittest.main()
