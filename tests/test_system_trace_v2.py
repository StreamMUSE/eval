from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eval_toolkit.system_trace_v2 import (
    TraceValidationError,
    build_summary,
    discover_sessions,
    evaluate_session,
    evaluate_sessions,
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
        prompt_row, _ = evaluate_session(prompt_session)
        standard_row, _ = evaluate_session(standard_session)

        summary = build_summary([prompt_row, standard_row])

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


if __name__ == "__main__":
    unittest.main()
