import json
import unittest
from pathlib import Path
from unittest.mock import patch

from threshold_lab import (
    best_by_f1,
    confusion,
    load_records,
    metrics,
    render_html,
    sweep,
)


class CapturingOutput:
    def __init__(self):
        self.document = ""
        self.encoding = None

    def write_text(self, document, *, encoding):
        self.document = document
        self.encoding = encoding


class ThresholdTests(unittest.TestCase):
    def setUp(self):
        self.records = [(1, 0.9), (1, 0.8), (0, 0.7), (0, 0.1)]

    def test_confusion_and_metrics(self):
        matrix = confusion(self.records, 0.75)
        self.assertEqual(matrix, {"tp": 2, "fp": 0, "tn": 2, "fn": 0})
        self.assertEqual(
            metrics({"tp": 1, "fp": 1, "tn": 1, "fn": 1}),
            {"precision": 0.5, "recall": 0.5, "f1": 0.5, "accuracy": 0.5},
        )

    def test_sweep_is_deterministic_and_includes_both_boundaries(self):
        thresholds = [row["threshold"] for row in sweep(self.records, 0.3)]
        self.assertEqual(thresholds, [0.0, 0.3, 0.6, 0.9, 1.0])

    def test_best_f1_has_deterministic_tie_breaking(self):
        tied = [
            {"threshold": 0.6, "f1": 0.8, "recall": 0.8},
            {"threshold": 0.4, "f1": 0.8, "recall": 0.8},
        ]
        self.assertEqual(best_by_f1(tied)["threshold"], 0.4)

    def test_invalid_probability_and_label_are_rejected(self):
        path = Path("records.json")
        for records in (
            [{"label": 1, "score": 1.2}],
            [{"label": 1, "score": float("nan")}],
            [{"label": 2, "score": 0.5}],
            [{"label": True, "score": 0.5}],
            [{"score": 0.5}],
        ):
            with self.subTest(records=records):
                with patch.object(Path, "read_text", return_value=json.dumps(records)):
                    with self.assertRaises(ValueError):
                        load_records(path)

    def test_empty_and_over_limit_input_are_rejected(self):
        path = Path("records.json")
        with patch.object(Path, "read_text", return_value="[]"):
            with self.assertRaises(ValueError):
                load_records(path)
        with patch("threshold_lab.MAX_ROWS", 2):
            data = json.dumps([{"label": 0, "score": 0.1}] * 3)
            with patch.object(Path, "read_text", return_value=data):
                with self.assertRaises(ValueError):
                    load_records(path)

    def test_invalid_direct_record_and_step_are_rejected(self):
        with self.assertRaises(ValueError):
            confusion([(1, 2.0)], 0.5)
        for step in (0, -0.1, float("inf"), True):
            with self.assertRaises(ValueError):
                sweep(self.records, step)

    def test_html_report_is_self_contained(self):
        output = CapturingOutput()
        render_html(sweep(self.records, 0.5), output)
        self.assertEqual(output.encoding, "utf-8")
        self.assertIn("AI Threshold Playground", output.document)
        self.assertIn("<svg", output.document)
        self.assertNotIn("<script src=", output.document)
        self.assertNotIn("http://", output.document)
        self.assertNotIn("https://", output.document)


if __name__ == "__main__":
    unittest.main()
