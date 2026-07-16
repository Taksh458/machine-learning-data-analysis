import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.run_analysis import build_time_series


class DataWorkflowTests(unittest.TestCase):
    def test_time_series_has_requested_rows(self):
        frame = build_time_series(50)
        self.assertEqual(len(frame), 50)

    def test_time_series_contains_expected_columns(self):
        frame = build_time_series(20)
        self.assertEqual(list(frame.columns), ["date", "value"])

    def test_generation_is_reproducible(self):
        first = build_time_series(15)
        second = build_time_series(15)
        self.assertTrue(first.equals(second))


if __name__ == "__main__":
    unittest.main()
