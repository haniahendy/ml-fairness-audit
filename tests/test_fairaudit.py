import json
import unittest

from fairaudit import audit, group_metrics, render_json, render_text

# Hand-computed fixture, threshold 0.5.
# Group A: TPR 3/4, FPR 1/4, selection 4/8, precision 3/4.
# Group B: TPR 1/4, FPR 2/4, selection 3/8, precision 1/3.
ROWS = (
    [(1, 1, "A")] * 3 + [(1, 0, "A")]
    + [(0, 1, "A")] + [(0, 0, "A")] * 3
    + [(1, 1, "B")] + [(1, 0, "B")] * 3
    + [(0, 1, "B")] * 2 + [(0, 0, "B")] * 2
)


class TestGroupMetrics(unittest.TestCase):
    def test_counts(self):
        groups = group_metrics(ROWS)
        self.assertEqual(groups["A"].n, 8)
        self.assertEqual(groups["A"].positives, 4)
        self.assertEqual(groups["B"].negatives, 4)

    def test_rates(self):
        groups = group_metrics(ROWS)
        self.assertAlmostEqual(groups["A"].tpr, 0.75)
        self.assertAlmostEqual(groups["A"].fpr, 0.25)
        self.assertAlmostEqual(groups["A"].selection_rate, 0.5)
        self.assertAlmostEqual(groups["A"].precision, 0.75)
        self.assertAlmostEqual(groups["B"].tpr, 0.25)
        self.assertAlmostEqual(groups["B"].fpr, 0.5)
        self.assertAlmostEqual(groups["B"].selection_rate, 0.375)
        self.assertAlmostEqual(groups["B"].precision, 1 / 3)

    def test_empty_cell_rates_are_none(self):
        rows = [(1, 1, "C"), (1, 0, "C")]
        groups = group_metrics(rows)
        self.assertIsNone(groups["C"].fpr)


class TestAudit(unittest.TestCase):
    def test_reference_defaults_to_largest_group(self):
        result = audit(ROWS + [(0, 0, "C")] * 9)
        self.assertEqual(result.reference, "C")

    def test_gaps_against_reference(self):
        result = audit(ROWS, reference="A")
        gap = result.gaps["B"]
        self.assertAlmostEqual(gap.demographic_parity_diff, 0.375 - 0.5)
        self.assertAlmostEqual(gap.equal_opportunity_diff, 0.25 - 0.75)
        self.assertAlmostEqual(gap.equalized_odds_max_abs_diff, 0.5)
        self.assertAlmostEqual(gap.predictive_parity_diff, 1 / 3 - 0.75)

    def test_reference_gap_is_zero(self):
        result = audit(ROWS, reference="A")
        gap = result.gaps["A"]
        self.assertAlmostEqual(gap.demographic_parity_diff, 0.0)
        self.assertAlmostEqual(gap.equalized_odds_max_abs_diff, 0.0)

    def test_unknown_reference_raises(self):
        with self.assertRaises(ValueError):
            audit(ROWS, reference="ZZZ")

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            audit([])


class TestReports(unittest.TestCase):
    def test_json_roundtrip(self):
        result = audit(ROWS, reference="A")
        payload = json.loads(render_json(result))
        self.assertEqual(payload["reference_group"], "A")
        self.assertEqual(payload["groups"]["B"]["n"], 8)
        self.assertAlmostEqual(
            payload["gaps_vs_reference"]["B"]["equal_opportunity_diff"], -0.5
        )

    def test_text_skips_reference_in_gap_table(self):
        result = audit(ROWS, reference="A")
        text = render_text(result)
        self.assertIn("reference group: A", text)
        gap_section = text.split("\n\n")[2]
        self.assertNotIn("A ", gap_section)
        self.assertIn("B ", gap_section)


if __name__ == "__main__":
    unittest.main()
