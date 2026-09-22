import unittest
import numpy as np
from run import auc, direction, distress_or_repetition, sample_balanced, validate_generation


class BenchmarkTests(unittest.TestCase):
    def test_coefficients_fail_closed(self):
        for percent in [1, .001, -5.001, float("inf"), float("nan")]:
            with self.assertRaises(ValueError):
                validate_generation(percent, [1, 0], 10)
        for percent in [0, -1, -2, -5]:
            np.testing.assert_allclose(validate_generation(percent, [1, 0], 10), [percent/10, 0])

    def test_auc_ties(self):
        self.assertEqual(auc([1, 2], [0, 1]), .875)
        self.assertEqual(auc([1], [1]), .5)

    def test_pca_removes_control_axis(self):
        x = np.array([[3., 2], [5, 2], [-3, 0], [3, 0]])
        v, count = direction(x, np.array([True, True, False, False]))
        self.assertEqual(count, 1)
        np.testing.assert_allclose(v, [0, 1], atol=1e-12)

    def test_stop_heuristics(self):
        self.assertIsNone(distress_or_repetition("The notebook rests on a shelf."))
        self.assertIsNotNone(distress_or_repetition("I feel utterly worthless."))
        self.assertIsNotNone(distress_or_repetition("I feel: overwhelmed."))
        self.assertIsNotNone(distress_or_repetition("one two three " * 10))

    def test_balanced_sample_independent_of_order(self):
        records = [{"category": c, "set": i, "prompt": str(i)} for c in ["a", "b"] for i in range(8)]
        self.assertEqual(sample_balanced(records, 4), sample_balanced(list(reversed(records)), 4))


if __name__ == "__main__":
    unittest.main()
