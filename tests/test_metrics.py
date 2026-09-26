"""Tests for the metrics a confidence gate is judged by.

The metrics here are the ones a product gates on, so the contracts defended are
the ones that change what ships: what a threshold accepts, what an error budget
releases, and whether the number on the gate is the number the runtime computes.
"""

import math
import unittest

import numpy as np

from flint import metrics

try:  # the engine is optional: these metrics must be checkable without it
    from laya import common as _laya_common
except ImportError:  # pragma: no cover - depends on the environment
    _laya_common = None


def _entropy_confidence(row):
    """`1 - H(p) / log k` with `k` the row's own option count."""
    p = np.asarray(row, dtype=np.float64)
    return 1.0 - float(-(p * np.log(p)).sum()) / math.log(len(p))


def _is_nan(value):
    return isinstance(value, float) and math.isnan(value)


class ConfidenceTests(unittest.TestCase):
    def test_uniform_rows_are_zero_and_one_hot_rows_are_one(self):
        """Confidence is a normalised entropy, not the top probability.

        A uniform row over seven options carries 1/7 on its best option and no
        information about which one it is, so it must read 0 — a gate keyed on
        the top probability would read 0.14 and accept it.
        """
        cases = {
            "uniform over two": [0.5, 0.5],
            "uniform over three": [1 / 3, 1 / 3, 1 / 3],
            "uniform over seven": [1 / 7] * 7,
            "one hot over three": [0.0, 1.0, 0.0],
        }
        expected = {
            "uniform over two": 0.0,
            "uniform over three": 0.0,
            "uniform over seven": 0.0,
            "one hot over three": 1.0,
        }
        for name, row in cases.items():
            with self.subTest(row=name):
                # Nine places, not twelve: the zeros are clipped to 1e-12 before
                # the log, so a one-hot row lands a few 1e-11 below 1.
                self.assertAlmostEqual(float(metrics.confidence([row])[0]), expected[name], places=9)

    def test_a_row_is_scored_with_its_own_option_count(self):
        """Padding a row out to a wider matrix must not change, or be used as, its k.

        Confidence is normalised by `log k`, so pooling a 2-option question into
        a 10-column matrix does not merely add zeros — it changes the number the
        gate reads. The runtime scores each question at its own width, so the
        ragged value is the correct one and the padded value is the bug.
        """
        alone = float(metrics.confidence([[0.9, 0.1]])[0])
        padded = float(metrics.confidence(np.array([[0.9, 0.1] + [0.0] * 8]))[0])

        self.assertAlmostEqual(alone, _entropy_confidence([0.9, 0.1]), places=12)
        self.assertNotEqual(alone, padded)
        # Neighbours of a different width must not move the row either.
        mixed = metrics.confidence([[0.9, 0.1], [0.2] * 7, [1.0, 0.0, 0.0]])
        self.assertAlmostEqual(float(mixed[0]), alone, places=12)


class TemperatureTests(unittest.TestCase):
    def test_temperature_never_moves_the_answer_and_keeps_rows_normalised(self):
        """A temperature rescales confidence; it cannot change what was picked."""
        rows = np.array([
            [0.50, 0.30, 0.20],
            [0.34, 0.33, 0.33],
            [0.70, 0.29, 0.01],
            [0.90, 0.05, 0.05],
            [0.40, 0.35, 0.25],
        ])
        expected = np.argmax(rows, axis=1)

        for temperature in (0.25, 0.5, 0.9, 1.0, 1.5, 3.0, 10.0):
            with self.subTest(temperature=temperature):
                scaled = np.asarray(metrics.apply_temperature(rows, temperature))
                np.testing.assert_array_equal(np.argmax(scaled, axis=1), expected)
                np.testing.assert_allclose(scaled.sum(axis=1), np.ones(len(rows)))


class SelectiveMetricsTests(unittest.TestCase):
    # Confidence order: 0.919, 0.531, 0.029, 0.007. Only the third is wrong.
    ROWS = [[0.99, 0.01], [0.9, 0.1], [0.6, 0.4], [0.55, 0.45]]
    LABELS = [0, 0, 1, 0]

    def test_coverage_at_error_holds_the_budget_at_both_ends(self):
        """A budget of 0 releases only the unbroken correct prefix; a budget of 1
        releases everything; and a perfect model releases everything either way."""
        rows, labels = self.ROWS, self.LABELS

        coverage, threshold = metrics.coverage_at_error(rows, labels, 0.0)
        self.assertAlmostEqual(coverage, 0.5)
        # The returned threshold must reproduce the prefix when deployed.
        self.assertAlmostEqual(threshold, float(metrics.confidence(rows)[1]))

        self.assertAlmostEqual(metrics.coverage_at_error(rows, labels, 1.0)[0], 1.0)
        self.assertAlmostEqual(metrics.coverage_at_error(rows, labels, 0.25)[0], 1.0)
        self.assertAlmostEqual(metrics.coverage_at_error(rows, [0, 0, 0, 0], 0.0)[0], 1.0)

    def test_a_threshold_that_accepts_nothing_has_no_error_rate(self):
        """An empty accepted set must report nan, not 0.0, or it reads as perfect."""
        coverage, error = metrics.gated_error(self.ROWS, self.LABELS, 1.0)
        self.assertEqual(coverage, 0.0)
        self.assertTrue(_is_nan(error))

        # The contrast that makes the nan worth having: a non-empty, all-correct
        # accepted set is the one that is allowed to report 0.0.
        accepted, error = metrics.gated_error([[0.99, 0.01]], [0], 0.5)
        self.assertEqual(accepted, 1.0)
        self.assertEqual(error, 0.0)

    def test_select_temperature_releases_more_than_the_unscaled_checkpoint(self):
        """Three peaked rows the model gets wrong outrank six flat rows it gets right.

        At T=1 the checkpoint can release nothing under a 25% error budget — the
        three wrong rows are the most confident in the set. Softening to T=1.4
        moves the six correct rows above them, and eight of nine decisions can
        ship. That gap is the reason the search exists; a version that always
        returned 1.0 would leave the budget unspent.
        """
        peaked_wrong = [0.85, 0.075, 0.075]
        flat_right = [0.5512, 0.1323, 0.0188, 0.0052, 0.0032, 0.0753, 0.0002, 0.2135, 0.0003]
        rows = [peaked_wrong] * 3 + [flat_right] * 6
        labels = [1] * 3 + [0] * 6
        budget = 0.25

        unscaled, _ = metrics.coverage_at_error(rows, labels, budget)
        temperature, coverage = metrics.select_temperature(rows, labels, budget)

        self.assertNotEqual(temperature, 1.0)
        self.assertGreater(coverage, unscaled)
        self.assertEqual(metrics.accuracy(metrics.apply_temperature(rows, temperature), labels),
                         metrics.accuracy(rows, labels))

    def test_paired_confidence_drop_follows_the_sign_of_the_change(self):
        """Losing the deciding evidence lowers confidence, and the mean says so."""
        dropped = metrics.paired_confidence_drop(
            unknown_probs=[[0.5, 0.5], [0.5, 0.5]],
            control_probs=[[0.9, 0.1], [0.99, 0.01]],
        )
        self.assertEqual(dropped["n_pairs"], 2)
        self.assertGreater(dropped["mean_drop"], 0.0)
        self.assertEqual(dropped["share_lower"], 1.0)
        self.assertEqual(dropped["unknown_at_0_9"], 0.0)

        raised = metrics.paired_confidence_drop(
            unknown_probs=[[0.99, 0.01]],
            control_probs=[[0.5, 0.5]],
        )
        self.assertLess(raised["mean_drop"], 0.0)
        self.assertEqual(raised["share_lower"], 0.0)

    def test_paired_confidence_drop_rejects_unpaired_sets(self):
        with self.assertRaises(ValueError):
            metrics.paired_confidence_drop([[0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]])

    def test_empty_inputs_report_rather_than_raise(self):
        """No decisions is a result to report, not an exception to crash a report on."""
        self.assertTrue(_is_nan(metrics.accuracy([], [])))

        coverage, error = metrics.gated_error([], [], 0.5)
        self.assertEqual(coverage, 0.0)
        self.assertTrue(_is_nan(error))

        coverage, threshold = metrics.coverage_at_error([], [], 0.5)
        self.assertEqual(coverage, 0.0)
        self.assertTrue(_is_nan(threshold))

        error, threshold = metrics.risk_at_coverage([], [], 0.5)
        self.assertTrue(_is_nan(error))
        self.assertTrue(_is_nan(threshold))

        paired = metrics.paired_confidence_drop([], [])
        self.assertEqual(paired["n_pairs"], 0)
        self.assertTrue(_is_nan(paired["mean_drop"]))


class CalibrationTests(unittest.TestCase):
    def test_constant_confidence_reduces_to_the_gap_against_accuracy(self):
        """With one confidence for every row there is nothing to bin.

        Reporting 0.0 or dividing by an empty bin width would both hide a model
        that is 50% accurate and says 0.5 on every row it gets wrong.
        """
        rows = [[0.75, 0.25], [0.75, 0.25], [0.75, 0.25], [0.75, 0.25]]
        conf = [0.5, 0.5, 0.5, 0.5]

        half_right = metrics.expected_calibration_error(rows, [0, 1, 0, 1], conf=conf)
        all_right = metrics.expected_calibration_error(rows, [0, 0, 0, 0], conf=conf)

        self.assertAlmostEqual(half_right, 0.0, places=12)
        self.assertAlmostEqual(all_right, 0.5, places=12)

    @unittest.skipIf(_laya_common is None, "the engine is not installed")
    def test_ece_agrees_with_the_engine_on_the_same_confidence(self):
        """The local copy exists to be checkable, not to disagree.

        Aligned the way the engine's own scorer is called — top probability as
        the confidence, 0/1 correctness as the outcome. The two bin differently
        (observed range here, fixed [0, 1] there) so an exact match is not the
        contract; a disagreement wider than a bin's worth of rounding is, and
        that is what the tolerance below fails on.
        """
        # A hundred rows where the model's confidence and its accuracy agree, and
        # twenty where it is sure and entirely wrong: a calibration gap that is
        # neither zero nor spread evenly over the bins.
        rows = [[0.9, 0.1]] * 53 + [[0.99, 0.01]] * 20 + [[0.9, 0.1]] * 47
        labels = [0] * 53 + [1] * 20 + [1] * 47

        conf = metrics.top_probability(rows)
        correct = np.array([int(np.argmax(r)) == y for r, y in zip(rows, labels)], dtype=np.float64)

        mine = metrics.expected_calibration_error(rows, labels, bins=15, conf=conf)
        theirs = _laya_common.ece_score(conf, correct, bins=15)

        self.assertAlmostEqual(mine, theirs, delta=0.05)


if __name__ == "__main__":
    unittest.main()
