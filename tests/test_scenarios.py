"""Tests for turning a labelled corpus into decision records.

The three constructions look like data plumbing and are not: they are where
"option order carries no information" and "the stored label has the type the
engine reads" are enforced, and both fail silently in training.
"""

import unittest

from flint import scenarios

LABELS = ["alpha", "beta", "gamma", "delta"]


def _choices(items, **kwargs):
    return scenarios.choice_records(items, LABELS, **kwargs)


def _question(record, key):
    return record["questions"][key]


class ChoiceRecordTests(unittest.TestCase):
    def test_label_key_holds_the_true_label_text(self):
        """The stored answer is the key that landed on the true label's text."""
        items = [("item %d" % i, i % len(LABELS)) for i in range(12)]

        records = _choices(items, width=3, seed=0)

        for record, (text, true_index) in zip(records, items):
            question = _question(record, "pick")
            criteria = question["criteria"]
            with self.subTest(item=text):
                self.assertIn(question["label"], criteria)
                self.assertEqual(criteria[question["label"]], LABELS[true_index])
                self.assertEqual(len(criteria), 3)

    def test_option_order_varies_within_and_across_seeds(self):
        """A fixed answer position is exactly the cue the shuffle exists to remove."""
        items = [("item %d" % i, i % len(LABELS)) for i in range(12)]

        first = _choices(items, width=3, seed=0)
        second = _choices(items, width=3, seed=1)

        positions = [_question(r, "pick")["label"] for r in first]
        self.assertGreater(len(set(positions)), 1)
        self.assertNotEqual([_question(r, "pick")["criteria"] for r in first],
                            [_question(r, "pick")["criteria"] for r in second])

    def test_none_option_is_offered_but_never_answered(self):
        """`with_none` adds an option the model must place probability on, not a lie."""
        items = [("item %d" % i, i % len(LABELS)) for i in range(12)]

        records = _choices(items, width=3, seed=0, with_none=True)

        for record, (text, true_index) in zip(records, items):
            question = _question(record, "pick")
            criteria = question["criteria"]
            foreign = [k for k, v in criteria.items() if v not in LABELS]
            with self.subTest(item=text):
                # Exactly one option is outside the label space: the "none" option.
                self.assertEqual(len(foreign), 1)
                self.assertNotIn(question["label"], foreign)
                self.assertEqual(criteria[question["label"]], LABELS[true_index])

    def test_out_of_range_arguments_are_rejected(self):
        """A width or a label index the label space cannot satisfy is an error."""
        cases = {
            "width wider than the label space": lambda: _choices([("a", 0)], width=5),
            "width below a real choice": lambda: _choices([("a", 0)], width=1),
            "label index outside the label space": lambda: _choices([("a", 7)], width=3),
            "empty label space": lambda: scenarios.choice_records([("a", 0)], [], width=2),
        }
        for name, call in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(ValueError):
                    call()


class NoulRecordTests(unittest.TestCase):
    def test_label_is_a_boolean(self):
        """The engine reads a single yes probability, so the label is not 0/1."""
        items = [("holds", True), ("does not hold", False), ("also holds", True)]

        records = scenarios.noul_records(items)

        for record, (text, expected) in zip(records, items):
            label = _question(record, "holds")["label"]
            with self.subTest(item=text):
                self.assertIsInstance(label, bool)
                self.assertIs(label, expected)


class ScoreRecordTests(unittest.TestCase):
    def test_label_is_the_index_into_the_scale(self):
        """A scale that does not start at zero must not leak its values into labels."""
        items = [("low", 2), ("high", 9), ("middle", 5)]

        records = scenarios.score_records(items, levels=[2, 5, 9])

        for record, (text, level) in zip(records, items):
            question = _question(record, "rate")
            with self.subTest(item=text):
                self.assertEqual(question["label"], [2, 5, 9].index(level))
                self.assertEqual(len(question["criteria"]), 3)

    def test_level_outside_the_scale_is_rejected(self):
        cases = {
            "level between two scale points": lambda: scenarios.score_records([("a", 4)], levels=[2, 5, 9]),
            "level below the scale": lambda: scenarios.score_records([("a", 0)], levels=[2, 5, 9]),
            "levels out of order": lambda: scenarios.score_records([("a", 2)], levels=[9, 5, 2]),
            "single level is not a scale": lambda: scenarios.score_records([("a", 2)], levels=[2]),
        }
        for name, call in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(ValueError):
                    call()


if __name__ == "__main__":
    unittest.main()
