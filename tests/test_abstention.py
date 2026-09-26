"""Tests for the paired evidence-deletion measurement.

What is defended here is the suite the measurement accepts and the way it pairs
it, because both fail quietly. A ``control_id`` that names nothing leaves a
partial pairing that still reports a drop over a subset the caller did not pass.
A pairing read off adjacency instead of ``_meta.id`` reports a number about the
record order rather than about the model. And the family split is the difference
between a drop the mixture handed the model and one it learned: pooling the two
publishes the covered families as general behaviour.

The checkpoint is a double serving hand-written distributions, so every expected
number below is arithmetic over the constants, not a property of a model.
"""

import sys
import types
import unittest
from unittest import mock

from flint import evaluate, metrics

# Two-option distributions worth naming: FLAT is 0.0 confidence, PEAKED is
# 0.919207 and [0.6, 0.4] is 0.029049 — entropy confidence, `1 - H(p) / log 2`.
FLAT = [0.5, 0.5]
PEAKED = [0.99, 0.01]


def _noul_question():
    """A yes/no question; its options are `false` then `true`."""
    return {"type": "noul"}


def _noul_probs(false, true):
    """A served distribution for a yes/no question, keyed as the engine serves it."""
    return {"false": false, "true": true}


def _record(rid, question=None, questions=None, family=None, control_id=None):
    """One suite record: a single question plus the pairing bookkeeping in `_meta`."""
    meta = {"id": rid}
    if control_id is not None:
        meta["control_id"] = control_id
    if family is not None:
        meta["family"] = family
    if questions is None:
        questions = {"q": _noul_question() if question is None else question}
    return {"_meta": meta, "questions": questions}


def _inference_stub():
    """The label lookup `per_question_rows` imports lazily.

    The real `flint.inference` imports torch, which this file must not need, and
    the label lookup is not what is under test here.
    """
    module = types.ModuleType("flint.inference")
    module.answer_index = lambda qtype, criteria, label: None
    return module


class _StubCheckpoint:
    """A checkpoint double whose answers are looked up by record id.

    Keeping the suite and the served distributions apart is what lets a test
    reorder `records` and still know which distribution the checkpoint will serve
    for each record.
    """

    def __init__(self):
        self.records = []
        self._served = {}

    def add(self, record, probabilities=None):
        """Append `record`, serving `probabilities` for its first question.

        A record added without them is served nothing, which is what a checkpoint
        answers when it cannot build the item at all.
        """
        self.records.append(record)
        rid = (record.get("_meta") or {}).get("id")
        if probabilities is not None and rid is not None:
            self._served[rid] = {next(iter(record["questions"])): probabilities}
        return self

    def pair(self, rid, unknown, control, family=None):
        """Add an unanswerable `rid:unknown` and the intact record `rid` it names.

        `unknown` and `control` are `[false, true]` pairs; the unknown record
        carries the `control_id` that the pairing has to follow.
        """
        question = _noul_question()
        self.add(_record(rid + ":unknown", question=question, family=family, control_id=rid),
                 _noul_probs(*unknown))
        self.add(_record(rid, question=question, family=family), _noul_probs(*control))
        return self

    def answer(self, records):
        """Served answers in the order asked for, keyed by question key.

        A record with nothing registered answers with an empty dict, as the real
        checkpoint does for an item the engine could not build.
        """
        return [{key: {"choice": "a", "probabilities": dict(probs), "confidence": 0.0}
                 for key, probs in self._served.get(r["_meta"]["id"], {}).items()}
                for r in records]


class _DropAssertions(unittest.TestCase):
    """Report dicts are compared field by field, so a mismatch reads as a name."""

    def assertDrop(self, got, expected, places=12):
        self.assertEqual(got["n_pairs"], expected["n_pairs"])
        for field in ("mean_drop", "share_lower", "unknown_at_0_9", "control_at_0_9"):
            self.assertAlmostEqual(got[field], expected[field], places=places, msg=field)


class DropDirectionTests(_DropAssertions):
    def test_confidence_falling_on_the_unanswerable_half_is_a_positive_drop(self):
        """The metric is control minus unknown, so evidence deletion that lowers
        confidence has to land above zero; the inverse reports every model as a
        model that answers better without the evidence."""
        checkpoint = _StubCheckpoint().pair("q1", unknown=FLAT, control=PEAKED)
        checkpoint.pair("q2", unknown=[0.45, 0.55], control=[0.995, 0.005])

        report = evaluate.abstention_report(checkpoint, checkpoint.records)

        self.assertEqual(report["n_pairs"], 2)
        self.assertGreater(report["all"]["mean_drop"], 0.0)
        self.assertEqual(report["all"]["share_lower"], 1.0)
        self.assertEqual(report["all"]["unknown_at_0_9"], 0.0)
        self.assertEqual(report["all"]["control_at_0_9"], 1.0)
        self.assertDrop(report["all"], metrics.paired_confidence_drop(
            [FLAT, [0.45, 0.55]], [PEAKED, [0.995, 0.005]]))

    def test_a_model_that_sharpens_without_the_evidence_reports_a_negative_drop(self):
        """The same pair read the other way: a model confident where the evidence
        is gone is guessing, and the drop has to come out negative rather than be
        reported as a confident model that noticed."""
        checkpoint = _StubCheckpoint().pair("q1", unknown=PEAKED, control=FLAT)

        all_pairs = evaluate.abstention_report(checkpoint, checkpoint.records)["all"]

        # Flat, 0.0, minus peaked, 0.919207.
        self.assertAlmostEqual(all_pairs["mean_drop"], -0.919207, places=6)
        self.assertEqual(all_pairs["share_lower"], 0.0)
        self.assertEqual(all_pairs["unknown_at_0_9"], 1.0)
        self.assertEqual(all_pairs["control_at_0_9"], 0.0)


class PairingTests(_DropAssertions):
    def _suite(self):
        """Three pairs whose drops point different ways, so a mispaired record lands
        somewhere else rather than cancelling: +0.919207, -0.919207, +0.029049."""
        return (_StubCheckpoint()
                .pair("q1", unknown=FLAT, control=PEAKED)
                .pair("q2", unknown=PEAKED, control=FLAT)
                .pair("q3", unknown=FLAT, control=[0.6, 0.4]))

    def test_each_unanswerable_record_is_measured_against_its_own_control(self):
        """Two pairs cancel, so the mean is a fact about all three pairings: taking
        any record's partner from the wrong pair moves the pooled number."""
        checkpoint = self._suite()

        report = evaluate.abstention_report(checkpoint, checkpoint.records)

        self.assertEqual(report["n_pairs"], 3)
        # (0.919207 - 0.919207 + 0.029049) / 3
        self.assertAlmostEqual(report["all"]["mean_drop"], 0.009683, places=6)
        self.assertDrop(report["all"], metrics.paired_confidence_drop(
            [FLAT, PEAKED, FLAT], [PEAKED, FLAT, [0.6, 0.4]]))

    def test_record_order_does_not_move_the_pairs(self):
        """The suite names its pairs, so a reversed, rotated or grouped suite must
        report the tidy suite's numbers; an adjacency pairing cannot."""
        checkpoint = self._suite()
        tidy = evaluate.abstention_report(checkpoint, checkpoint.records)
        unanswerable = [r for r in checkpoint.records if r["_meta"].get("control_id")]
        intact = [r for r in checkpoint.records if not r["_meta"].get("control_id")]

        for name, order in (("reversed", list(reversed(checkpoint.records))),
                            ("halves grouped", unanswerable + intact),
                            ("rotated", checkpoint.records[1:] + checkpoint.records[:1])):
            with self.subTest(order=name):
                report = evaluate.abstention_report(checkpoint, order)
                self.assertEqual(report["n_pairs"], tidy["n_pairs"])
                self.assertDrop(report["all"], tidy["all"])


class SuiteRefusalTests(unittest.TestCase):
    """A suite the pairing cannot be read from has to stop the run.

    Each of these would otherwise still produce a number, and the caller has no
    way to tell from the report which subset it was measured over.
    """

    def _checkpoint(self):
        return _StubCheckpoint().pair("q1", unknown=FLAT, control=PEAKED)

    def test_a_control_that_names_no_record_is_refused_rather_than_skipped(self):
        """A perfectly good pair sits next to the dangling one, which is exactly the
        case a skip hides: the report would be a drop over the pairs that resolved."""
        checkpoint = self._checkpoint().add(
            _record("q2:unknown", control_id="q2"), _noul_probs(*FLAT))

        with self.assertRaises(SystemExit):
            evaluate.abstention_report(checkpoint, checkpoint.records)

    def test_a_pair_with_a_distribution_missing_on_one_side_is_refused(self):
        """A checkpoint that cannot build one record's item serves nothing for it,
        which leaves a pair with no distribution to compare. The scorable pair
        beside it must not carry the run: dropping the unscorable pair would
        report a drop over a subset smaller than the suite without saying so, and
        the suite the model could not answer would read as a smaller measurement
        rather than a broken one. Either side missing is the same failure."""
        for name, unknown_served, control_served in (
                ("the unanswerable half served nothing", False, True),
                ("the intact half served nothing", True, False)):
            with self.subTest(suite=name):
                checkpoint = self._checkpoint()  # one pair that is perfectly scorable
                checkpoint.add(_record("unanswered", control_id="intact"),
                               _noul_probs(*FLAT) if unknown_served else None)
                checkpoint.add(_record("intact"),
                               _noul_probs(*PEAKED) if control_served else None)

                with self.assertRaises(SystemExit) as refused:
                    evaluate.abstention_report(checkpoint, checkpoint.records)

                # And the refusal names both records of the pair it could not score.
                message = str(refused.exception)
                self.assertIn("unanswered", message)
                self.assertIn("intact", message)

    def test_a_record_without_an_id_is_refused(self):
        """Nothing can name it as a control, so there is no pairing to read."""
        for name, record in (("no _meta at all", {"questions": {"q": _noul_question()}}),
                             ("_meta with no id", {"_meta": {"family": "alpha"},
                                                   "questions": {"q": _noul_question()}})):
            with self.subTest(record=name):
                checkpoint = self._checkpoint().add(record)
                with self.assertRaises(SystemExit):
                    evaluate.abstention_report(checkpoint, checkpoint.records)

    def test_a_record_with_anything_but_one_question_is_refused(self):
        """Averaging several questions into one confidence would make the drop a
        number about the suite's shape instead of about the evidence, so the count
        has to be checked as a property of the suite and not only of its odd
        record: a suite where every record carries the wrong number of questions
        is just as unscoreable as one where a single record does."""
        two = {"q1": _noul_question(), "q2": _noul_question()}
        good = self._checkpoint().records
        cases = {
            "one bad record beside good pairs (two questions)":
                good + [_record("q3", questions=two)],
            "one bad record beside good pairs (no questions)":
                good + [_record("q3", questions={})],
            "every record with two questions":
                [_record("q3:unknown", questions=two, control_id="q3"), _record("q3", questions=two)],
            "every record with no questions":
                [_record("q3:unknown", questions={}, control_id="q3"), _record("q3", questions={})],
        }
        for name, records in cases.items():
            with self.subTest(suite=name):
                checkpoint = _StubCheckpoint()
                for record in records:
                    checkpoint.add(record)
                with self.assertRaises(SystemExit):
                    evaluate.abstention_report(checkpoint, records)


class FamilySplitTests(_DropAssertions):
    def _suite(self):
        """Two pairs on a family the mixture covers, pointing opposite ways, and one
        on a family it does not: +0.919207, -0.919207, +0.919207."""
        return (_StubCheckpoint()
                .pair("memorised:up", unknown=FLAT, control=PEAKED, family="alpha")
                .pair("memorised:down", unknown=PEAKED, control=FLAT, family="alpha")
                .pair("general:up", unknown=FLAT, control=PEAKED, family="gamma"))

    def test_the_split_separates_the_families_and_the_pairs_add_up(self):
        """The covered and uncovered halves are measured apart, over every pair:
        pooling them averages a real drop with a memorised one and publishes the
        result as the model's general behaviour."""
        checkpoint = self._suite()

        report = evaluate.abstention_report(checkpoint, checkpoint.records,
                                            mix_records=[{"_meta": {"family": "alpha"}}])

        inside, outside = report["familiesInMixture"], report["familiesOutsideMixture"]
        self.assertEqual(inside["n_pairs"], 2)
        self.assertEqual(outside["n_pairs"], 1)
        self.assertEqual(inside["n_pairs"] + outside["n_pairs"], report["n_pairs"])
        # The covered family's two pairs cancel exactly; the uncovered pair only falls.
        self.assertAlmostEqual(inside["mean_drop"], 0.0, places=12)
        self.assertAlmostEqual(outside["mean_drop"], 0.919207, places=6)
        # And the pooled mean is neither half, which is why the split exists.
        self.assertAlmostEqual(report["all"]["mean_drop"], 0.306402, places=6)
        self.assertDrop(report["all"], metrics.paired_confidence_drop(
            [FLAT, PEAKED, FLAT], [PEAKED, FLAT, PEAKED]))

    def test_without_the_mixture_the_split_is_absent_and_says_so(self):
        """`None`, not an empty split: with no mixture the covered/uncovered
        question was never asked, and `{}` would read as no family being in it."""
        checkpoint = self._suite()

        report = evaluate.abstention_report(checkpoint, checkpoint.records)

        self.assertIsNone(report["familiesInMixture"])
        self.assertIsNone(report["familiesOutsideMixture"])
        self.assertTrue(report["note"])

    def test_an_empty_mixture_is_an_empty_split_not_an_absent_one(self):
        """An empty mixture answers the question — nothing is covered — so it must
        not be confused with having no mixture to ask about."""
        checkpoint = self._suite()

        report = evaluate.abstention_report(checkpoint, checkpoint.records, mix_records=[])

        self.assertEqual(report["familiesInMixture"]["n_pairs"], 0)
        self.assertEqual(report["familiesOutsideMixture"]["n_pairs"], 3)


class ServedDistributionTests(unittest.TestCase):
    """What a served answer becomes, in option order.

    Order is the contract, not the set of numbers: the distribution is read by
    position, so `sorted()` or `dict.values()` in place of the criteria order,
    the false-then-true order, or the level order pairs every option with
    another option's probability. A second implementation would also let the
    scored report and the abstention report publish numbers about two different
    models, so both are read here.
    """

    CASES = (
        ("a choice reads its options in the order the caller wrote the criteria",
         {"type": "choice", "criteria": {"b": "second", "a": "first", "c": "third"}},
         {"b": 0.2, "a": 0.7, "c": 0.1},
         [0.2, 0.7, 0.1]),
        ("a yes/no reads false then true, whatever order they were served in",
         _noul_question(),
         {"true": 0.25, "false": 0.75},
         [0.75, 0.25]),
        ("a score reads its levels by position",
         {"type": "score", "criteria": ["low", "mid", "high"]},
         {"2": 0.1, "0": 0.6, "1": 0.3},
         [0.6, 0.3, 0.1]),
    )

    def test_probs_for_returns_the_distribution_in_option_order(self):
        for name, question, probabilities, expected in self.CASES:
            with self.subTest(question=name):
                answer = {"probabilities": probabilities, "confidence": 0.5}
                self.assertEqual(evaluate._probs_for(question, answer), expected)

    def test_per_question_rows_reads_each_answer_in_option_order_too(self):
        """The scored report flattens the same served answers, and it has to land
        on the same ordered list `_probs_for` gives the abstention report."""
        checkpoint = _StubCheckpoint()
        records = []
        for i, (_, question, probabilities, _) in enumerate(self.CASES):
            records.append(_record("r%d" % i, question=question))
            checkpoint.add(records[-1], probabilities)

        with mock.patch.dict(sys.modules, {"flint.inference": _inference_stub()}):
            rows = evaluate.per_question_rows(checkpoint, records)

        self.assertEqual([row["probs"] for row in rows], [case[3] for case in self.CASES])


if __name__ == "__main__":
    unittest.main()
