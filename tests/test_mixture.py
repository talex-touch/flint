"""Tests for the mixture builder and its audits.

Two properties are defended here, both of which lie quietly when broken: a
source weight smuggled into the file as repeated rows, and a leak that an
input-text overlap check cannot see because only the metadata differs.
"""

import unittest

from flint import mixture


def _record(state, label="c1", criteria=None):
    """One decision record in the client wire format."""
    return {
        "state": state,
        "questions": {
            "pick": {
                "type": "choice",
                "instructions": "which option does the state describe?",
                "criteria": {"c1": "alpha", "c2": "beta"} if criteria is None else criteria,
                "label": label,
            }
        },
    }


def _sourced(record, source):
    out = dict(record)
    out["_meta"] = {"source": source}
    return out


class RecordFingerprintTests(unittest.TestCase):
    def test_fingerprint_covers_content_but_not_bookkeeping(self):
        """Same state and questions => same fingerprint, whatever the metadata says.

        This is the premise of every dedupe and every leak check below: if a
        `_meta.source` or a label a builder attached could change the
        fingerprint, a duplicate would look distinct and a leak would look
        clean. The obverse has to hold too — if content did not change the
        fingerprint, nothing would ever be reported as duplicated.
        """
        base = _record("the account was closed after the dispute")

        variants = {
            "different source": _sourced(base, "train"),
            "extra bookkeeping": dict(base, _meta={"source": "train", "shard": 7, "rows": 3}),
            "different top-level label": dict(base, label=9),
        }
        expected = mixture.record_fingerprint(base)
        for name, record in variants.items():
            with self.subTest(variant=name):
                self.assertEqual(mixture.record_fingerprint(record), expected)

        different = {
            "different state": dict(base, state="a different state"),
            "different option text": _record("the account was closed after the dispute",
                                             criteria={"c1": "alpha", "c2": "omega"}),
            "different answer": _record("the account was closed after the dispute", label="c2"),
        }
        for name, record in different.items():
            with self.subTest(variant=name):
                self.assertNotEqual(mixture.record_fingerprint(record), expected)


class MixtureBuildTests(unittest.TestCase):
    def test_weight_does_not_repeat_rows(self):
        """A weight is a sampling weight, so it must not lengthen the file.

        `weight=4` on three rows means "sample these three four times as often",
        which the trainer's sampler owns. Emitting twelve rows instead would
        make the mixture's recorded source shares a lie — the failure this class
        exists to prevent.
        """
        records = [_record("state %d" % i) for i in range(3)]
        rows, _ = mixture.Mixture(seed=0).add("weighted", records, weight=4.0).build()

        self.assertEqual(len(rows), 3)
        self.assertEqual(len({mixture.record_fingerprint(r) for r in rows}), 3)

    def test_two_sources_emitting_the_same_record_are_rejected(self):
        """Two sources cannot both own the same record and keep their shares."""
        rows = [_record("shared state"), _record("only in one source")]
        builder = mixture.Mixture(seed=0).add("first", rows).add("second", [_record("shared state")])

        with self.assertRaises(ValueError):
            builder.build()

    def test_cap_truncates_after_shuffling_and_reproduces_by_seed(self):
        """`cap` keeps a spread of the source, and the seed makes it repeatable."""
        source = [_record("state %d" % i) for i in range(10)]

        def kept(seed):
            rows, _ = mixture.Mixture(seed=seed).add("s", source, cap=3).build()
            return sorted(r["state"] for r in rows)

        self.assertEqual(len(kept(0)), 3)
        self.assertEqual(kept(0), kept(0))
        # A cap taken from the head of the file instead of from a shuffle would
        # give the same three rows for every seed.
        self.assertNotEqual(kept(0), kept(1))


class AuditTests(unittest.TestCase):
    def test_audit_finds_duplicates_across_sources(self):
        """The same record under two sources is one duplicate, attributed to both."""
        rows = [
            _sourced(_record("state a"), "s1"),
            _sourced(_record("state a"), "s2"),
            _sourced(_record("state b"), "s1"),
        ]

        report = mixture.audit(rows)

        self.assertEqual(report["rows"], 3)
        self.assertEqual(report["uniqueRows"], 2)
        self.assertEqual(report["duplicateRows"], 1)
        self.assertEqual(sorted(report["duplicateSources"]), ["s1", "s2"])


class FindOverlapTests(unittest.TestCase):
    def test_overlap_is_content_not_input_text(self):
        """A leak is a record the model has seen, not merely a sentence it has seen.

        The eval copy of "state a" carries different provenance and byte-identical
        content: an overlap check that compares input text, or that trusts the
        `source` field, finds nothing here. The last eval row repeats the state
        with a different question, which is a different record — a check keyed on
        the input text alone would count that one too.
        """
        train = [_sourced(_record("state a"), "train"), _sourced(_record("state b"), "train")]
        eval_rows = [
            _sourced(_record("state a"), "eval"),
            _sourced(_record("state c"), "eval"),
            _sourced(_record("state a", criteria={"c1": "alpha", "c2": "omega"}), "eval"),
        ]

        self.assertEqual(mixture.find_overlap(train, eval_rows), 1)


if __name__ == "__main__":
    unittest.main()
