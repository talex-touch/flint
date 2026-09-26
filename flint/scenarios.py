"""Turning a labelled corpus into decision records.

Flint does not train on a task, it trains on *decisions about a task*. The
difference matters: a classifier is handed a text and asked what it is, with the
label space fixed at training time; a decision model is handed a state, a
question and a closed set of options, and has to say which option the evidence
picks — and how sure it is. Same corpus, different record.

Records are written in the wire format a client sends (`type` / `instructions` /
`criteria` / `label`), not in the engine's internal shape (`t` / `ins` / `crit`).
`flint.train` owns the conversion, so an evaluation suite can be posted to a
running server verbatim — which is the only way to know the number you publish
is the number the product would get.

Three constructions, one per question type:

- a single-label corpus becomes a **choice** over the true label plus sampled
  distractors;
- a binary corpus becomes a **noul** (yes/no) question;
- an ordinal corpus becomes a **score** over its levels.

Two properties are enforced here rather than left to the trainer, because both
are silent when broken:

**Option order carries no information.** Every choice record's options are
shuffled with the builder's seed, and the label is recorded as the *key that
landed in the answer position*, so a model that learns "the answer is usually
first" has learned nothing usable. `flint.evaluate` shuffles again with a
different seed and reports the difference: a model whose score moves when the
options move has learned the layout, not the question.

**Distractor width is recorded, not implied.** A 4-way choice and a 40-way
choice are different tasks and nothing here pools them into one number; the
width travels on the record so a report can split by it.
"""

from __future__ import annotations

import random
from typing import Iterable, Sequence

__all__ = ["choice_records", "noul_records", "score_records"]

#: The key used for "none of these" in a choice question. The engine treats it
#: as an ordinary option, which is the point: declining is a decision the model
#: has to make on the evidence, not an error path bolted on afterwards.
NONE_KEY = "none"


def _criteria_map(descriptions: Sequence[str]) -> dict[str, str]:
    """``["a", "b"]`` -> ``{"c1": "a", "c2": "b"}`` in the engine's key style."""
    return {"c%d" % (i + 1): str(d) for i, d in enumerate(descriptions)}


def choice_records(
    items: Iterable[tuple[str, int]],
    labels: Sequence[str],
    width: int = 4,
    seed: int = 0,
    instructions: str = "which option does the state describe?",
    with_none: bool = False,
    label_usage_cap: int | None = None,
) -> list[dict]:
    """One single-label example becomes one ``choice`` question.

    ``items`` are ``(text, label_index)`` pairs and ``labels`` is the label
    space. Each record draws ``width - 1`` distractors from the labels other
    than the true one, then shuffles the options and stores the answer as the
    key that ended up holding the true label.

    ``label_usage_cap`` bounds how often any one label is used as a distractor
    across the call. Without it a 40-class corpus presents its rarest labels to
    nearly every record, and the model learns the common ones by elimination
    rather than by reading.

    ``with_none`` appends an explicit "none of these" option *without* ever
    making it the answer: it is there so the model sees the option and has to
    place probability on it, which is what makes the option usable at inference.
    A record whose answer is ``none`` has to be built from a genuine negative
    example, and inventing one here would be a lie about the corpus.
    """
    items = list(items)
    if not labels:
        raise ValueError("labels must not be empty")
    n_labels = len(labels)
    if n_labels < 2:
        raise ValueError("a choice question needs at least two labels")
    if not 2 <= width <= n_labels:
        raise ValueError("width must be in [2, %d], got %r" % (n_labels, width))

    rng = random.Random(seed)
    usage: dict[int, int] = {}
    records: list[dict] = []

    for text, label_index in items:
        if not 0 <= label_index < n_labels:
            raise ValueError("label index %r is outside the label space" % (label_index,))
        pool = [i for i in range(n_labels) if i != label_index]
        if label_usage_cap is not None:
            rng.shuffle(pool)
            pool.sort(key=lambda i: usage.get(i, 0))
        else:
            rng.shuffle(pool)
        chosen = pool[: width - 1]
        for i in chosen:
            usage[i] = usage.get(i, 0) + 1

        descriptions = [labels[label_index]] + [labels[i] for i in chosen]
        if with_none:
            descriptions.append("none of these options applies")

        order = list(range(len(descriptions)))
        rng.shuffle(order)
        shuffled = [descriptions[i] for i in order]
        criteria = _criteria_map(shuffled)
        keys = list(criteria)

        # The true label started at index 0 of `descriptions`; find where it
        # moved to, then read that position's key.
        answer_key = keys[order.index(0)]

        records.append({
            "state": text,
            "questions": {
                "pick": {
                    "type": "choice",
                    "instructions": instructions,
                    "criteria": criteria,
                    "label": answer_key,
                }
            },
            "_meta": {
                "kind": "choice",
                "width": len(shuffled),
                "true_label": int(label_index),
                "has_none": bool(with_none),
            },
        })
    return records


def noul_records(
    items: Iterable[tuple[str, bool]],
    yes_description: str = "yes, the statement holds",
    no_description: str = "no, the statement does not hold",
) -> list[dict]:
    """A binary example becomes one yes/no question.

    ``noul`` is not a two-way choice under another name. The engine reports a
    single probability for "yes" rather than a distribution, so the label is a
    boolean and the two criterion descriptions are what the model actually
    reads. There is no option order to shuffle.
    """
    records = []
    for text, is_true in items:
        records.append({
            "state": text,
            "questions": {
                "holds": {
                    "type": "noul",
                    "instructions": "does the state support this?",
                    "criteria": {"false": no_description, "true": yes_description},
                    "label": bool(is_true),
                }
            },
            "_meta": {"kind": "noul", "positive": bool(is_true)},
        })
    return records


def score_records(
    items: Iterable[tuple[str, int]],
    levels: Sequence[int],
    instructions: str = "which level describes the state?",
) -> list[dict]:
    """An ordinal example becomes one ``score`` question over ``levels``.

    ``levels`` are the scale's values in ascending order. The stored label is an
    *index* into ``levels``, not the level value, so a scale that does not start
    at zero or is not contiguous behaves the same as one that is. The scale has
    a fixed order by definition, so there is no shuffling here.
    """
    levels = list(levels)
    if len(levels) < 2:
        raise ValueError("a score question needs at least two levels")
    if any(b <= a for a, b in zip(levels, levels[1:])):
        raise ValueError("levels must be strictly ascending, got %r" % (levels,))
    index = {v: i for i, v in enumerate(levels)}

    records = []
    for text, level in items:
        if level not in index:
            raise ValueError("level %r is not in the scale %r" % (level, levels))
        records.append({
            "state": text,
            "questions": {
                "rate": {
                    "type": "score",
                    "instructions": instructions,
                    "criteria": ["level %d" % (i + 1) for i in range(len(levels))],
                    "label": index[level],
                }
            },
            "_meta": {"kind": "score", "level": int(level), "levels": list(levels)},
        })
    return records
