# Evaluation suites

This directory is where suites live. None are committed, and that is a
deliberate choice rather than an empty gesture.

A suite is only meaningful if you can vouch for two things, and neither travels
with a file:

1. **Its records never appeared in training**, compared on `(state, questions)`
   and ignoring `_meta` and labels.
2. **It was generated with the held-out families excluded**, not merely with
   different records.

Point 2 is the one that catches people. A suite produced by the same generator as
a training source passes an exact-input overlap check while sharing the *task*,
and a model that memorised the task scores far above its ability.
`docs/evaluation.md` §5 describes a case where a family was declared excluded in
a metadata field, was not actually excluded in the data, and produced a
twenty-point apparent gain that was pure recall.

So: build a suite with `flint.scenarios`, keep the generator's exclusion
instruction in the mixture spec next to the training sources, and let
`flint-evaluate --mix` refuse the run when the two disagree.

## Format

One JSON object per line, a decision record — the same shape a client sends
(`docs/api.md`). A suite record may carry a `label`; without one it can still be
scored for calibration and for abstention, but not for accuracy.

```json
{"state": "Order 4471 shipped on the 3rd. The customer asked to return it on the 9th.",
 "questions": {"pick": {"type": "choice",
                        "instructions": "which option does the state describe?",
                        "criteria": {"c1": "inside the return window",
                                     "c2": "outside the return window"},
                        "label": "c1"}},
 "_meta": {"suite": "returns", "family": "window", "generator": "returns-v1"}}
```

`_meta` is free-form and is never scored; `flint.mixture.Mixture` overwrites
`_meta.source` on rows it emits, so anything a suite needs to carry about itself
belongs under other keys.

Build one:

```python
from flint.scenarios import choice_records
from flint.mixture import write_jsonl

records = choice_records(
    items=[(text, label_index), ...],   # your held-out corpus
    labels=["inside the return window", "outside the return window", ...],
    width=4, seed=7,
)
write_jsonl("evals/returns.jsonl", records)
```

## Two things to record with every result

- **The question count.** On a hundred questions one wrong answer moves a 5%
  error budget by more than the difference between two checkpoints.
- **The temperature.** It is part of the checkpoint, not a serving detail; two
  servers on the same weights at different temperatures are different products,
  and a number that does not say which one it was measured at cannot be
  reproduced.
