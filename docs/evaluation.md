# Evaluation

A decision model is not evaluated like a classifier, and evaluating it like one
produces numbers that are wrong in a specific direction: too good. It is wrong in
three ways, and each of them cost a real result before it was understood.

## 1. Accuracy is not the number the product spends

Flint is allowed to decline. The product sets a threshold on confidence and
releases only the answers above it. So the quantity that matters is not "what
share is correct" but:

> **At a fixed error budget, what share of decisions can be released?**

Take the decisions in confidence order and release as many as possible while the
released set stays under the budget. That is `metrics.coverage_at_error`, and it
is the number two checkpoints differ on most. Two models with identical accuracy
can differ by fifteen points of coverage, and the one with lower accuracy can be
the one that covers more.

Report it at several budgets — 5%, 10%, 15% — because a model can lead at one and
trail at another, and a product's acceptable error rate is a product decision,
not a property of the model.

## 2. Comparing two models at one threshold measures the threshold

This is the mistake that made an early comparison wrong. Two checkpoints were
compared by counting how many answers each released above a fixed confidence
threshold, and how many of those were wrong. The conclusion was that one was
clearly worse.

It was not. The two checkpoints shipped different temperatures, and a temperature
is a monotone rescaling — it does not change which option wins, so the accuracy
was identical, but it moves the released set. The "worse" checkpoint was being
measured at a sharper operating point and releasing more answers *including more
of the wrong ones*. The capability gap was mostly an operating-point gap.

The fix is to compare across the whole range, in both directions:

- **risk at matched coverage** — fix how much must be released, compare error;
- **coverage at matched risk** — fix the error budget, compare how much is released.

`metrics.selective_risk_curve` produces the first. After that change the same two
checkpoints turned out to be close to equal on one axis and clearly ordered on
another — a result that the single-threshold comparison could not have produced.

## 3. The temperature is an operating point, not a calibration constant

Fitting a temperature by minimising negative log-likelihood is the standard move,
and it optimises the wrong thing here.

NLL asks for confidence that matches accuracy **in aggregate**. The operating
point asks for a confidence **ordering** that puts the wrong rows at the bottom.
Those are different objectives, and they disagree systematically: the
NLL-optimal temperature sits near the *worst* end of the coverage curve.

Measured on one checkpoint at a 10% error budget, with the argmax unchanged and
therefore the accuracy identical:

| temperature | chosen by | released | wrong among released |
| --- | --- | --- | --- |
| ~0.93 | NLL | 90% | 30 |
| ~1.74 | coverage at the budget | 73% | 17 |

Same model, same accuracy, roughly half the wrong answers reaching a user, at the
cost of releasing seventeen points fewer decisions. `metrics.select_temperature`
fits the second one, and it is the right default for anything a user sees.

The mechanism is the one described in `docs/api.md`: confidence is normalised
entropy, and a temperature rescales each row's entropy by an amount that depends
on how peaked that row already was. Rows therefore move against each other, and
the accepted set changes. If confidence were the top probability, a temperature
would be a pure threshold shift and this section would be unnecessary — which is
why the two design decisions are documented together.

Fit it on one set and **confirm it on another**. A temperature chosen and reported
on the same data is a number that cannot fail, and therefore one that tells you
nothing.

## 4. Check that the answer is about the question, not the layout

A choice question's options are permuted during training, so position should carry
no information. Whether that worked is a measurement, not an assumption: score
the suite twice with the options laid out differently, map each answer back to the
option's *identity* rather than its position, and count how often the identity
changed.

A model whose answer flips when the layout changes has learned the layout. No
aggregate score reveals this — accuracy can be identical across both layouts while
a third of the answers are about different options.

## 5. Never evaluate on anything nearby

The suite must not overlap the training mixture, and the check must run on the
same fingerprint the mixture is built on — `state` plus `questions`, ignoring
`_meta` and labels. `flint.evaluate --mix` refuses to run otherwise.

Checking only whether the input text appears in training is not enough. A suite
can be built by the same generator as a training source and pass an exact-input
overlap check while sharing the *task*, and a model that has memorised the task
scores far above its real ability.

There is a sharper version of this that a metadata field will not catch. One
mixture declared a source excluded from training, in a `source` field, on a
family of generated policy questions — and the evaluation set for that family was
byte-identical to rows the model had trained on, under a different label. The
declared exclusion was true of the *intent* and false of the *data*. The check
that found it compared content fingerprints and ignored the claim:

```
for each eval record: is its (state, questions) fingerprint in the training set?
```

On that suite, a checkpoint trained with the family excluded scored 45% and the
one that had memorised it scored 64% — a twenty-point difference that was pure
recall, and that would have been reported as a generalisation gain.

Practical rule: a held-out set is only held out if the generator was told to hold
it out and the data agrees. Verify both.

## 6. Delete the evidence and see whether confidence drops

Every measurement above needs a label. This one cannot have one, and that is the
point: take a question whose answer turns on one sentence, delete that sentence,
and the item becomes unanswerable. There is no correct option left to score, so
scoring it would measure the construction of the suite rather than the model.

What is measured instead is the **difference** between two records that are
identical except for that sentence — one unanswerable, one intact — and the
question asked of the pair is whether confidence fell.

```
flint-evaluate --checkpoint <dir> --suite <paired.jsonl> --mix <mix.jsonl> --abstain
```

The suite declares the pairing in `_meta`: an unanswerable record carries
`control_id`, naming its intact partner's `meta.id`, and every record carries
`family`. Both are required — a partial pairing reports a drop over a subset that
is not the one you think, and `abstain` refuses rather than skipping.

Confidence here is the gate's confidence, normalised entropy, not the top
probability. That distinction is not cosmetic: on the same three checkpoints the
two bases agree on the *direction* and disagree on the magnitude, and the share
of unanswerable items still scoring above 0.9 is 0.11 on entropy confidence and
0.18 on top probability for the same model. Only one of those is the number a
product would act on, and it is the one the gate reads.

Measured on the reference line, 110 pairs, `--mix` pointing at the mixture each
checkpoint was trained from:

| checkpoint | mean drop | pairs where it fell | families in mixture (n=90) | families outside (n=20) |
| --- | ---: | ---: | ---: | ---: |
| v5 | −0.0423 | 35.5% | −0.0557 | +0.0180 |
| v6 | +0.0306 | 50.9% | −0.0126 | +0.2249 |
| **v7** | **+0.0550** | **71.8%** | **+0.0784** | **−0.0505** |

**Read the last two columns together, and then disbelieve the last one.** The
drop is clear on the families the mixture contains and absent — the sign even
reverses — on the families it does not. With twenty pairs in that column, the
outside number is noise rather than a measured absence, so the honest reading is
"the effect was not demonstrated outside the mixture", not "the effect is
negative outside it". Either way it is not a general property of the model: it is
a property of the model on the task distribution, and the columns are the reason
this section exists instead of a single number.

Two things this measurement is not. It is not an accuracy result — an
unanswerable item has no accuracy. And it is not a substitute for the coverage
curve: a model can abstain well and still be wrong on everything it releases.

## The report

`python -m flint.evaluate --checkpoint <dir> --suite <jsonl> [--mix <jsonl>]`
writes:

| field | meaning |
| --- | --- |
| `accuracy` | share of questions where the top option is the label |
| `ece`, `eceEntropyConfidence` | calibration gap on top probability and on the gate's confidence |
| `coverageAtError` | coverage, threshold, and what a deployed threshold actually accepts, at 5/10/15% |
| `riskAtCoverage` | error rate at 60/70/80/90/95% coverage |
| `byQtype`, `byWidth` | the same, split by question type and option count |
| `orderSensitivity` | flip rate and accuracy under a permuted option layout |

Every split is reported because a single pooled number over questions with
different option counts is a number nobody can act on: confidence is normalised
by the option count, so the pooled figure depends on the width mix of the suite
rather than on the model. `byWidth` is where a regression shows up.

Two cautions about reading the report:

- **Check `scoreable` against `questions`.** A record whose label is not among its
  own options cannot be scored; if that number is not zero, the suite has a
  format bug and the accuracy is over a subset.
- **A coverage curve over a small suite is noise.** On a hundred questions, one
  wrong answer moves the 5% budget by more than the difference between two
  checkpoints. Quote no coverage number without the question count next to it.
