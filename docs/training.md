# Training a Flint checkpoint

Flint trains a decision head on top of a released encoder. The engine is not
ours (`docs/method.md`), the recipe is, and so is the responsibility for what the
resulting weights may be used for.

Run:

```bash
flint-build --spec <mixture spec> --out data/mix.jsonl      # build and audit a mixture
flint-train --mix data/mix.jsonl --dev data/dev.jsonl --out runs/flint-v1 --device mps
flint-evaluate --checkpoint runs/flint-v1 --suite data/dev.jsonl --mix data/mix.jsonl
```

`python -m flint.smoke` runs the whole path against a tiny local encoder and a
synthetic mixture, in seconds and with no network. Run it after any change to the
adapter, the trainer or the checkpoint layout — it is a test that the plumbing is
intact, not that the model is any good.

## 1. What a run is

| step | module | what it produces |
| --- | --- | --- |
| records | `flint.scenarios` | a labelled corpus becomes decision records |
| mixture | `flint.mixture` | named sources, deduplicated, with provenance, audited |
| items | `flint.engine.to_item` | wire records become engine sequences and targets |
| training | `flint.train` | the head, plus a fitted temperature |
| report | `flint.evaluate` | coverage, calibration, order sensitivity |

## 2. Corpora, and what their terms actually say

A checkpoint is publishable only when **every corpus that produced its targets**
permits redistribution of derived works. That is a stricter condition than "the
corpus is free to download", and it excludes most of the obvious candidates.

Verified against the Hugging Face API on **2026-09-25**. Terms change; re-read
them before a run and record what you read.

| corpus | license as published | verdict |
| --- | --- | --- |
| `legacy-datasets/banking77` | **CC BY 4.0** | Attribution only. Usable, with the attribution recorded in the dataset manifest. |
| `google/boolq` | **CC BY-SA 3.0** | Share-alike. Whether derived weights inherit the obligation is a legal question, not a technical one. Treat as research-only until someone who can answer it does. |
| `nyu-mll/multi_nli` | **mixed**: CC BY 3.0, CC BY-SA 3.0, MIT, other | The share-alike component governs the whole. Same verdict as above. |
| `fancyzhx/ag_news` | **unknown** | Not a licence. Exclude until the publisher declares one. |
| `SetFit/sst5` | **none declared** | No declaration is not permission. Exclude. |
| `Yelp/yelp_review_full` | **other** | "Other" means read the actual terms, which usually means the Yelp dataset agreement rather than a Creative Commons licence. Do not assume. |

Three rules follow, and they are the ones that matter:

1. **An undeclared or "unknown" licence is a refusal, not a formality.** The
   absence of a licence is the default of "all rights reserved".
2. **A share-alike corpus contaminates a checkpoint.** This is not about the
   corpus's copyright in its text; it is about what obligations attach to a model
   trained on it. If nobody can answer that in writing, the checkpoint stays
   local.
3. **Record the revision, not just the name.** Every dataset above is a moving
   target on the Hub; `fancyzhx/ag_news` at one commit and at another are
   different data. A mixture built from a name alone is not reproducible. Pin the
   commit sha in the mixture spec and write it into `train_meta.json`.

Anything derived from a corpus you cannot defend produces a checkpoint that is
**not published here**. That is the same rule this organisation applies to
weights reverse-engineered from a vendor's product: ideas are reusable, weights
are not. The recipe below is published either way, so a clean corpus can be
substituted and the run repeated.

## 3. Building the mixture

Use `flint.mixture.Mixture`, not a concatenation of files. It enforces the three
things that a hand-built mixture gets wrong silently:

**Each record appears exactly once.** Source weights are *sampling* weights for the
trainer's sampler, recorded in the report — not repeated rows. Repeating rows to
express a weight is the tempting shortcut and it destroys auditability: the file
no longer says what the recipe said, and the duplicate count grows with the
weight, so "this source matters twice as much" and "someone pasted the same row
twice" become the same observation. `Mixture.build()` raises if two sources
produce the same record, because then the source shares cannot both be true.

**Every row carries its provenance.** `_meta.source` is written on emit. A
mixture without it cannot be audited afterwards — not the duplicate count, not
which source a regression came from, not whether a held-out set was really held
out. This is not hypothetical: a mixture built from a script that assigned
weights by literal repetition produced a training set that was 15% exact
duplicates, inherited unchanged by every subsequent version, and the file itself
carried no way to see it.

**Caps are applied after a seeded shuffle.** A corpus ordered by class, truncated
at 300 rows, contributes the first class only. `Mixture.add(..., cap=n)` shuffles
with the mixture seed first, so a cap is a sample rather than a prefix — and is
reproducible.

Then check the result:

```python
from flint.mixture import audit, load_jsonl
audit(load_jsonl("data/mix.jsonl"))
```

## 4. The target

Training minimises the log score against a target distribution over the options.
That loss is a *proper scoring rule*: its expectation is minimised by reporting
the true distribution, which is the property that makes the resulting confidence
worth gating on. A cross-entropy against hard one-hot labels is the same rule with
a target that claims certainty.

**Label smoothing is not cosmetic.** Against hard targets the model is being told
to be certain, and certainty on a genuinely ambiguous item is exactly the
over-confidence that makes a confidence gate useless. `--smooth 0.03` mixes a
uniform component into every target.

**Soft targets need smoothing more, not less.** A record may carry `probs`, a
distribution over the options, which is what distillation from a stronger teacher
produces. The trap is that such teachers are often *saturated*: a large fraction
of items come back with probability 1.0 on the winner and nothing anywhere else.
Copied verbatim, those targets collapse to one-hot and the distillation transfers
nothing that a hard label would not have — while looking like a much richer
signal in the mixture. Smoothing is what keeps them from collapsing. If you use
soft targets, measure the fraction that are effectively one-hot before believing
you have more information than a label.

## 5. Permuting options

`--no-shuffle-options` disables it; it is on by default for choice questions and
must stay off for the other two types. A `score` scale is ascending by definition
and a `noul` question's false/true slots are semantic — permuting either is
relabelling the target, not augmenting it. A `none` option, when present, is
pinned last so that a sentinel does not wander through the list and teach a
position cue.

Whether the permutation worked is measured, not assumed: `flint.evaluate` reports
a flip rate by re-scoring with a different layout (see `docs/evaluation.md`).

## 6. Before spending money

```python
from flint.mixture import find_overlap, load_jsonl
find_overlap(load_jsonl(mix), load_jsonl(dev))   # must be 0
```

`flint.train` and `flint.evaluate` both refuse to run when this is non-zero. It
compares `(state, questions)` fingerprints and ignores `_meta` and labels, which
is the only comparison that catches the case where the same content was stamped
with different provenance. `docs/evaluation.md` §5 describes the leak that passed
an input-only check and inflated a result by twenty points.

Overlap zero is necessary and not sufficient. A suite generated by the same
generator as a training source shares the *task* without sharing any record. The
only defence is to generate the held-out set with the generator explicitly told
to exclude the families the training set uses, and to keep that instruction in
the mixture spec.

## 7. The checkpoint

```
runs/<name>/
  encoder/                 the backbone, saved whole (transformers format)
  head.safetensors         the head's own tensors only
  rl_agent_config.json     engine config: encoder, head layers, temperatures
  train_meta.json          mixture, hyper-parameters, dev scores, step history
  temperature.json         the fitted operating point, per type and per width
```

`train_meta.json` is part of the artifact. A checkpoint whose training mixture and
seed are unknown cannot be reproduced, re-evaluated or defended, and the number
it reports becomes unfalsifiable.

## 8. Reproducing the reference build

```bash
flint-train \
  --mix data/mix.jsonl \
  --dev data/dev.jsonl \
  --out runs/flint-0.1.0 \
  --encoder jhu-clsp/mmBERT-base \
  --epochs 1 --batch 16 \
  --lr-enc 8e-6 --lr-head 3e-5 \
  --max-len 1024 --head-max-len 512 \
  --smooth 0.03 --seed 20260925 \
  --device mps
```

The temperature is fitted at the end of the run against `--max-error-rate`
(default 0.10) and written to `temperature.json`. Fit it on one set and confirm it
on another; a temperature chosen and reported on the same data cannot fail, and
therefore proves nothing.
