# flint

**An on-device model that makes decisions instead of writing text.**

You give Flint a piece of context and a small closed set of options. It returns
the option the evidence picks, together with a confidence it can be held to. It
never generates text, so it cannot hallucinate one, and it answers in tens of
milliseconds on a laptop.

It exists for the decisions an application makes constantly and cannot afford to
send anywhere: **which of these candidates the user meant, which of these values
belongs in this field, which of these actions follows from this state.** Selection
and preference — choosing among things that already exist — rather than
generation. A model that answers in 40 ms locally can be asked on every keystroke;
one that answers in a second cannot be asked at all.

This repository is the recipe, the evaluation protocol and the serving contract.
The architecture is not ours and is not reimplemented here.

## Why this is a different shape of model

|  | a chat model | Flint |
| --- | --- | --- |
| output | text, of any shape | one option from a closed set |
| failure mode | a fluent wrong answer | a wrong option, with a probability attached |
| cost | per token | one pass, no tokens |
| where it runs | somewhere else | the machine the user is on |
| what you build with it | a prompt | an `if` |

The important row is the second one. A generated answer is either right or wrong
with no signal in between; a decision carries a probability, so the caller can set
a threshold, act above it and decline below it. That is the whole reason this
model is shaped this way — see `docs/evaluation.md` for what that threshold costs
and buys, measured.

## What is ours and what is not

Flint trains a decision head on top of an encoder someone else released, using an
engine someone else built.

- **Ours:** how a labelled corpus becomes decision records
  (`flint/scenarios.py`), how a training mixture is assembled and audited so it
  is the mixture the recipe claims (`flint/mixture.py`), what the evaluation
  measures and why (`flint/metrics.py`, `docs/evaluation.md`), the inference and
  serving path, and the checkpoints.
- **Not ours:** the architecture. `flint/engine.py` is the only module that talks
  to it, and it is a thin adapter — the engine is a public Apache-2.0 package,
  and `docs/method.md` records the design and the literature it comes from.

Nothing about the engine is copied into this repository. If you want to
understand the model, read the engine; if you want to understand how to train one
that is honest about its own confidence, read `flint/metrics.py`.

## Layout

```
flint/metrics.py        coverage at an error budget, calibration, order sensitivity
flint/scenarios.py      corpus -> decision records, option order randomised
flint/mixture.py        named sources, deduplicated, provenance, audit
flint/engine.py         the one adapter to the upstream decision engine
flint/train.py          the training run, and the fitted operating point
flint/evaluate.py       the report
flint/inference.py      checkpoint layout, loading, answering
flint/serve.py          POST /v1/systemone (stdlib only)
flint/smoke.py          the whole path, offline, in seconds
docs/method.md          the architecture and the literature behind it
docs/training.md        the recipe, and which corpora are safe to use
docs/evaluation.md      what to measure, and the three ways it goes wrong
docs/api.md             request and response format
docs/licensing.md       what the MIT grant reaches, and what it does not
```

## Quickstart

Requires Python 3.10+.

```bash
pip install -e .          # pulls in the engine (Apache-2.0), torch, transformers

# Prove the pipeline works without downloading anything:
python -m flint.smoke
```

`flint.smoke` builds a tiny encoder and tokenizer locally, synthesises decision
records, trains for a few steps, writes a checkpoint, reloads it through the same
loader serving uses and answers a question. It takes seconds and needs no network.
It proves the plumbing, not the model — nothing with a 64-dimensional randomly
initialised encoder is any good.

A real run:

```bash
flint-build --spec spec.json --out data/mix.jsonl
flint-train --mix data/mix.jsonl --dev data/dev.jsonl --out runs/flint-0.1.0 --device mps
flint-evaluate --checkpoint runs/flint-0.1.0 --suite data/dev.jsonl --mix data/mix.jsonl
```

Serving:

```bash
flint-serve --checkpoint runs/flint-0.1.0 --port 8080 --threshold 0.5
curl -s localhost:8080/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Order 4471 shipped on the 3rd. The customer asked to return it on the 9th.",
  "questions": {"pick": {"type": "choice",
                         "instructions": "which option does the state describe?",
                         "criteria": {"c1": "inside the return window",
                                      "c2": "outside the return window"}}}}'
```

```json
{"pick": {"choice": "c2", "probabilities": {"c1": 0.31, "c2": 0.69}, "confidence": 0.19,
          "release": false}}
```

## The numbers we have

Measured on a 322M encoder on an Apple M4 Pro, option counts as noted. These are
from the reference build, not from a marketing run, and the host is named because
a latency without a host is a rumour.

| quantity | value | how |
| --- | --- | --- |
| one 16-option decision | ~40 ms | MLX, warm, single request, CPU+GPU shared |
| one pass over 40 options | under 60 ms | same host, no batching |
| released set at a 5% error budget | depends on the mixture, not the size | `docs/evaluation.md` |

What we do **not** have: a public leaderboard number. This is not a smaller chat
model and there is no benchmark it can be dropped into — the task is "choose among
these options", and the only honest measurement is on a suite you can inspect.
`docs/evaluation.md` describes how to build one and how to keep it from lying.

## No weights are published yet

The recipe and the tooling are here. The weights are not, and the reason is in
`docs/training.md` §2: a checkpoint is only publishable when **every corpus that
produced its targets** permits redistribution of derived works, and most of the
obvious public corpora do not clearly do so — several are unlicensed, one is
share-alike, one is "other". Training on them and publishing the result would be
a licence problem inherited silently by every user.

So the rule here is the one this organisation already applies to weights
extracted from a vendor's product: **ideas are reusable, weights are not.**
Point `flint-build` at corpora you can defend, run `flint-train`, and you have a
checkpoint that is yours to license. `python -m flint.smoke` shows the whole path
running today.

## License

Code, schemas and documentation: **MIT** (see `LICENSE`).

Weights are never committed to this repository, and `LICENSE` says nothing about
them. A checkpoint carries the licences of its backbone encoder and of every
corpus behind its targets; [`docs/licensing.md`](docs/licensing.md) records which
of those the MIT grant does and does not reach, and why the two are separable.
`docs/training.md` records the licences actually found on the corpora that would
be the obvious choices, most of which do not permit redistribution.
