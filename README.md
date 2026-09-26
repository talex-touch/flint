# flint

**An on-device model that makes decisions instead of writing text.**

[English](README.md) · [简体中文](README.zh.md)

You give Flint a piece of context and a small closed set of options. It returns
the option the evidence picks, together with a confidence it can be held to. It
never generates text, so it cannot hallucinate one, and it answers in tens of
milliseconds on a laptop.

It exists for the decisions an application makes constantly and cannot afford to
send anywhere: **which of these candidates the user meant, which of these values
belongs in this field, which of these actions follows from this state.** Selection
and preference — choosing among things that already exist — rather than
generation. A model that answers in tens of milliseconds locally can be asked on
every keystroke; one that answers in a second cannot be asked at all.

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

Measured on the released checkpoint — 322M parameters, fp16, on an Apple M4 Pro.
The host is named because a latency without a host is a rumour, and the revision
is named because a number without a revision is a different model next month.

| quantity | value | how |
| --- | --- | --- |
| one 16-option decision | **24.8 ms** p50, 25.1 ms p95 | MLX, warm, single request, 15 samples |
| one question across the suites | p50 10.3 ms, p95 58.9 ms | 1,380 questions, option counts as the suites have them |
| the model on disk | 643,835,524 bytes | fp16, CPU or GPU, no quantisation required |

Accuracy is not one number, and printing a single one would be the dishonesty
this repository exists to avoid. It is useful where the task resembles what it
was trained for, and near chance where it does not:

| suite | n | accuracy | |
| --- | ---: | ---: | --- |
| paste, calibration set | 200 | **78.5%** | the temperature was fitted on this set — optimistic |
| pinyin candidates, out-of-domain | 96 | 67.7% | adjacent, unseen wording |
| short continuation, out-of-domain | 48 | 56.2% | adjacent, unseen wording |
| paste, out-of-domain | 192 | 34.4% | same task, unseen state |
| MMLU | 116 | 23.3% | abstract reasoning — 4 options |
| MMLU-Pro | 200 | 13.5% | abstract reasoning — 10 options |

The last four rows are the honest ones, and the 78.5% is not one of them: accuracy
does not depend on the temperature, but the threshold does, and the set that chose
the operating point cannot also report it. A decision model that keeps its
in-domain accuracy when the state changes is not what this is — the state is the
input, and a state it has not seen is a task it has not seen.

Its distinguishing property is not accuracy. It is **abstention** — knowing when
it does not know — and on that axis the recipe change that produced this
checkpoint is worth more than accuracy is:

| | mean confidence drop | pairs where it fell |
| --- | ---: | ---: |
| previous recipe | −0.0423 | 35.5% |
| **this checkpoint** | **+0.0550** | **71.8%** |

Read the last row honestly: those cover the families the mixture contains, where
the drop is **+0.0784**. On the families it does not contain the number is −0.0505
over twenty pairs — the effect is not demonstrated there, and it is not a general
rule about abstention. Confidence here is the gate's own, normalised entropy; the
same checkpoints scored on top probability would give different magnitudes, which
is why
[`docs/evaluation.md`](docs/evaluation.md) §6 pins the basis down and is the part
of this repository worth reading first.

What we do **not** have: a public leaderboard number. This is not a smaller chat
model and there is no benchmark it can be dropped into — the task is "choose
among these options", and the only honest measurement is on a suite you can
inspect. `docs/evaluation.md` describes how to build one and how to keep it from
lying.

## The released model

**`talex-flint-1.0`** — 322M parameters, fp16, Apache-2.0.

[**Download**](https://github.com/talex-touch/flint/releases/tag/talex-flint-1.0) —
the release body is the model card and carries the archive's sha256.

```bash
# MLX, what the latency above was measured with
pip install laya-mlx
python -c "import laya_mlx; a = laya_mlx.load('talex-flint-1.0'); print(a.predict('Order 4471 shipped.', {'q': {'type': 'noul', 'instructions': 'was it shipped?'}}))"

# PyTorch, through this repository
python -c "
from flint.inference import FlintCheckpoint
c = FlintCheckpoint.load('talex-flint-1.0')
print(c.answer([{'state': 'Order 4471 shipped.', 'questions': {'q': {'type': 'noul', 'instructions': 'was it shipped?'}}}])[0])"
```

Both paths load the same directory and were checked against each other: on 200
held-out paste decisions they agree on **200/200** options.

Weights are not committed to this repository — 643 MB does not belong in git.
They are not a separate product either: the archive is the checkpoint this
recipe writes, and `flint-train` will write another one the same way from a
mixture you can defend. `docs/licensing.md` records why the weights are
Apache-2.0 while the code is MIT, and what that distinction reaches.

## License

| | |
| --- | --- |
| code, schemas, documentation | **MIT** — [`LICENSE`](LICENSE) |
| the released weights (`talex-flint-1.0`) | **Apache-2.0** |

Two licences because they are two different things. The weights are a derivative
of two permissively licensed upstreams — the `laya` engine and its released
checkpoint (Apache-2.0), over `jhu-clsp/mmBERT-base` (MIT) — and Apache-2.0 is
the one that carries forward every notice those licences require. Code written
here has no such obligation and stays MIT.

[`docs/licensing.md`](docs/licensing.md) covers the part that is easy to get
wrong: what a *different* backbone would do to that answer, and why a checkpoint
trained on a corpus you cannot redistribute is a licence problem inherited
silently by everyone who downloads it. `docs/training.md` §2 records the licences
actually found on the corpora that would be the obvious choices — most of them do
not permit redistribution, which is why the mixture behind these weights was
synthesised instead.
