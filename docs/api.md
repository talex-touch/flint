# The request and response format

Flint speaks TypeSafe's System One shape. That is a deliberate choice: the format
is someone else's published contract rather than a local invention, so a client
written against it can be pointed at a Flint server by changing a base URL, and a
suite written for one can be replayed against the other.

One endpoint: `POST /v1/systemone`. One auxiliary route: `GET /health`.

## The request

```json
{
  "state": "Order 4471 shipped on the 3rd. The customer asked to return it on the 9th.",
  "questions": {
    "pick": {
      "type": "choice",
      "instructions": "which option does the state describe?",
      "criteria": {
        "c1": "inside the return window",
        "c2": "outside the return window"
      }
    }
  }
}
```

`state` is the context and may be a string, an object or a list. An object is
serialised as JSON before it reaches the encoder, which is how a structured state
— a form, a set of candidates, a game position — is passed without inventing a
second format for it.

`questions` maps a caller-chosen key to a question. The key is echoed back; Flint
never invents one. Several questions may share one `state`, and they cannot see
each other: each is packed into its own sequence, so a question about one part of
the state cannot be influenced by what another question in the same request asked.

### The three question types

Each carries `type`, `instructions`, and `criteria`, and the three types differ in
what `criteria` is and what `label` looks like when the record is used for
training. This inconsistency is the upstream format, not a design here.

**`choice`** — pick one of a closed set. `criteria` is an object keyed by
option id; order is not meaningful and is permuted during training and during
order-sensitivity evaluation.

```json
{"type": "choice",
 "instructions": "which option does the state describe?",
 "criteria": {"c1": "inside the return window", "c2": "outside the return window"},
 "label": "c1"}
```

A `none` key is an ordinary option. There is nothing special about it: a model
that can only choose among the things it was shown cannot say "none of these",
and that is the answer a suggestion product needs most often.

**`noul`** — a yes/no judgement. `criteria` is an object with `false` and `true`
descriptions, and the label is a boolean. It is not a two-option choice: the
engine reports a single probability for yes.

```json
{"type": "noul",
 "instructions": "would this reviewer recommend the business?",
 "criteria": {"false": "negative or mixed", "true": "clearly positive"},
 "label": true}
```

**`score`** — a rating on an ordered scale. `criteria` is a **list** of
descriptions in ascending level order, and the label is an **index** into it.

```json
{"type": "score",
 "instructions": "which pricing tier applies?",
 "criteria": ["list price", "volume discount", "wholesale rate"],
 "label": 1}
```

### Training records

A training record is the same object with two optional additions:

- `probs` — a distribution over the options, keyed the way `criteria` is. When
  present it becomes the training target instead of a one-hot label, which is
  what distillation from a stronger teacher produces. `label` is still used for
  accuracy and for fitting the temperature.
- `_meta` — bookkeeping. `flint.mixture` writes `_meta.source` on every row it
  emits, and that field is the only thing that makes a mixture auditable after
  the fact.

`_meta` is excluded from the record fingerprint that duplicate detection and leak
detection use. A build step that stamps a different `_meta` onto the same content
produces two rows that are still the same row.

## The response

```json
{
  "pick": {
    "choice": "c2",
    "probabilities": {"c1": 0.28, "c2": 0.71},
    "confidence": 0.24
  }
}
```

- `choice` is the winning option, in the same shape the caller sent it: a
  criteria key for a choice, a boolean for a noul, an index for a score.
- `probabilities` is the full distribution, keyed the same way. It sums to 1.
- `confidence` is **normalised entropy confidence, `1 - H(p) / log k`**, where
  `k` is the number of options *for that question*. It is not the top
  probability. Section "Confidence" below explains why the difference is not
  cosmetic.
- `release` appears only when the server was started with `--threshold`, and is
  `confidence >= threshold`. It reports what the configured operating point would
  do; the server never withholds an answer on its own, because only the caller
  knows what a wrong answer costs.

## Confidence is entropy, not the top probability

The two are different numbers, and the difference decides which answers ship.

```
a choice between 2 options at (0.9, 0.1)        confidence 0.531
a choice between 10 options at (0.9, 0.011×9)   confidence 0.763
```

Both rows put 0.9 on their winner, and the second is scored as **more**
confident. That is the part that reads backwards: a 0.9 against a single 0.1 is
a near-tie in relative terms, while a 0.9 against nine 0.011s is a decisive lead.
Normalised entropy measures how decisively the winner leads *relative to the size
of the field*, and the field is usually between two and forty options in a
suggestion product — so the range is not academic.

Two consequences that follow and are easy to get wrong:

1. **A threshold means something different at each option count.** A gate fitted
   on three-option questions is not the same gate at thirty.
2. **A temperature moves rows against each other.** A temperature is monotone, so
   it never changes which option wins — but it changes each row's entropy by an
   amount that depends on how peaked that row already was. The confidence
   *ordering* changes, and the accepted set moves with it. This is the mechanism
   behind the coverage/error trade-off in `docs/evaluation.md`.

## Errors

| Status | When |
| --- | --- |
| `400` | body is not JSON; no `state`; no `questions`; an empty body |
| `404` | a path other than `/v1/systemone` or `/health` |
| `413` | body larger than 8 MiB |

Every error body is `{"error": "<what was wrong>"}`.

## Health

```json
{
  "status": "ok",
  "temperature": [1.0, 1.0, 1.0],
  "temperatureByOptions": {"choice:3-5": 1.5},
  "threshold": 0.5
}
```

The temperature is worth exposing: it is part of what a checkpoint *is*, not a
serving detail. Two servers on the same weights with different temperatures are
different products, and a report that does not name the temperature it was
measured at cannot be reproduced.
