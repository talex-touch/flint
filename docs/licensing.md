# Licensing

Two licences, because there are two different things here, and conflating them is
how projects end up unable to say what a user may actually do.

| what | licence | where |
| --- | --- | --- |
| code, schemas, documentation | **MIT** | [`LICENSE`](../LICENSE) |
| the released weights (`talex-flint-1.0`) | **Apache-2.0** | shipped in the release archive as `LICENSE` + `NOTICE` |

## Why the code licence is unmodified

`LICENSE` is the MIT text with nothing added. A licence file that is *almost* the
standard text stops being recognised as one — GitHub reports the repository as
having no licence at all — and a repository that reads as "no licence" is worse
than one with a narrower licence, because nobody can act on it. Anything that
belongs next to the grant goes in this file instead.

## Why the weights are Apache-2.0 and not MIT

A checkpoint is a derivative work of the things it was made from, so its licence
has to be one those things permit. This one is built from:

| upstream | licence | what it contributes |
| --- | --- | --- |
| `laya` (engine + released checkpoint) | Apache-2.0 | the architecture, the decision head, and the encoder weights this run started from |
| `jhu-clsp/mmBERT-base` | **MIT** | the backbone that `laya` was itself built on |

Both are permissive, so the result may be relicensed — but Apache-2.0 is the one
that carries forward the notice obligations of both. Releasing these weights as
MIT would mean dropping an Apache-2.0 notice, which is not ours to drop. Code
written in this repository has no such obligation, so it stays MIT.

## What a different backbone does to that answer

`flint.train --encoder <id>` builds a checkpoint whose terms are not these. The
`--encoder` you pass is the one whose licence you inherit, and the engine will
record it in the checkpoint's `rl_agent_config.json` (`encoder`) and
`train_meta.json` (`base`) so the answer stays recoverable later. A restrictive
backbone makes the checkpoint unrestrictive-but-not-releasable; a share-alike one
makes it share-alike. Neither is a problem until you publish.

## Checkpoints trained on data you cannot redistribute

This is the condition that catches people, because it is invisible in the
artifact. A checkpoint is only publishable when **every corpus that produced its
targets** permits redistribution of derived works, and `docs/training.md` §2
records the licences actually found on the corpora that would be the obvious
choices: `fancyzhx/ag_news` reports `unknown`, `SetFit/sst5` declares nothing,
`google/boolq` is CC-BY-**SA**, `yelp_review_full` is `other`, `nyu-mll/multi_nli`
is a mixture that includes a share-alike component.

Training on those and publishing the result is a licence problem inherited
silently by everyone who downloads it, which is why the mixture behind
`talex-flint-1.0` was **synthesised** — queries generated from a private pool and
labelled by a model this organisation owns — rather than scraped from a corpus
whose terms nobody checked. That is not a shortcut around the licence question;
it is the answer to it.

The general rule is the one this organisation already applies to weights
extracted from a vendor's product: **ideas are reusable, weights are not.** Point
`flint-build` at a spec listing corpora you can defend, and the checkpoint
`flint-train` writes is yours to license with no part of the MIT grant above
needed for it.
