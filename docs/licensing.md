# Licensing

The `LICENSE` file at the repository root is the MIT licence, unmodified, and
covers the code, schemas and documentation in this repository. It is kept free of
additions on purpose: a licence file that is *almost* the standard text stops
being recognised as one, and a repository that reads as "no licence" is worse
than one with a narrower licence, because nobody can act on it.

Two things the MIT licence does not cover, and why they are separated:

## Backbone weights

A Flint checkpoint is a head trained on top of an encoder someone else released.
The encoder keeps its own upstream licence, which is recorded in the checkpoint's
`train_meta.json` under `base`. This repository does not relicense it, and
`flint.train --encoder <id>` builds a checkpoint whose terms depend on the
encoder it was given.

The reference backbone is Apache-2.0, which places no restriction on the
resulting checkpoint. A different encoder may not.

## Checkpoints trained on data we cannot redistribute

A checkpoint is only publishable when **both** the backbone and every corpus that
produced its targets permit redistribution of derived works. That second
condition is the one that catches people, and `docs/training.md` §2 records the
licences actually found for the corpora that would be the obvious choices: most
of them are unlicensed, undeclared, share-alike or "other".

**No weights are published in this repository**, and the reason is that rule
rather than an unfinished release. What is published is the recipe, so a corpus
you can defend can be substituted and the run repeated — and then the checkpoint
is yours to license, and no part of this repository's MIT grant is needed for it.

This is the same position the organisation takes on weights extracted from a
vendor's product: ideas are reusable, weights are not.
