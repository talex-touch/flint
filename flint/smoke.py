"""An end-to-end run that needs no network, no GPU and no downloads.

Everything else in this repository is a claim about a pipeline that normally
costs hours and a multi-gigabyte encoder to exercise. This builds a tiny encoder
and a tiny tokenizer in a temporary directory, synthesises a handful of decision
records, trains for a few steps, fits a temperature, reloads the result through
the same loader serving uses, and answers a question with it.

Run it after any change to the adapter, the trainer or the checkpoint layout:

```bash
python -m flint.smoke
```

It is not a test of model quality — nothing with a 64-dimensional encoder
randomly initialised can be. It is a test that the plumbing is not broken, which
is the failure that otherwise shows up four hours into a real run.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile

__all__ = ["run", "main"]

WORDS = [
    "order", "invoice", "refund", "window", "days", "customer", "tier", "unit",
    "price", "discount", "shipping", "delay", "warranty", "claim", "account",
    "eligible", "policy", "case", "amount", "total",
]


def _tiny_tokenizer(directory: str, extra_vocab):
    """A word-level BERT tokenizer over a vocabulary written to disk.

    Built here rather than downloaded so that this file proves the pipeline works
    on a machine with no model cache — and so a failure is unambiguously ours.
    """
    from transformers import BertTokenizerFast

    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    for word in list(WORDS) + list(extra_vocab):
        if word not in vocab:
            vocab.append(word)
    for token in ("?", ":", ",", ".", "none", "of", "these", "applies", "option", "level"):
        if token not in vocab:
            vocab.append(token)

    path = os.path.join(directory, "vocab.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(vocab) + "\n")
    return BertTokenizerFast(vocab_file=path, do_lower_case=True)


def _tiny_encoder(vocab_size: int, hidden: int = 64):
    from transformers import AutoModel, BertConfig

    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=hidden * 2,
        max_position_embeddings=512,
    )
    return AutoModel.from_config(config)


def _synthetic_records(count: int, seed: int, labels):
    from .scenarios import choice_records

    import random

    rng = random.Random(seed)
    items = []
    for i in range(count):
        text = " ".join(rng.choice(WORDS) for _ in range(8))
        items.append(("%s case %d" % (text, i), rng.randrange(len(labels))))
    return choice_records(items, labels, width=4, seed=seed, with_none=True)


def run(workdir: str | None = None, steps: int = 12) -> dict:
    from . import engine
    from .mixture import write_jsonl
    from .train import train

    labels = ["refund", "price", "shipping", "warranty", "account", "delay"]
    keep = workdir or tempfile.mkdtemp(prefix="flint-smoke-")
    os.makedirs(keep, exist_ok=True)

    encoder_dir = os.path.join(keep, "base-encoder")
    os.makedirs(encoder_dir, exist_ok=True)
    tokenizer = _tiny_tokenizer(encoder_dir, extra_vocab=labels)
    encoder = _tiny_encoder(len(tokenizer))
    encoder.save_pretrained(encoder_dir)
    tokenizer.save_pretrained(encoder_dir)

    records = _synthetic_records(48, seed=1, labels=labels)
    dev = _synthetic_records(12, seed=2, labels=labels)
    mix_path = os.path.join(keep, "mix.jsonl")
    dev_path = os.path.join(keep, "dev.jsonl")
    write_jsonl(mix_path, records)
    write_jsonl(dev_path, dev)

    out = os.path.join(keep, "checkpoint")
    meta = train(argparse.Namespace(
        mix=mix_path, dev=dev_path, out=out,
        encoder=encoder_dir, tokenizer=None, tokenizer_subfolder=None,
        init_from=None,
        head_layers=2, epochs=1, batch=8,
        lr_enc=1e-5, lr_head=1e-4,
        max_len=256, head_max_len=192,
        smooth=0.03, freeze_encoder=True, no_shuffle_options=False,
        max_error_rate=0.10, max_steps=steps, log_every=max(1, steps // 3),
        seed=20260925, device="cpu",
    ))

    # Reload through the same path serving uses, and answer with it. A checkpoint
    # that trains but does not load is the failure this step exists to catch.
    from .inference import FlintCheckpoint

    ckpt = FlintCheckpoint.load(out, device="cpu")
    answers = ckpt.answer(dev[:3], option_seed=7)
    first = answers[0]["pick"]
    assert first["choice"] in dev[0]["questions"]["pick"]["criteria"], (
        "the served answer is not one of the options that were offered"
    )
    assert abs(sum(first["probabilities"].values()) - 1.0) < 1e-6

    summary = {
        "workdir": keep,
        "mixtureRows": meta["mixRows"],
        "trainItems": meta["trainItems"],
        "skipped": meta["skipped"],
        "steps": meta["steps"],
        "devAccuracy": meta["devAccuracy"],
        "temperature": meta["temperature"],
        "temperatureByOptions": meta["temperatureByOptions"],
        "sampleAnswer": first,
        "engineQtypes": engine.qtypes(),
    }
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workdir", default=None, help="keep the run here instead of a temp dir")
    ap.add_argument("--steps", type=int, default=12)
    args = ap.parse_args()
    summary = run(args.workdir, args.steps)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
