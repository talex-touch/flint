"""Loading a Flint checkpoint and answering with it.

One implementation, used by all three consumers — training (to score the dev
set), evaluation (to produce a report) and serving (to answer a request). A
second implementation would eventually disagree with the first, and the number
it disagreed about would be a published one.

A checkpoint directory is the layout the engine loads, unchanged:

```
<ckpt>/
  model.safetensors     the whole model — backbone and decision head — in one file
  encoder/config.json   the backbone's architecture; a config and no weights
  tokenizer/            the tokenizer, as the backbone saved it
  rl_agent_config.json  engine config: encoder id, head layers, temperature(s)
  train_meta.json       what produced it: mixture, hyper-parameters, dev scores
```

The temperature is not a file of its own. The engine reads it out of
``rl_agent_config.json``, so a checkpoint carries one answer to "how sharp is
this" rather than two that can drift apart — and the one it carries is the one
the engine will actually serve with.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import torch

from . import metrics

__all__ = ["QTYPE_INDEX", "QTYPE_NAMES", "answer_index", "FlintCheckpoint", "temperature_bucket"]

#: Must match ``laya.common.QTYPES``. The engine indexes its type embedding and
#: its temperature list with these integers, so a mismatch here would not raise
#: anywhere — it would just quietly score one question type with another's
#: settings.
QTYPE_INDEX = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPE_INDEX.items()}


def answer_index(qtype: str, criteria, label):
    """Where the correct answer sits in the option list, or ``None`` if it is not
    in it.

    The three question types store their label in three different shapes, which
    is the engine's format rather than a choice of ours: a ``choice`` label is
    the *key* of the right option, a ``score`` label is an *index* into the
    scale, and a ``noul`` label is a *boolean* for yes. Returning ``None``
    instead of raising lets the caller count skipped records — a mixture that
    silently loses 10% of its rows to a label format mistake trains anyway and
    reports nothing.
    """
    if qtype == "choice":
        if not isinstance(criteria, dict):
            return None
        keys = list(criteria)
        return keys.index(label) if label in keys else None
    if qtype == "noul":
        if isinstance(label, bool):
            return 1 if label else 0
        return None
    if qtype == "score":
        if not isinstance(criteria, list):
            return None
        try:
            i = int(label)
        except (TypeError, ValueError):
            return None
        return i if 0 <= i < len(criteria) else None
    return None


def temperature_bucket(qtype: str, k: int) -> str:
    """The engine's option-count bucket key, e.g. ``"choice:3-5"``."""
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return "%s:%s" % (qtype, size)


@dataclass
class FlintCheckpoint:
    """A loaded checkpoint, ready to answer."""

    path: str
    cfg: dict
    model: torch.nn.Module
    tokenizer: object
    device: torch.device
    temperature: list = field(default_factory=lambda: [1.0, 1.0, 1.0])
    temperature_by_options: dict = field(default_factory=dict)
    train_meta: dict = field(default_factory=dict)
    batch_size: int = 8

    @classmethod
    def load(cls, path: str, device: str | None = None) -> "FlintCheckpoint":
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        from . import engine  # imported lazily: this module must import without torch

        cfg_path = os.path.join(path, "rl_agent_config.json")
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)

        dev = torch.device(device) if device else torch.device("cpu")
        # The backbone's directory holds a config and no weights, so the model is
        # built from the architecture and then filled in whole from
        # model.safetensors. Loading the backbone separately from that directory
        # would only work in a layout where it also held weights, which in this
        # one it does not — and the failure would look like a shape mismatch
        # rather than a missing file.
        model = engine.build_model(cfg, encoder_dir=os.path.join(path, "encoder")).to(dev).eval()
        state = load_file(os.path.join(path, "model.safetensors"))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise ValueError(
                "checkpoint %s does not match this engine build: %d tensors missing (%s), "
                "%d unknown (%s)"
                % (path, len(missing), ", ".join(sorted(missing)[:3]),
                   len(unexpected), ", ".join(sorted(unexpected)[:3]))
            )

        tokenizer = AutoTokenizer.from_pretrained(os.path.join(path, "tokenizer"))

        meta = {}
        meta_path = os.path.join(path, "train_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)

        return cls(
            path=path, cfg=cfg, model=model, tokenizer=tokenizer, device=dev,
            temperature=list(cfg.get("temperature") or [1.0, 1.0, 1.0]),
            temperature_by_options=dict(cfg.get("temperature_by_options") or {}),
            train_meta=meta,
        )

    def scale_for(self, qtype: str, k: int) -> float:
        """The temperature this question is served at.

        The option-count bucket wins when the checkpoint carries one: entropy
        confidence is normalised by ``log k``, so a 3-option question and a
        30-option one do not sit on the same confidence scale and do not want
        the same correction.
        """
        qt = QTYPE_INDEX[qtype]
        return float(self.temperature_by_options.get(temperature_bucket(qtype, k), self.temperature[qt]))

    @torch.no_grad()
    def answer(self, records, option_seed: int | None = None) -> list[dict]:
        """Answer every question in every record.

        Returns one dict per record, keyed by question key, each holding the
        chosen option, the full distribution and the confidence the gate reads.
        Nothing here decides whether to *use* an answer — that is the caller's
        threshold, and it belongs to the caller because only the caller knows
        what a wrong answer costs.
        """
        from . import engine

        # Records and questions are two different index spaces: a record may hold
        # several questions, and a question may fail to build. Everything below
        # flattens to items and keeps a per-record map back, rather than assuming
        # they line up — they do for a one-question record, which is exactly why
        # the assumption is invisible until it is wrong.
        items: list[dict] = []
        per_record: list[dict[str, int]] = []
        for r_i, record in enumerate(records):
            row_order: dict[str, int] = {}
            for q_i, (key, question) in enumerate(record["questions"].items()):
                built = engine.to_item(
                    self.tokenizer, record, key, question,
                    max_len=self.cfg.get("max_len", 1024),
                    head_max_len=self.cfg.get("head_max_len", 512),
                    smoothing=0.0,
                    option_seed=None if option_seed is None else option_seed + r_i * 977 + q_i,
                    # A client asks the question; it does not supply the answer.
                    require_label=False,
                )
                if built is None:
                    continue
                row_order[key] = len(items)
                items.append(built)
            per_record.append(row_order)

        if not items:
            return [{} for _ in records]

        probabilities: list[np.ndarray | None] = [None] * len(items)
        for start in range(0, len(items), self.batch_size):
            chunk = items[start:start + self.batch_size]
            collated = engine.collate(self.tokenizer, [chunk])
            if collated is None:
                continue
            logits, _ = self.model(
                collated["input_ids"].to(self.device),
                collated["attention_mask"].to(self.device),
                collated["marker_pos"].to(self.device),
                collated["marker_mask"].to(self.device),
                collated["qtype"].to(self.device),
            )
            mask = collated["marker_mask"]
            for i in range(len(chunk)):
                k = int(mask[i].sum())
                probabilities[start + i] = (
                    torch.softmax(logits[i, :k].float(), -1).cpu().numpy().astype(np.float64)
                )

        answers = []
        for r_i, record in enumerate(records):
            per_question = {}
            for key, question in record["questions"].items():
                pos = per_record[r_i].get(key)
                probs = probabilities[pos] if pos is not None else None
                if probs is None:
                    continue
                qtype = question.get("type", "choice")
                scaled = metrics.apply_temperature(
                    probs[None, :], self.scale_for(qtype, len(probs))
                )[0]
                per_question[key] = {
                    "choice": _label_for(qtype, question.get("criteria"), int(np.argmax(scaled))),
                    "probabilities": _probabilities_for(qtype, question.get("criteria"), scaled),
                    "confidence": float(metrics.confidence(scaled[None, :])[0]),
                }
            answers.append(per_question)
        return answers


def _label_for(qtype: str, criteria, index: int):
    """Turn a position back into the shape the caller sent."""
    if qtype == "choice" and isinstance(criteria, dict):
        keys = list(criteria)
        return keys[index] if index < len(keys) else None
    if qtype == "noul":
        return bool(index == 1)
    return index


def _probabilities_for(qtype: str, criteria, probs) -> dict:
    if qtype == "choice" and isinstance(criteria, dict):
        keys = list(criteria)
        return {k: float(probs[i]) for i, k in enumerate(keys) if i < len(probs)}
    if qtype == "noul":
        return {"true": float(probs[1]) if len(probs) > 1 else 0.0,
                "false": float(probs[0])}
    return {str(i): float(p) for i, p in enumerate(probs)}
