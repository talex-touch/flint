"""The one place that talks to the decision engine.

Flint does not implement the model architecture. It trains and serves a head on
top of a released one, and every call into that engine goes through this module
so there is exactly one place where the two contracts meet.

The contract that matters is the shape mismatch. A client sends

```
{"pick": {"type": "choice", "instructions": "...",
          "criteria": {"c1": "...", "c2": "..."}, "label": "c1"}}
```

while the engine's ``build_sequence`` reads ``{"t": ..., "ins": ..., "crit": ...}``
and expects the label as an *index*. Three question types store their label three
different ways (a key, an index, a boolean), which is the engine's format and not
ours to change — so the translation is explicit here rather than spread across
whatever happened to touch a record.
"""

from __future__ import annotations

import random

__all__ = ["engine", "build_model", "build_model_on", "collate", "to_item", "qtypes"]


def engine():
    """The upstream package, imported on demand.

    Imported lazily so that ``flint.metrics``, ``flint.mixture`` and the test
    suite can run on a machine with no torch, no transformers and no engine
    installed. The metrics are the part of this repository worth reading
    carefully, and they should not require a GPU stack to be checkable.
    """
    try:
        import laya.common as common
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "the decision engine is not installed. `pip install laya` (Apache-2.0) "
            "provides the architecture this repository trains a head on."
        ) from exc
    return common


def qtypes() -> dict:
    """``{"choice": 0, "score": 1, "noul": 2}`` from the engine itself.

    Read rather than hard-coded here, so that a future engine version that
    reorders them cannot silently make flint score every question with the
    wrong type embedding.
    """
    return dict(engine().QTYPES)


def build_model(cfg: dict, encoder_dir: str | None = None):
    """Build the engine's model from a config, optionally with a local encoder."""
    return engine().build_model(cfg, encoder_dir=encoder_dir)


def build_model_on(encoder, cfg: dict):
    """Wrap an already-loaded encoder. Used when loading a saved checkpoint, so
    the encoder is loaded by transformers (which knows how to) rather than
    reconstructed from config (which would give random weights)."""
    return engine().DecisionModel(
        encoder,
        int(cfg.get("head_layers", 2)),
        len(cfg.get("act_costs", {})) + 1,
    )


def collate(tokenizer, groups):
    return engine().collate_items(groups, tokenizer.pad_token_id)


def to_item(
    tokenizer,
    record: dict,
    question_key: str,
    question: dict,
    max_len: int = 1024,
    head_max_len: int = 512,
    smoothing: float = 0.0,
    option_seed: int | None = None,
    require_label: bool = True,
):
    """One question of one record becomes one engine training item, or ``None``.

    ``None`` means "this question cannot be built" — a label that is not among
    the options (when a label is required), an unknown question type, or a
    sequence whose options did not survive tokenisation. The caller counts these;
    a mixture that quietly loses a tenth of its rows to a format mistake trains
    perfectly well and reports nothing.

    ``require_label`` is what separates training from serving. A request from a
    client carries no label — it is asking for one — so serving must not treat
    the absence as a broken record. When it is ``False`` the label defaults to
    option 0 and the target it produces is discarded; the sequence, the markers
    and the option order are built identically either way, which is what keeps
    the served answer comparable to the evaluated one.

    ``smoothing`` mixes a uniform distribution into the target. It is not
    cosmetic: a model trained on hard one-hot targets is being told to be
    certain, and certainty on an ambiguous item is exactly the over-confidence
    that makes a confidence gate useless. Records that carry a teacher
    distribution in ``probs`` already have a soft target and are smoothed on top
    of it, which is what keeps a saturated teacher from collapsing back to
    one-hot.

    ``option_seed`` permutes a ``choice`` question's options, remapping both the
    answer and the target so the question is unchanged and only its layout moved.
    Ordering is only permuted where order is *not* information: a ``score`` scale
    is ascending by definition and a ``noul`` question's false/true slots are
    semantic, so shuffling either would be relabelling, not augmentation. An
    explicit "none" option stays last, because a sentinel that wanders through
    the list teaches position cues rather than the sentinel.
    """
    common = engine()
    qtype = question.get("type", "choice")
    known = qtypes()
    if qtype not in known:
        return None
    criteria = question.get("criteria")

    label = _answer_index(qtype, criteria, question.get("label"))
    if label is None:
        if require_label:
            return None
        label = 0

    built = {"t": qtype, "ins": question.get("instructions", ""), "crit": criteria}
    options = common.render_options(built)
    k = len(options)
    if k < 1:
        return None

    target = _target_from_record(question, criteria, options, label)
    target = _smooth(target, smoothing)

    order = None
    if option_seed is not None and qtype == "choice" and k > 2:
        order = _order_for(k, criteria, option_seed)
        label = order.index(label)
        target = [target[i] for i in order]

    ids, markers = common.build_sequence(
        tokenizer, record["state"], built, max_len, head_max_len, option_order=order
    )
    if len(markers) != k:
        return None

    return {
        "ids": ids,
        "markers": markers,
        "qtype": known[qtype],
        "target": target,
        "label": label,
        "question": question_key,
    }


def _answer_index(qtype: str, criteria, label):
    """Kept here rather than imported from ``inference`` so that the training
    adapter has no dependency on the serving layer."""
    if qtype == "choice":
        if not isinstance(criteria, dict):
            return None
        keys = list(criteria)
        return keys.index(label) if label in keys else None
    if qtype == "noul":
        if isinstance(label, bool):
            return 1 if label else 0
        if isinstance(label, str) and label.lower() in ("true", "false"):
            return 1 if label.lower() == "true" else 0
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


def _target_from_record(question, criteria, options, label) -> list[float]:
    """A teacher distribution if the record carries one, otherwise one-hot.

    ``probs`` is keyed the way the options are (a key for a choice, a boolean for
    a noul, an index for a score). Keys that match nothing contribute zero, and a
    distribution that sums to zero is treated as absent rather than as a uniform
    target — a silently uniform target looks like a working run.
    """
    raw = question.get("probs")
    k = len(options)
    if isinstance(raw, dict) and raw:
        if isinstance(criteria, dict):
            values = [max(0.0, float(raw.get(key, 0.0))) for key in criteria]
        elif isinstance(criteria, list) and "true" in raw:
            values = [max(0.0, float(raw.get("false", 0.0))), max(0.0, float(raw.get("true", 0.0)))]
        else:
            values = [max(0.0, float(raw.get(str(i), raw.get(i, 0.0)))) for i in range(k)]
        total = sum(values)
        if total > 0:
            return [v / total for v in values]

    target = [0.0] * k
    target[label] = 1.0
    return target


def _smooth(target: list[float], smoothing: float) -> list[float]:
    if smoothing <= 0.0:
        return target
    k = len(target)
    weight = min(0.999, float(smoothing))
    return [(1.0 - weight) * v + weight / k for v in target]


def _order_for(k: int, criteria, seed: int) -> list[int]:
    """A permutation that leaves a trailing "none" option in place."""
    rng = random.Random(seed)
    none_at = None
    if isinstance(criteria, dict) and "none" in criteria:
        none_at = list(criteria).index("none")
    movable = [i for i in range(k) if i != none_at]
    rng.shuffle(movable)
    return movable + ([none_at] if none_at is not None else [])
