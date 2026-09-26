"""Metrics for a decision model.

A decision model is judged on two axes that a plain accuracy number hides:

1. **How much can be released at a fixed error budget.** Flint is allowed to
   decline, and a product gates on its confidence — so the question is not "what
   is the accuracy" but "at 5% wrong, how many answers can ship".
2. **Whether its confidence means anything.** A model that is 95% accurate but
   says 0.99 on the 5% it gets wrong cannot be gated on at all.

Inputs are either a ``[n, k]`` array (every row the same number of options) or a
sequence of 1-D arrays (rows with different option counts). The ragged form is
not a convenience: confidence here is normalised by the option count, so padding
a 4-option question out to 40 columns does not merely add zeros — it changes the
confidence the gate would read. Pooling different widths is something the
runtime does, per question, with its own ``k``; these functions have to do the
same or they measure a product nobody ships.

The reject option and the error/coverage trade-off are old ideas in statistical
decision theory; ``docs/method.md`` records the references.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "confidence",
    "top_probability",
    "accuracy",
    "gated_error",
    "coverage_at_error",
    "risk_at_coverage",
    "selective_risk_curve",
    "expected_calibration_error",
    "apply_temperature",
    "select_temperature",
    "paired_confidence_drop",
]


def _rows(probs) -> list[np.ndarray]:
    """Normalise either input shape to a list of 1-D probability rows."""
    if isinstance(probs, np.ndarray) and probs.ndim == 2:
        candidates = [probs[i] for i in range(probs.shape[0])]
    elif isinstance(probs, np.ndarray) and probs.ndim == 1:
        raise ValueError("a single 1-D array is ambiguous; pass [1, k] or a list of rows")
    else:
        candidates = list(probs)

    out = []
    for i, row in enumerate(candidates):
        r = np.asarray(row, dtype=np.float64)
        if r.ndim != 1:
            raise ValueError("row %d is not 1-D (shape %r)" % (i, r.shape))
        if len(r) < 2:
            raise ValueError("row %d has %d option(s); a decision needs at least two"
                             % (i, len(r)))
        if not np.all(np.isfinite(r)):
            raise ValueError("row %d contains non-finite values" % i)
        if r.min() < 0:
            raise ValueError("row %d contains negative values" % i)
        out.append(r)
    return out


def _labels(labels, n: int) -> np.ndarray:
    y = np.asarray(labels, dtype=np.int64)
    if y.shape != (n,):
        raise ValueError("labels must have shape (%d,), got %r" % (n, y.shape))
    return y


def top_probability(probs) -> np.ndarray:
    """The probability the model put on the option it picked, one per row."""
    return np.array([float(r.max()) for r in _rows(probs)])


def confidence(probs) -> np.ndarray:
    """The model's stated confidence, as the runtime computes it.

    **This is normalised Shannon entropy confidence — ``1 - H(p) / log k`` — not
    the top probability.** The engine computes it this way, the gate is a
    threshold on it, and ``k`` is that question's own option count. The two
    quantities are not interchangeable, and they are not even monotone in each
    other across questions with different ``k``, which is why this module keeps
    rows ragged.

    The consequence worth knowing before reading any number below: a temperature
    rescales each row's entropy by an amount that depends on how peaked *that
    row* already was, so it reorders rows against each other and therefore
    changes which rows a fixed threshold accepts. That is the mechanism behind
    the coverage/error trade-off in `select_temperature`.
    """
    out = []
    for r in _rows(probs):
        k = len(r)
        safe = np.clip(r, 1e-12, 1.0)
        entropy = float(-(safe * np.log(safe)).sum())
        out.append(float(np.clip(1.0 - entropy / np.log(k), 0.0, 1.0)))
    return np.array(out)


def accuracy(probs, labels) -> float:
    """Share of rows where the highest-probability option is the label."""
    rows = _rows(probs)
    y = _labels(labels, len(rows))
    if not rows:
        return float("nan")
    return float(np.mean([int(np.argmax(r)) == int(t) for r, t in zip(rows, y)]))


def gated_error(probs, labels, threshold: float):
    """Error inside the accepted set, and how much of the whole that is.

    Returns ``(coverage, error)``. A threshold that accepts nothing returns
    ``(0.0, nan)`` — an empty set has no error rate, and reporting 0.0 would read
    as a perfect one.
    """
    rows = _rows(probs)
    y = _labels(labels, len(rows))
    if not rows:
        return 0.0, float("nan")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1], got %r" % (threshold,))
    conf = confidence(rows)
    accepted = conf >= threshold
    n = int(accepted.sum())
    if n == 0:
        return 0.0, float("nan")
    wrong = [int(np.argmax(r)) != int(t) for r, t in zip(rows, y)]
    return n / len(rows), float(np.mean([w for w, a in zip(wrong, accepted) if a]))


def coverage_at_error(probs, labels, max_error_rate: float):
    """The largest prefix of decisions, in confidence order, under a fixed error
    budget.

    Decisions are taken most-confident first; the returned coverage is the
    largest fraction whose error rate does not exceed ``max_error_rate``, and the
    returned threshold is the confidence of the last accepted decision. Returns
    ``(coverage, threshold)``.

    Ties at the boundary are the caller's problem: deploying ``threshold`` with
    an inclusive comparison accepts every row at that confidence, which can be a
    slightly larger set than the prefix counted here. ``gated_error`` reports
    what a deployed threshold actually does.
    """
    rows = _rows(probs)
    y = _labels(labels, len(rows))
    if not rows:
        return 0.0, float("nan")
    if not 0.0 <= max_error_rate <= 1.0:
        raise ValueError("max_error_rate must be in [0, 1], got %r" % (max_error_rate,))

    conf = confidence(rows)
    wrong = np.array([int(np.argmax(r)) != int(t) for r, t in zip(rows, y)], dtype=np.float64)
    order = np.argsort(-conf, kind="stable")
    cumulative = np.cumsum(wrong[order])
    counts = np.arange(1, len(rows) + 1, dtype=np.float64)
    ok = cumulative / counts <= max_error_rate
    best = int(np.flatnonzero(ok)[-1]) + 1 if ok.any() else 0
    if best == 0:
        return 0.0, float("nan")
    return best / len(rows), float(conf[order[best - 1]])


def risk_at_coverage(probs, labels, coverage: float):
    """Error rate of the most-confident ``coverage`` share of decisions.

    The other direction of the same trade-off: fix how much must be answered and
    read off how wrong that is. This is the natural way to compare two models
    across their whole operating range instead of at one arbitrary threshold.
    """
    rows = _rows(probs)
    y = _labels(labels, len(rows))
    if not rows:
        return float("nan"), float("nan")
    if not 0.0 < coverage <= 1.0:
        raise ValueError("coverage must be in (0, 1], got %r" % (coverage,))
    conf = confidence(rows)
    wrong = np.array([int(np.argmax(r)) != int(t) for r, t in zip(rows, y)], dtype=np.float64)
    order = np.argsort(-conf, kind="stable")
    k = max(1, int(round(coverage * len(rows))))
    return float(wrong[order[:k]].mean()), float(conf[order[k - 1]])


def selective_risk_curve(probs, labels, coverages=(0.6, 0.7, 0.8, 0.9, 0.95)):
    """``risk_at_coverage`` at several coverages, as a list of dicts."""
    out = []
    for c in coverages:
        err, thr = risk_at_coverage(probs, labels, c)
        out.append({"coverage": float(c), "error": err, "threshold": thr})
    return out


def expected_calibration_error(probs, labels, bins: int = 15, conf=None) -> float:
    """Gap between stated confidence and observed accuracy, binned.

    Equal-width bins over the observed confidence range, weighted by population.
    By default the confidence used is the **top probability**, which is the
    convention this metric is normally quoted under; pass
    ``conf=metrics.confidence(probs)`` to measure it on the entropy confidence
    the runtime gates on instead. The two answer different questions — "is the
    top number honest" versus "is the number the gate reads honest" — and a model
    can pass one and fail the other.

    ``laya.common.ece_score`` computes the same quantity inside the engine; this
    copy exists so the metric can be evaluated and tested without the engine
    installed.
    """
    rows = _rows(probs)
    y = _labels(labels, len(rows))
    if not rows:
        return float("nan")
    if bins < 1:
        raise ValueError("bins must be >= 1")
    c = top_probability(rows) if conf is None else np.asarray(conf, dtype=np.float64)
    if c.shape != (len(rows),):
        raise ValueError("conf must have shape (%d,), got %r" % (len(rows), c.shape))
    correct = np.array([int(np.argmax(r)) == int(t) for r, t in zip(rows, y)], dtype=np.float64)

    if c.max() <= c.min():
        # Every row carries the same confidence: the metric degenerates to the
        # gap between that number and the observed accuracy, which is the honest
        # answer rather than a division by an empty bin width.
        return float(abs(correct.mean() - c.mean()))

    edges = np.linspace(c.min(), c.max(), bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        # The last bin is closed on the right so a row at the maximum is counted.
        in_bin = (c > lo) & (c <= hi) if lo > edges[0] else (c >= lo) & (c <= hi)
        n = int(in_bin.sum())
        if n:
            total += n / len(rows) * abs(correct[in_bin].mean() - c[in_bin].mean())
    return float(total)


def apply_temperature(probs, temperature: float):
    """Rescale probability rows to a different temperature.

    A temperature never changes which option is picked — it is monotone — but it
    is not a no-op on the *gate*. Confidence here is entropy-based, and scaling a
    row's logits by ``1 / T`` changes that row's entropy by an amount that
    depends on how peaked the row already was. Rows therefore move against each
    other in confidence order, and a fixed threshold accepts a different set.

    ``log(p) / T`` followed by a softmax is exactly what scaling the underlying
    logits by ``1 / T`` does, so a checkpoint's stored probabilities are enough;
    the logits themselves are never needed.
    """
    rows = _rows(probs)
    if temperature <= 0:
        raise ValueError("temperature must be > 0, got %r" % (temperature,))
    if temperature == 1.0:
        return np.stack(rows) if len({len(r) for r in rows}) == 1 else rows
    out = []
    for r in rows:
        z = np.log(np.clip(r, 1e-12, None)) / float(temperature)
        z -= z.max()
        e = np.exp(z)
        out.append(e / e.sum())
    return np.stack(out) if len({len(r) for r in rows}) == 1 else out


def select_temperature(probs, labels, max_error_rate: float = 0.10, grid=None):
    """Pick the temperature that covers the most decisions under an error budget.

    **This is deliberately not the temperature that minimises negative
    log-likelihood**, and the gap between them is why this function exists.
    Fitting T by NLL asks for confidence that matches accuracy *in aggregate*;
    the operating point asks for a confidence *ordering* that puts the wrong rows
    at the bottom. Those are different objectives, and NLL's optimum lands near
    the worst end of the coverage curve — measured, not argued: on one
    checkpoint at a 10% error budget, the NLL-optimal temperature released 55% of
    decisions and the one this function returns released 70%, with the argmax
    unchanged and so the accuracy identical.

    Returns ``(temperature, coverage)``. ``1.0`` means no candidate beat leaving
    the checkpoint alone, which is a real result and should be reported as one
    rather than quietly replaced by the best grid point. The grid starts below 1
    on purpose: sharpening is a legitimate choice when the error budget is loose,
    and a search that can only soften would miss it.
    """
    rows = _rows(probs)
    y = _labels(labels, len(rows))
    if grid is None:
        grid = np.concatenate([np.linspace(0.5, 1.5, 11), np.linspace(1.6, 4.0, 13)])
    active = [float(t) for t in grid]
    best_coverage, best_temperature = -1.0, 1.0
    for t in active:
        coverage, _ = coverage_at_error(apply_temperature(rows, t), y, max_error_rate)
        if coverage > best_coverage:
            best_coverage, best_temperature = coverage, t
    if 1.0 not in active:
        coverage, _ = coverage_at_error(rows, y, max_error_rate)
        if coverage >= best_coverage:
            return 1.0, coverage
    return best_temperature, max(best_coverage, 0.0)


def paired_confidence_drop(unknown_probs, control_probs):
    """How much confidence falls when the deciding evidence is removed.

    Callers build two sets that differ in exactly one way: in the first the
    sentence that decides the answer has been deleted, so the item is
    unanswerable; in the second the same item is intact. No labels are involved
    on purpose — an unanswerable item has no correct option, and scoring one
    would measure the test's construction rather than the model.

    Returns the mean confidence drop (positive means the model noticed), the
    share of pairs where confidence fell at all, and the share of unanswerable
    items the model is still at least 0.9 confident about — the rows a product
    would release as if they were fine.
    """
    u = confidence(unknown_probs)
    c = confidence(control_probs)
    if len(u) != len(c):
        raise ValueError("unknown and control sets must be paired, got %d vs %d" % (len(u), len(c)))
    if not len(u):
        return {"n_pairs": 0, "mean_drop": float("nan"), "share_lower": float("nan"),
                "unknown_at_0_9": float("nan"), "control_at_0_9": float("nan")}
    return {
        "n_pairs": int(len(u)),
        "mean_drop": float((c - u).mean()),
        "share_lower": float((u < c).mean()),
        "unknown_at_0_9": float((u >= 0.9).mean()),
        "control_at_0_9": float((c >= 0.9).mean()),
    }
