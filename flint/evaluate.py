"""Scoring a Flint checkpoint on a suite.

A single accuracy number cannot describe a decision model, because the product
does not use every answer — it gates on confidence and declines the rest. So a
report here always contains three things:

1. **Coverage at an error budget.** At most 5% of the released answers wrong,
   what share of questions can be released? That is the number a product
   actually spends, and it is the one two checkpoints differ on most.
2. **Risk at fixed coverage.** The same trade-off read the other way, so two
   models can be compared across their whole operating range instead of at one
   arbitrary threshold — the mistake that made an earlier comparison wrong.
3. **Order sensitivity.** The suite is scored twice with the options permuted
   differently. A model whose accuracy moves when the options move learned the
   layout rather than the question, and no aggregate hides that.

The report also refuses to be produced from a contaminated suite: if any record
in the suite appears in the training mixture, the run stops.
"""

from __future__ import annotations

import argparse
import json
import os

__all__ = ["per_question_rows", "evaluate_suite", "main"]


def per_question_rows(checkpoint, records):
    """Flatten a suite into one row per question, with its predictions.

    ``probs`` stays a 1-D array per row rather than a padded matrix: a score
    question over five levels and a choice over forty do not belong in the same
    rectangle, and padding one to the other's width would silently change the
    entropy confidence the gate reads for both.
    """
    from .inference import answer_index

    rows = []
    for record, answers in zip(records, checkpoint.answer(records)):
        for key, question in record["questions"].items():
            answer = answers.get(key)
            if answer is None:
                continue
            qtype = question.get("type", "choice")
            criteria = question.get("criteria")
            if qtype == "choice" and isinstance(criteria, dict):
                probs = [answer["probabilities"][k] for k in criteria]
            elif qtype == "noul":
                probs = [answer["probabilities"]["false"], answer["probabilities"]["true"]]
            else:
                probs = [answer["probabilities"][str(i)] for i in range(len(criteria))]

            index = answer_index(qtype, criteria, question.get("label"))
            rows.append({
                "record": record,
                "key": key,
                "qtype": qtype,
                "width": len(probs),
                "probs": probs,
                "label": index,
                "correct": (index is not None
                            and int(max(range(len(probs)), key=lambda i: probs[i])) == index),
            })
    return rows


def _bucket(rows):
    """Scoreable rows as ``(ragged probs, labels)``, keeping each row's own width."""
    usable = [r for r in rows if r["label"] is not None]
    if not usable:
        return None, None
    return [r["probs"] for r in usable], [r["label"] for r in usable]


def evaluate_suite(checkpoint, records, suite_name="suite", max_error_rates=(0.05, 0.10, 0.15),
                   coverages=(0.6, 0.7, 0.8, 0.9, 0.95), order_seeds=(0, 1)):
    import numpy as np

    from . import metrics

    base_rows = per_question_rows(checkpoint, records)
    matrix, labels = _bucket(base_rows)
    if matrix is None:
        return {"suite": suite_name, "questions": 0, "note": "no scoreable questions"}

    correct = np.array([r["correct"] for r in base_rows if r["label"] is not None])

    report = {
        "suite": suite_name,
        "records": len(records),
        "questions": len(base_rows),
        "scoreable": int(len(labels)),
        "accuracy": float(correct.mean()),
        "ece": metrics.expected_calibration_error(matrix, labels),
        "eceEntropyConfidence": metrics.expected_calibration_error(
            matrix, labels, conf=metrics.confidence(matrix)
        ),
        "coverageAtError": {},
        "riskAtCoverage": metrics.selective_risk_curve(matrix, labels, coverages),
        "byQtype": {},
        "byWidth": {},
    }

    for budget in max_error_rates:
        coverage, threshold = metrics.coverage_at_error(matrix, labels, budget)
        # A budget nothing can meet leaves no threshold to deploy. Report the
        # zeros rather than a NaN threshold disguised as a measurement.
        if coverage > 0.0 and threshold == threshold:
            gated_coverage, gated_error = metrics.gated_error(matrix, labels, threshold)
        else:
            gated_coverage, gated_error = 0.0, float("nan")
        report["coverageAtError"]["%.2f" % budget] = {
            "coverage": coverage,
            "threshold": None if threshold != threshold else threshold,
            "deployedCoverage": gated_coverage,
            "deployedError": gated_error,
        }

    for label, key_fn in (("byQtype", lambda r: r["qtype"]), ("byWidth", lambda r: str(r["width"]))):
        groups: dict[str, list] = {}
        for r in base_rows:
            if r["label"] is not None:
                groups.setdefault(key_fn(r), []).append(r)
        for name, group in sorted(groups.items()):
            m, y = _bucket(group)
            if m is None:
                continue
            report[label][name] = {
                "n": len(group),
                "accuracy": metrics.accuracy(m, y),
                "coverageAt10": metrics.coverage_at_error(m, y, 0.10)[0],
            }

    # Order sensitivity. The same records and the same labels, with the options
    # laid out differently. What is compared is identity, not position: an
    # answer that moves from one option to another when the layout changes is a
    # layout cue, and it is invisible in any aggregate score.
    sensitivity = {}
    baseline = _top_option_keys(checkpoint, records)
    for seed in order_seeds:
        if seed == 0:
            continue
        permuted = _top_option_keys(checkpoint, records, option_seed=seed)
        flipped = sum(1 for k, v in baseline.items() if v["top"] != permuted[k]["top"])
        total = len(baseline)
        sensitivity["seed%d" % seed] = {
            "questions": total,
            "flipped": flipped,
            "flipRate": (flipped / total) if total else None,
            "accuracyPermuted": (sum(1 for v in permuted.values() if v["correct"]) / total)
            if total else None,
            "accuracyUnpermuted": (sum(1 for v in baseline.values() if v["correct"]) / total)
            if total else None,
        }
    report["orderSensitivity"] = sensitivity
    return report


def _top_option_keys(checkpoint, records, option_seed=None):
    """``{question id: {top, correct}}`` with the answer as an option *identity*.

    Identity is the criteria key for a choice and the boolean for a noul, so the
    same dictionary key survives an option permutation and two runs can be
    compared. Comparing positions instead would report every permutation as a
    disagreement.
    """
    import numpy as np

    out = {}
    answers = checkpoint.answer(records, option_seed=option_seed)
    for r_i, (record, answer) in enumerate(zip(records, answers)):
        for key, question in record["questions"].items():
            if key not in answer:
                continue
            probs = answer[key]["probabilities"]
            top = max(probs, key=probs.get) if probs else None
            label = question.get("label")
            if question.get("type") == "choice":
                identity = top
            elif question.get("type") == "noul":
                identity = top == "true"
                label = bool(label)
            else:
                identity = int(top)
                label = int(label) if label is not None else None
            out["%d:%s" % (r_i, key)] = {"top": identity, "correct": identity == label}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--suite", required=True, help="jsonl of decision records")
    ap.add_argument("--mix", default=None,
                    help="the training mixture, if you have it: the run stops if the suite overlaps it")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from .inference import FlintCheckpoint
    from .mixture import find_overlap, load_jsonl

    records = load_jsonl(args.suite)
    if args.mix:
        overlap = find_overlap(load_jsonl(args.mix), records)
        if overlap:
            raise SystemExit(
                "refusing to evaluate: %d of %d suite records appear in the training mixture. "
                "Any number produced here would be a memorisation score."
                % (overlap, len(records))
            )

    checkpoint = FlintCheckpoint.load(args.checkpoint, device=args.device)
    report = evaluate_suite(checkpoint, records, suite_name=os.path.basename(args.suite))
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print("wrote %s" % args.out)
    else:
        print(text)


if __name__ == "__main__":
    main()
