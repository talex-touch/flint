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

A fourth measurement cannot be made on a labelled suite at all, because the item
it needs has no label: delete the sentence that decides an answer, keep everything
else, and see whether confidence falls. That is `--abstain`, and it is the only
one of the four that separates a model reading its input from one guessing on
shape. Nothing here reads a label when that mode is on.

The report also refuses to be produced from a contaminated suite: if any record
in the suite appears in the training mixture, the run stops.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

__all__ = ["per_question_rows", "evaluate_suite", "abstention_report", "main"]


def _probs_for(question, answer):
    """One question's distribution, in option order, out of a served answer.

    Shared by the scored report and the abstention report on purpose: if the two
    disagreed about what a model's distribution is, the numbers they produce
    would not be about the same model.
    """
    qtype = question.get("type", "choice")
    criteria = question.get("criteria")
    if qtype == "choice" and isinstance(criteria, dict):
        return [answer["probabilities"][k] for k in criteria]
    if qtype == "noul":
        return [answer["probabilities"]["false"], answer["probabilities"]["true"]]
    return [answer["probabilities"][str(i)] for i in range(len(criteria))]


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
            probs = _probs_for(question, answer)
            qtype = question.get("type", "choice")
            criteria = question.get("criteria")

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


def abstention_report(checkpoint, records, mix_records=None):
    """Paired evidence deletion: does confidence fall when the answer is removed?

    Accuracy cannot answer this question, and scoring an unanswerable item answers
    a different one. So no label is read here: what is compared is the confidence
    of two records that differ in exactly one way — in one the sentence that
    decides the answer has been deleted, in the other it is intact. A model that
    never read the evidence shows no drop between them.

    The suite carries the pairing in ``_meta``:

    ``_meta.control_id``  on the unanswerable record, naming the intact partner's
    ``_meta.id``

    ``_meta.family``      the question family, used to split the result into
                          families the mixture contains and families it does not

    That split is the point. On the reference build the drop was clear on the
    families the mixture contained and *not shown* on the families it did not,
    and a pooled number would have reported the first half as the model's general
    behaviour. The second half is twenty pairs: read it as "not demonstrated
    here", not as an effect in the opposite direction.

    One question per record is required rather than convenient: the pairing is a
    property of the record, and averaging several questions into one confidence
    would make the drop a number about the suite's shape.
    """
    from . import metrics

    sizes = sorted({len(r.get("questions") or {}) for r in records})
    if sizes != [1]:
        raise SystemExit(
            "an abstention suite must hold exactly one question per record, found %s; "
            "the pairing is a property of the record" % (sizes or "no questions"))

    meta = [r.get("_meta") or {} for r in records]
    missing = [i for i, m in enumerate(meta) if m.get("id") is None]
    if missing:
        raise SystemExit(
            "%d of %d records carry no _meta.id, so nothing can be paired; the first is "
            "record %d" % (len(missing), len(records), missing[0]))
    by_id = {m["id"]: i for i, m in enumerate(meta)}

    answers = checkpoint.answer(records)
    # The metric takes the distributions, not a confidence number per record: a
    # choice over five options and a yes/no do not sit on the same confidence
    # scale, and entropy is normalised by each row's own width. Reducing to one
    # number here first would silently score every row at whichever width the
    # caller assumed.
    probs_by_record = []
    for record, answer in zip(records, answers):
        key = next(iter(record["questions"]))
        probs_by_record.append(
            _probs_for(record["questions"][key], answer[key]) if key in answer else None)

    unknown, control, families = [], [], []
    unscorable = []
    for i, m in enumerate(meta):
        control_id = m.get("control_id")
        if not control_id:
            continue
        j = by_id.get(control_id)
        if j is None:
            raise SystemExit(
                "record %d names control %r, which is not in this suite; a partial pairing "
                "would report a drop over a subset that is not the one you think"
                % (i, control_id))
        if probs_by_record[i] is None or probs_by_record[j] is None:
            unscorable.append((i, j))
            continue
        unknown.append(probs_by_record[i])
        control.append(probs_by_record[j])
        families.append(m.get("family"))

    # A pair with no distribution on one side is the same failure as a dangling
    # `control_id`, and skipping it quietly would be worse: the pair count would
    # simply shrink, so a suite the model could not answer would read as a smaller
    # measurement rather than a broken one.
    if unscorable:
        raise SystemExit(
            "%d of %d pairs have no distribution on one side and cannot be scored "
            "(first: %r, paired with %r). Take those records out of the suite, or find out "
            "why the model returned nothing for them; dropping them here would report a "
            "drop over a subset smaller than the suite without saying so."
            % (len(unscorable), len(unscorable) + len(unknown),
               meta[unscorable[0][0]].get("id"), meta[unscorable[0][1]].get("id")))

    report = {"n_pairs": len(unknown), "all": metrics.paired_confidence_drop(unknown, control)}

    if mix_records is None:
        # Without the mixture there is no way to tell a family the model was
        # trained on from one it was not, and that is the split that decides
        # whether a drop means anything.
        report["familiesInMixture"] = None
        report["familiesOutsideMixture"] = None
        report["note"] = ("pass --mix to split by whether the family was in the training "
                          "mixture; without it a drop cannot be told from memorisation")
        return report

    seen = {(m.get("_meta") or {}).get("family") for m in mix_records}
    seen.discard(None)
    inside = [k for k, f in enumerate(families) if f in seen]
    outside = [k for k, f in enumerate(families) if f not in seen]
    report["familiesInMixture"] = metrics.paired_confidence_drop(
        [unknown[k] for k in inside], [control[k] for k in inside])
    report["familiesOutsideMixture"] = metrics.paired_confidence_drop(
        [unknown[k] for k in outside], [control[k] for k in outside])
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
    ap.add_argument("--abstain", action="store_true",
                    help="the suite is paired evidence deletion, not a scored suite: "
                         "report whether confidence falls when the answer is removed")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from .inference import FlintCheckpoint
    from .mixture import find_overlap, load_jsonl

    records = load_jsonl(args.suite)
    mix_records = load_jsonl(args.mix) if args.mix else None
    if mix_records is not None:
        overlap = find_overlap(mix_records, records)
        if overlap:
            raise SystemExit(
                "refusing to evaluate: %d of %d suite records appear in the training mixture. "
                "Any number produced here would be a memorisation score."
                % (overlap, len(records))
            )

    checkpoint = FlintCheckpoint.load(args.checkpoint, device=args.device)
    if args.abstain:
        if mix_records is None:
            print("warning: no --mix, so the drop cannot be split into families the model "
                  "was trained on and families it was not", file=sys.stderr)
        report = abstention_report(checkpoint, records, mix_records=mix_records)
    else:
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
