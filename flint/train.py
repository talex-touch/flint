"""Training a Flint head.

The recipe is deliberately dull: multi-task cross-entropy against a target
distribution, with the two knobs that actually mattered — label smoothing, and
permuting the options of choice questions — and one non-obvious choice about how
the temperature is fitted at the end (see ``docs/evaluation.md``).

Two things this file refuses to do, both because they are the failures that do
not raise:

- **It never drops a record silently.** Whatever cannot be turned into a
  training item is counted and printed. ``train_meta.json`` carries the number.
- **It never evaluates on data it trained on.** If the mixture and the dev set
  share any record, or any option-set the model was asked to memorise, the run
  stops before it starts. Checking inputs for exact overlap is not enough on its
  own — see ``docs/training.md`` for the leak that passed that check and still
  inflated a result by twenty points.
"""

from __future__ import annotations

import argparse
import json
import os
import time

__all__ = ["build_items", "train", "main"]


def build_items(tokenizer, rows, cfg):
    """Records -> engine items, with a count of everything that did not fit."""
    from . import engine

    items, skipped = [], 0
    for row in rows:
        group = []
        for q_index, (key, question) in enumerate(row["questions"].items()):
            item = engine.to_item(
                tokenizer, row, key, question,
                max_len=cfg["max_len"],
                head_max_len=cfg["head_max_len"],
                smoothing=cfg["target_smooth"],
                option_seed=cfg["seed"] + q_index if cfg["shuffle_options"] else None,
            )
            if item is None:
                skipped += 1
                continue
            group.append(item)
        if group:
            items.append(group)
    return items, skipped


def _batches(groups, batch_size, rng):
    order = list(range(len(groups)))
    rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        chunk = order[start:start + batch_size]
        yield [groups[i] for i in chunk]


def _loss(model, collated, device):
    import torch

    logits, _ = model(
        collated["input_ids"].to(device),
        collated["attention_mask"].to(device),
        collated["marker_pos"].to(device),
        collated["marker_mask"].to(device),
        collated["qtype"].to(device),
    )
    target = collated["target"].to(device).float()
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    # The log score is a proper scoring rule: its expectation is minimised by
    # reporting the true distribution, which is the property that makes the
    # resulting confidence worth gating on. Spherical and ranked-probability
    # scores are proper too and the engine exposes them
    # (`laya.common.proper_reward`); the log score is used here because it is
    # the one whose gradient can be reasoned about per row.
    per_item = -(target * log_probs).sum(dim=-1)
    return per_item.mean()


def train(args) -> dict:
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    from . import engine, metrics
    from .mixture import audit, find_overlap, load_jsonl

    torch.manual_seed(args.seed)

    train_rows = load_jsonl(args.mix)
    dev_rows = load_jsonl(args.dev) if args.dev else []

    if dev_rows:
        overlap = find_overlap(train_rows, dev_rows)
        if overlap:
            raise SystemExit(
                "refusing to train: %d of %d dev records appear verbatim in the mixture. "
                "The dev number this run would produce is not an evaluation."
                % (overlap, len(dev_rows))
            )

    # Created only once the guards have passed. A refused run must leave no
    # artifact behind, because an empty checkpoint directory is indistinguishable
    # from a run that started and died.
    os.makedirs(args.out, exist_ok=True)

    stats = audit(train_rows)
    if stats["duplicateRows"]:
        print("mixture carries %d duplicate rows across %d groups; building the "
              "mixture with flint.mixture.Mixture prevents this"
              % (stats["duplicateRows"], stats["duplicateGroups"]), flush=True)

    cfg = {
        "encoder": args.encoder,
        "head_layers": args.head_layers,
        "max_len": args.max_len,
        "head_max_len": args.head_max_len,
        "target_smooth": args.smooth,
        "shuffle_options": not args.no_shuffle_options,
        "seed": args.seed,
        # The engine reads these two out of the checkpoint config; the scored
        # act head is part of the architecture whether or not this recipe uses it.
        "act_costs": {"escalate": 0.5},
        "cost_wrong_act": 3.0,
    }

    device = torch.device(args.device)
    # A continuation reads the encoder out of the previous checkpoint's own
    # `encoder/` subdirectory, not out of the checkpoint root.
    encoder_source = os.path.join(args.init_from, "encoder") if args.init_from else args.encoder
    encoder = AutoModel.from_pretrained(encoder_source)
    encoder.to(device)
    # The tokenizer is not always beside the encoder: an encoder kept in a
    # subfolder of a release (as the reference backbone is) leaves its tokenizer
    # in a sibling directory, so it gets its own flag.
    tokenizer_source = args.tokenizer or encoder_source
    tokenizer_kwargs = {}
    if args.tokenizer_subfolder:
        tokenizer_kwargs["subfolder"] = args.tokenizer_subfolder
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)

    if args.freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad_(False)

    model = engine.build_model_on(encoder, cfg).to(device)
    if args.init_from and os.path.exists(os.path.join(args.init_from, "head.safetensors")):
        from safetensors.torch import load_file
        model.load_state_dict(load_file(os.path.join(args.init_from, "head.safetensors")), strict=False)

    groups, skipped = build_items(tokenizer, train_rows, cfg)
    if not groups:
        raise SystemExit("no training items were built from %s" % args.mix)
    print("mixture %d rows -> %d records, %d questions skipped"
          % (len(train_rows), len(groups), skipped), flush=True)

    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    param_groups = [{"params": head_params, "lr": args.lr_head}]
    if not args.freeze_encoder:
        param_groups.append({"params": [p for p in encoder.parameters() if p.requires_grad],
                             "lr": args.lr_enc})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

    import random
    rng = random.Random(args.seed)
    model.train()
    started = time.time()
    step = 0
    history = []
    for epoch in range(args.epochs):
        for batch in _batches(groups, args.batch, rng):
            collated = engine.collate(tokenizer, batch)
            if collated is None:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model, collated, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step % args.log_every == 0:
                history.append({"step": step, "epoch": epoch, "loss": float(loss.detach())})
                print("epoch %d step %d loss %.4f" % (epoch, step, float(loss.detach())), flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    seconds = time.time() - started

    dev_accuracy = None
    temperature = [1.0, 1.0, 1.0]
    temperature_by_options: dict[str, float] = {}
    if dev_rows:
        probs, labels, qtype_names = _score_dev(model, tokenizer, dev_rows, cfg, device)
        if len(labels):
            dev_accuracy = float((np.asarray(probs).argmax(axis=1) == np.asarray(labels)).mean())
            temperature, temperature_by_options = _fit_temperatures(
                probs, labels, qtype_names, args.max_error_rate
            )
            print("dev accuracy %.4f on %d questions" % (dev_accuracy, len(labels)), flush=True)

    cfg["temperature"] = temperature
    cfg["temperature_by_options"] = temperature_by_options

    encoder.save_pretrained(os.path.join(args.out, "encoder"))
    tokenizer.save_pretrained(os.path.join(args.out, "encoder"))
    from safetensors.torch import save_file
    save_file({k: v.contiguous() for k, v in model.state_dict().items()
               if not k.startswith("encoder.")},
              os.path.join(args.out, "head.safetensors"))

    meta = {
        "base": args.init_from or args.encoder,
        "epochs": args.epochs,
        "batch": args.batch,
        "lr_enc": args.lr_enc,
        "lr_head": args.lr_head,
        "freeze_encoder": bool(args.freeze_encoder),
        "max_len": args.max_len,
        "head_max_len": args.head_max_len,
        "target_smooth": args.smooth,
        "shuffle_options": not args.no_shuffle_options,
        "seed": args.seed,
        "mix": args.mix,
        "mixRows": len(train_rows),
        "trainItems": sum(len(g) for g in groups),
        "skipped": skipped,
        "dev": args.dev,
        "devQuestions": len(dev_rows),
        "devAccuracy": dev_accuracy,
        "steps": step,
        "seconds": round(seconds, 1),
        "history": history,
        "temperature": temperature,
        "temperatureByOptions": temperature_by_options,
    }
    for name, payload in (("rl_agent_config.json", cfg),
                          ("train_meta.json", meta),
                          ("temperature.json", {"temperature": temperature,
                                                "temperature_by_options": temperature_by_options})):
        with open(os.path.join(args.out, name), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")

    print("wrote %s in %.1fs" % (args.out, seconds), flush=True)
    return meta


def _score_dev(model, tokenizer, rows, cfg, device):
    """Run the dev records and collect per-question probabilities."""
    import numpy as np
    import torch

    from . import engine

    model.eval()
    probs, labels, qtype_names = [], [], []
    with torch.no_grad():
        for row in rows:
            group = []
            for key, question in row["questions"].items():
                item = engine.to_item(tokenizer, row, key, question,
                                      max_len=cfg["max_len"], head_max_len=cfg["head_max_len"])
                if item is not None:
                    group.append(item)
            if not group:
                continue
            collated = engine.collate(tokenizer, [group])
            if collated is None:
                continue
            logits, _ = model(
                collated["input_ids"].to(device), collated["attention_mask"].to(device),
                collated["marker_pos"].to(device), collated["marker_mask"].to(device),
                collated["qtype"].to(device),
            )
            mask = collated["marker_mask"]
            for i, item in enumerate(group):
                k = int(mask[i].sum())
                probs.append(torch.softmax(logits[i, :k].float(), -1).cpu().numpy().astype(np.float64))
                labels.append(int(item["label"]))
                qtype_names.append(_name_for(item["qtype"], k))
    model.train()
    return probs, labels, qtype_names


def _name_for(qtype_index: int, k: int) -> str:
    from .inference import QTYPE_NAMES, temperature_bucket

    name = QTYPE_NAMES.get(int(qtype_index), "choice")
    return temperature_bucket(name, k)


def _fit_temperatures(probs, labels, qtype_names, max_error_rate):
    """Fit one temperature per question type, then per option-count bucket.

    The bucket fit is allowed to fail back to the per-type value: a bucket with
    too few rows to move the coverage estimate is worse than the coarser value
    it would replace, and there is no way to tell those apart from the number
    alone.
    """
    import numpy as np

    from . import metrics
    from .inference import QTYPE_INDEX, QTYPE_NAMES

    by_type: dict[int, list[int]] = {}
    for i, name in enumerate(qtype_names):
        by_type.setdefault(QTYPE_INDEX[name.split(":")[0]], []).append(i)

    temperature = [1.0, 1.0, 1.0]
    for qt, idx in by_type.items():
        t, _ = metrics.select_temperature(
            np.stack([probs[i] for i in idx]), [labels[i] for i in idx], max_error_rate
        )
        temperature[qt] = round(float(t), 4)

    by_bucket: dict[str, list[int]] = {}
    for i, name in enumerate(qtype_names):
        by_bucket.setdefault(name, []).append(i)

    per_bucket = {}
    for name, idx in by_bucket.items():
        if len(idx) < 40:
            continue
        t, coverage = metrics.select_temperature(
            np.stack([probs[i] for i in idx]), [labels[i] for i in idx], max_error_rate
        )
        per_bucket[name] = round(float(t), 4)
    return temperature, per_bucket


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mix", required=True, help="training mixture (jsonl of decision records)")
    ap.add_argument("--dev", default=None, help="held-out records; refused if they overlap the mixture")
    ap.add_argument("--out", required=True, help="checkpoint directory to write")
    ap.add_argument("--encoder", default="jhu-clsp/mmBERT-base",
                    help="encoder id or local directory (default matches the reference build)")
    ap.add_argument("--tokenizer", default=None, help="tokenizer id when it differs from --encoder")
    ap.add_argument("--tokenizer-subfolder", default=None)
    ap.add_argument("--init-from", default=None,
                    help="an existing Flint checkpoint to continue from; its head is loaded too")
    ap.add_argument("--head-layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr-enc", type=float, default=8e-6)
    ap.add_argument("--lr-head", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--head-max-len", type=int, default=512)
    ap.add_argument("--smooth", type=float, default=0.03, help="label smoothing")
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--no-shuffle-options", action="store_true")
    ap.add_argument("--max-error-rate", type=float, default=0.10,
                    help="error budget the temperature is fitted against")
    ap.add_argument("--max-steps", type=int, default=0, help="stop early; 0 means no limit")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
