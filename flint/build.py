"""Building a mixture from a spec file.

Separate from `flint.scenarios` on purpose. Turning a corpus into decision
records needs the corpus — a download, a licence to have read, a revision to
pin — and that work belongs to whoever has the corpus. Assembling the records
into a mixture, deduplicating, capping, tagging provenance and auditing the
result is mechanical and is what this does, offline and reproducibly.

A spec is JSON:

```json
{
  "seed": 20260925,
  "sources": [
    {"name": "banking77", "path": "data/banking77.jsonl", "weight": 1.0, "cap": 2000},
    {"name": "policy-written", "path": "data/policy.jsonl"}
  ],
  "require": ["banking77", "policy-written"]
}
```

`weight` is a sampling weight for the trainer, recorded in the report; it never
repeats rows. `cap` keeps at most that many rows, taken after a seeded shuffle.
`require` names sources whose absence must be an error — a build that silently
produced a mixture from two of its three intended sources looks exactly like a
successful one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

__all__ = ["build_from_spec", "main"]


def build_from_spec(spec: dict):
    from .mixture import Mixture, audit, load_jsonl, write_jsonl

    if not isinstance(spec, dict):
        raise ValueError("the spec must be a JSON object")
    sources = spec.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("the spec has no `sources`")

    required = list(spec.get("require") or [])
    names = [s.get("name") for s in sources]
    missing_required = [n for n in required if n not in names]
    if missing_required:
        raise ValueError("the spec requires %s but does not define it/them"
                         % ", ".join(missing_required))

    mixture = Mixture(seed=int(spec.get("seed", 0)))
    per_source = {}
    for source in sources:
        name = source.get("name")
        path = source.get("path")
        if not name or not path:
            raise ValueError("every source needs a `name` and a `path`, got %r" % (source,))
        if not os.path.exists(path):
            raise FileNotFoundError("source %r points at %s, which does not exist" % (name, path))
        rows = load_jsonl(path)
        mixture.add(name, rows, weight=float(source.get("weight", 1.0)),
                    cap=source.get("cap"))
        per_source[name] = {"path": path, "rows": len(rows)}

    rows, report = mixture.build()
    report["requested"] = per_source
    report["audit"] = audit(rows)
    return rows, report


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True, help="JSON spec describing the sources")
    ap.add_argument("--out", required=True, help="mixture to write (jsonl)")
    ap.add_argument("--report", default=None,
                    help="where to write the build report (default: <out>.report.json)")
    args = ap.parse_args()

    with open(args.spec, encoding="utf-8") as f:
        spec = json.load(f)

    rows, report = build_from_spec(spec)
    from .mixture import write_jsonl

    written = write_jsonl(args.out, rows)
    report_path = args.report or (args.out + ".report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print("wrote %d rows to %s" % (written, args.out))
    print("report %s" % report_path)
    for name, stats in report["sources"].items():
        print("  %-24s %6d in -> %6d distinct -> %6d emitted (weight %.2f, cap %s)"
              % (name, stats["inputRows"], stats["distinctRows"], stats["emitted"],
                 stats["weight"], stats["cap"]))
        if stats["internalDuplicates"]:
            print("  %-24s %d duplicate rows inside the source were dropped"
                  % ("", stats["internalDuplicates"]))
    duplicate_rows = report["audit"]["duplicateRows"]
    if duplicate_rows:
        print("refusing: %d duplicate rows survived the build" % duplicate_rows, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
