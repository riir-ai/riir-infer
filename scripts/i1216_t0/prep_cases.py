#!/usr/bin/env python3
"""Issue 1216 T0 prep (plan 634) — practice cases.jsonl + gold sidecar from the
typed_decisions TRAIN split, builder parity with reflex `build_typed_decisions`.

Emits JSONL rows the existing `dump_encoder_states` bin consumes verbatim:
  {"id": "<casepos>:<qid>:<gold_idx>", "state": <state>, "question": <qdef>}
The gold index rides the id (last `:` field) so the analysis joins picks to
gold through the LENC cache's own id column — zero extraction code changes.

Splits (pre-registered, plan 634): fit = case rows [0, 840), gate = [840, 1200)
in file order. `--slice fit|gate|fid` selects the emission slice.
"""
import argparse
import glob
import json
import os

SPLIT_FIT_END = 840  # first 70% of the 1200 train rows, in file order

TRAIN_GLOB = "/Users/katopz/git/riir-reflex/.raw/datasets/typed_decisions/train-*.json"


def load_rows(max_rows):
    rows = []
    for path in sorted(glob.glob(TRAIN_GLOB)):
        with open(path) as f:
            data = json.load(f)
        for page_row in data.get("rows", []):
            rows.append(page_row["row"])
            if max_rows and len(rows) >= max_rows:
                return rows
    return rows


def parse_or_raw(v):
    """reflex parse_json_string_or_value: a JSON string parses, else the raw value."""
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return v
    return v


def qdef_kind(qdef):
    t = qdef.get("type")
    return t if t in ("choice", "score", "noul") else None


def gold_answer(kind, qdef, g):
    """The reflex gold_answer read: returns the gold option INDEX or None."""
    probs = g.get("probabilities") or {}
    if kind == "choice":
        crit = qdef.get("criteria")
        if isinstance(crit, list):
            keys = [k if isinstance(k, str) else json.dumps(k) for k in crit]
        elif isinstance(crit, dict):
            keys = list(crit.keys())
        else:
            return None
        label = g.get("label")
        if not isinstance(label, str) or label not in keys:
            return None
        return keys.index(label)
    if kind == "score":
        crit = qdef.get("criteria")
        if not isinstance(crit, list):
            return None
        label = g.get("label")
        try:
            idx = int(str(label).strip())
        except (TypeError, ValueError):
            return None
        return idx if 0 <= idx < len(crit) else None
    # noul
    label = str(g.get("label", "")).lower()
    if label not in ("false", "true"):
        return None
    return 1 if label == "true" else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", choices=["fit", "gate", "fid"], required=True,
                    help="fit=[0,840) gate=[840,1200) fid=[0,100) (fit prefix)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="cap questions emitted (0=all)")
    args = ap.parse_args()

    rows = load_rows(1200)
    if len(rows) != 1200:
        raise SystemExit(f"expected 1200 train rows, got {len(rows)}")
    lo, hi = {"fit": (0, SPLIT_FIT_END), "gate": (SPLIT_FIT_END, 1200),
              "fid": (0, 100)}[args.slice]

    # QUESTIONS_PER_CASE cap mirrors nothing in the reflex builder (it takes
    # all of a row's resolvable questions) — the cap here is only a smoke aid.
    n_rows = n_q = 0
    with open(args.out, "w") as out:
        for pos in range(lo, hi):
            row = rows[pos]
            workflow = row.get("workflow")
            if not isinstance(workflow, str):
                continue
            state = parse_or_raw(row.get("state"))
            qmap = parse_or_raw(row.get("questions"))
            gmap = parse_or_raw(row.get("gold"))
            if not isinstance(qmap, dict) or not isinstance(gmap, dict):
                continue
            emitted_any = False
            for qid, qdef in qmap.items():
                g = gmap.get(qid)
                if g is None:
                    continue
                kind = qdef_kind(qdef)
                if kind is None:
                    continue
                ins = qdef.get("instructions")
                if not isinstance(ins, str):
                    continue
                gold = gold_answer(kind, qdef, g)
                if gold is None:
                    continue
                # the agent wire form: to_internal reads type/instructions/criteria
                qdef_wire = {"type": kind, "instructions": ins,
                             "criteria": qdef.get("criteria")}
                rid = f"{pos}:{qid}:{gold}"
                out.write(json.dumps({"id": rid, "state": state,
                                      "question": qdef_wire}, ensure_ascii=False) + "\n")
                n_q += 1
                emitted_any = True
                if args.limit and n_q >= args.limit:
                    break
            if emitted_any:
                n_rows += 1
            if args.limit and n_q >= args.limit:
                break
    size = os.path.getsize(args.out)
    print(f"{args.slice}: rows={n_rows} questions={n_q} -> {args.out} ({size} bytes)")


if __name__ == "__main__":
    main()
