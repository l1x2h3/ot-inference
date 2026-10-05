#!/usr/bin/env python3
"""Extra F1-RadGraph runs: pure-length baselines (full test) and the
uniform-marginal variant (subset) -- independent-evaluator coverage for
review4 items 7/8."""
import json
import os

D = "/home/deployer/otlora/data/processed/mimic_mlf"
OUT = "/home/deployer/otlora/results/radgraph_full.json"

JOBS = [
    ("longest", f"{D}/sft42_full_longest_pred.jsonl"),
    ("medlen", f"{D}/sft42_full_medlen_pred.jsonl"),
    ("var_unif", f"{D}/sft42_k8_var_unif_pred.jsonl"),
    ("biclip_note", None),  # placeholder, real medclip file lands later
]


def main():
    from radgraph import F1RadGraph
    scorer = F1RadGraph(reward_level="all", cuda=0, batch_size=16)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for key, path in JOBS:
        if path is None or not os.path.exists(path) or key in res:
            continue
        rows = [json.loads(l) for l in open(path)]
        hyps = [r["pred"].strip() or "." for r in rows]
        refs = [r["gt"].strip() or "." for r in rows]
        (f1, prec, rec), per_item, _, _ = scorer(hyps=hyps, refs=refs)
        res[key] = {"F1": float(f1), "P": float(prec), "R": float(rec),
                    "n": len(rows)}
        json.dump(res, open(OUT, "w"), indent=1)
        print(f"DONE {key}: F1={f1:.4f} P={prec:.4f} R={rec:.4f}", flush=True)
    print("RADGRAPH_EXTRA_DONE")


if __name__ == "__main__":
    main()
