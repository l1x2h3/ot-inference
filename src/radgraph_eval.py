#!/usr/bin/env python3
"""F1-RadGraph (Chen et al. 2023) on the full MIMIC-MLF test set for all
selection strategies - the independent clinical evaluator requested by review.

Uses the radgraph-xl checkpoint via the `radgraph` package (StanfordAIMI/
RRG_scorers). reward_level="all" = micro-F1 over all entity/relation
annotations, the standard F1-RadGraph.
"""
import json
import os
import sys

STRATS = ["greedy", "random", "mbr", "ot", "ot_len", "logprob", "hybrid"]
D = "data/processed/mimic_mlf"
OUT = "results/radgraph_full.json"


def main():
    from radgraph import F1RadGraph
    scorer = F1RadGraph(reward_level="all", cuda=0, batch_size=16)
    res = {}
    if os.path.exists(OUT):
        res = json.load(open(OUT))
    for s in STRATS:
        if s in res:
            print("SKIP", s, flush=True)
            continue
        rows = [json.loads(l) for l in open(f"{D}/sft42_full_{s}_pred.jsonl")]
        hyps = [r["pred"].strip() or "." for r in rows]
        refs = [r["gt"].strip() or "." for r in rows]
        (f1, prec, rec), per_item, _, _ = scorer(hyps=hyps, refs=refs)
        res[s] = {"F1": float(f1), "P": float(prec), "R": float(rec)}
        json.dump(res, open(OUT, "w"), indent=1)
        print(f"DONE {s}: F1-RadGraph={f1:.4f} P={prec:.4f} R={rec:.4f}", flush=True)
    print("RADGRAPH_ALL_DONE")


if __name__ == "__main__":
    main()
