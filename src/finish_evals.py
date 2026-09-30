#!/usr/bin/env python3
"""Finish all pending NLG + clinical evals for selection experiments.
Runs compute_nlg + clinical_f1_from_pairs wherever *_pred.jsonl exists but
the corresponding *_pred_nlg.json / *_pred_ce2.json is missing."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from eval_metrics import compute_nlg  # noqa: E402
from chexpert_label import clinical_f1_from_pairs  # noqa: E402
from rerank_baselines import mbr_pick  # noqa: E402

ROOT = "data/processed"
JOBS = [
    ("mimic_mlf", "sft42_full", ["greedy", "random", "ot", "ot_len", "logprob", "hybrid"]),
    ("mimic_mlf", "sfts1337_k8", ["greedy", "random", "ot", "ot_len", "logprob", "hybrid"]),
    ("mimic_mlf", "sfts9233_k8", ["greedy", "random", "ot", "ot_len", "logprob", "hybrid"]),
    ("rrg_test", "rrg_sft42", ["greedy", "random", "ot", "ot_len", "logprob", "hybrid"]),
    ("iu", "iu_sft42", ["greedy", "random", "ot", "ot_len", "logprob", "hybrid"]),
]


def main():
    for ds, tag, strats in JOBS:
        d = os.path.join(ROOT, ds)
        for s in strats:
            pred = os.path.join(d, f"{tag}_{s}_pred.jsonl")
            nlg = os.path.join(d, f"{tag}_{s}_pred_nlg.json")
            ce2 = os.path.join(d, f"{tag}_{s}_pred_ce2.json")
            if not os.path.exists(pred):
                print("MISSING PRED", ds, tag, s)
                continue
            if not os.path.exists(nlg):
                compute_nlg(pred, nlg)
                print("NLG", ds, tag, s, flush=True)
            if not os.path.exists(ce2):
                rows = [json.loads(l) for l in open(pred)]
                clinical_f1_from_pairs(rows, ce2)
                print("CE2", ds, tag, s, flush=True)
            n, c = json.load(open(nlg)), json.load(open(ce2))
            print(f"DONE {ds}/{tag}/{s}: B1={n['BLEU-1']:.3f} CEma={c['F1_macro']:.3f} "
                  f"uniq={n['unique_reports']}", flush=True)

    # MBR control on the full test set (CPU, from cands)
    d = os.path.join(ROOT, "mimic_mlf")
    pred = os.path.join(d, "sft42_full_mbr_pred.jsonl")
    if not os.path.exists(pred):
        rows = [json.loads(l) for l in open(os.path.join(d, "sft42_full_cands.jsonl"))]
        with open(pred, "w") as f:
            for r in rows:
                cands = [r["greedy"]] + r["cands"]
                f.write(json.dumps({"id": r["id"], "gt": r["gt"],
                                    "pred": cands[mbr_pick(cands)]}) + "\n")
        compute_nlg(pred, pred.replace(".jsonl", "_nlg.json"))
        clinical_f1_from_pairs([json.loads(l) for l in open(pred)],
                               pred.replace(".jsonl", "_ce2.json"))
        n, c = json.load(open(pred.replace(".jsonl", "_nlg.json"))), \
            json.load(open(pred.replace(".jsonl", "_ce2.json")))
        print(f"DONE full/mbr: B1={n['BLEU-1']:.3f} CEma={c['F1_macro']:.3f} "
              f"uniq={n['unique_reports']}")
    print("FINISH_EVALS_DONE")


if __name__ == "__main__":
    main()
