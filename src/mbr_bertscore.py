#!/usr/bin/env python3
"""MBR with a neural utility (review4 item 9): pick the candidate with the
highest mean BERTScore-F1 (roberta-large) against the other candidates.
800-item subset, same candidate pool. Writes {tag}_mbrbs_pred.jsonl (+evals).
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mimic_mlf")
    ap.add_argument("--data_root", default="/home/deployer/otlora/data/processed")
    ap.add_argument("--cands_tag", default="sft42_k8")
    ap.add_argument("--model", default="roberta-large")
    args = ap.parse_args()

    from bert_score import BERTScorer

    scorer = BERTScorer(model_type=args.model, lang="en", device="cuda:0",
                        rescale_with_baseline=False, batch_size=64)
    base = os.path.join(args.data_root, args.dataset)
    rows = [json.loads(l) for l in open(os.path.join(base, f"{args.cands_tag}_cands.jsonl"))]

    picks = []
    for s in range(0, len(rows), 8):
        chunk = rows[s:s + 8]
        for r in chunk:
            cands = [r["greedy"]] + r["cands"]
            k = len(cands)
            cs, rs = [], []
            for i in range(k):
                for j in range(k):
                    if i != j:
                        cs.append(cands[i])
                        rs.append(cands[j])
            P, R, F = scorer.score(cs, rs)
            sums = [0.0] * k
            ptr = 0
            for i in range(k):
                for j in range(k):
                    if i != j:
                        sums[i] += F[ptr].item()
                        ptr += 1
            best = max(range(k), key=lambda i: sums[i])
            picks.append(cands[best])
        print(f"scored {s + len(chunk)}/{len(rows)}", flush=True)

    p = os.path.join(base, f"{args.cands_tag}_mbrbs_pred.jsonl")
    with open(p, "w") as f:
        for r, pr in zip(rows, picks):
            f.write(json.dumps({"id": r["id"], "gt": r["gt"], "pred": pr}) + "\n")

    from eval_metrics import compute_nlg
    from chexpert_label import clinical_f1_from_pairs
    compute_nlg(p, p.replace(".jsonl", "_nlg.json"))
    clinical_f1_from_pairs([json.loads(l) for l in open(p)], p.replace(".jsonl", "_ce2.json"))
    n, c = json.load(open(p.replace(".jsonl", "_nlg.json"))), \
        json.load(open(p.replace(".jsonl", "_ce2.json")))
    print(f"MBRBS B1={n['BLEU-1']:.3f} CEma={c['F1_macro']:.3f} uniq={n['unique_reports']}")


if __name__ == "__main__":
    main()
