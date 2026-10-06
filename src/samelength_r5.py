#!/usr/bin/env python3
"""Same-length contrast for the corrected scorers (rev5 picks)."""
import json
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "/home/deployer/otlora/src")
from chexpert_label import label_report, CATEGORIES

D = "/home/deployer/otlora/data/processed/mimic_mlf"
rows = [json.loads(l) for l in open(f"{D}/sft42_k8_cands.jsonl")]


def peritem1(h, g):
    ch, cg = Counter(h.lower().split()), Counter(g.lower().split())
    return sum((ch & cg).values()), sum(ch.values()), sum(cg.values())


def bleu1(M, C, R):
    bp = 1.0 if C > R else np.exp(1 - R / max(C, 1e-9))
    return bp * M / max(C, 1e-9)


out = {}
for strat in ["sal_sh_all", "hyb_sh_all", "sal_all"]:
    preds = [json.loads(l) for l in open(f"{D}/sft42_k8_r5_{strat}_pred.jsonl")]
    A, pa, pb = [], [], []
    gl = [label_report(r["gt"]) for r in rows]
    picks_a, picks_b = [], []
    for i, r in enumerate(rows):
        pool = [r["greedy"]] + r["cands"]
        pick = preds[i]["pred"]
        if pick not in pool:
            continue
        others = [c for c in pool if c != pick]
        if not others:
            continue
        w = len(pick.split())
        near = min(others, key=lambda c: abs(len(c.split()) - w))
        A.append(peritem1(pick, r["gt"]) + peritem1(near, r["gt"]))
        picks_a.append(pick); picks_b.append(near)
    A = np.array(A)
    K = len(CATEGORIES)
    g = np.array([[1 if gl[i][c] == 1 else 0 for c in CATEGORIES] for i in range(len(picks_a))])

    def labs(texts):
        ppos = np.array([[1 if label_report(t)[c] == 1 else 0 for c in CATEGORIES] for t in texts])
        am = np.array([[1 if (label_report(t)[c] != 0 or gl[i][c] != 0) else 0
                        for c in CATEGORIES] for i, t in enumerate(texts)])
        return ppos, am

    pa, ama = labs(picks_a)
    pb, amb = labs(picks_b)

    def macro_i(w, ppos, am):
        pw, gw = ppos[w].astype(bool), g[w].astype(bool)
        tp = (pw & gw).sum(0); fp = (pw & ~gw).sum(0); fn = (~pw & gw).sum(0)
        pr = tp / np.maximum(tp + fp, 1); rc = tp / np.maximum(tp + fn, 1)
        f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-9)
        return f1[am[w].sum(0) > 0].mean()

    rng = np.random.default_rng(0)
    n = len(A)
    W = rng.integers(0, n, size=(2000, n))
    d_b1 = np.array([bleu1(A[w, 0].sum(), A[w, 1].sum(), A[w, 2].sum())
                     - bleu1(A[w, 3].sum(), A[w, 4].sum(), A[w, 5].sum()) for w in W])
    d_ce = np.array([macro_i(w, pa, ama) - macro_i(w, pb, amb) for w in W])
    out[strat] = {"n": n, "dB1": round(float(d_b1.mean()), 5),
                  "ci": [round(float(np.percentile(d_b1, 2.5)), 5),
                         round(float(np.percentile(d_b1, 97.5)), 5)],
                  "dCEma": round(float(d_ce.mean()), 5),
                  "ci_ce": [round(float(np.percentile(d_ce, 2.5)), 5),
                             round(float(np.percentile(d_ce, 97.5)), 5)]}
    print(strat, out[strat], flush=True)
json.dump(out, open("/home/deployer/otlora/results/samelength_r5.json", "w"), indent=1)
