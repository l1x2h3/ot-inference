#!/usr/bin/env python3
"""Paired bootstrap, vectorized (review4 item 6). Same metrics as the paper:
corpus BLEU-1 (clipped unigram precision x corpus brevity penalty, single ref)
and clinical macro-F1 (rule labeler). Per-item counts are precomputed once;
resampling is then pure numpy, so 1000 resamples take seconds.
"""
import json
from collections import Counter

import numpy as np

D = "/home/deployer/otlora/data/processed/mimic_mlf"
STRATS = ["greedy", "random", "ot", "logprob", "hybrid"]
PAIRS = [("ot", "random"), ("hybrid", "logprob"), ("hybrid", "random"), ("hybrid", "greedy")]
N_BOOT = 1000
CATS = None


def peritem_counts(texts, gts):
    """m = clipped unigram matches, c = hyp words, r = ref words."""
    m = np.zeros(len(texts))
    c = np.zeros(len(texts))
    r = np.zeros(len(texts))
    for i, (h, g) in enumerate(zip(texts, gts)):
        ch, cg = Counter(h.lower().split()), Counter(g.lower().split())
        m[i] = sum((ch & cg).values())
        c[i] = sum(ch.values())
        r[i] = sum(cg.values())
    return m, c, r


def bleu1(m, c, r, w):
    M, C, R = (m * w).sum(), (c * w).sum(), (r * w).sum()
    bp = 1.0 if C > R else np.exp(1 - R / max(C, 1e-9))
    return bp * M / max(C, 1e-9)


def peritem_label_counts(texts):
    from chexpert_label import label_report, CATEGORIES
    global CATS
    CATS = CATEGORIES
    K = len(CATEGORIES)
    tp = np.zeros((len(texts), K))
    fp = np.zeros((len(texts), K))
    fn = np.zeros((len(texts), K))
    for i, t in enumerate(texts):
        lab = label_report(t)
        for k, cat in enumerate(CATEGORIES):
            v = lab[cat]
            if v == 1:
                tp[i, k] = 1  # provisional; gt decides below
    return tp, fp, fn


def main():
    import sys
    sys.path.insert(0, "/home/deployer/otlora/src")
    data = {s: [json.loads(l) for l in open(f"{D}/sft42_full_{s}_pred.jsonl")]
            for s in STRATS}
    gts = [r["gt"] for r in data["greedy"]]
    n = len(gts)
    print("precomputing unigram counts...", flush=True)
    cnt = {s: peritem_counts([r["pred"] for r in data[s]], gts) for s in STRATS}
    print("precomputing label counts...", flush=True)
    from chexpert_label import label_report, CATEGORIES
    K = len(CATEGORIES)
    glab = [label_report(t) for t in gts]
    gpos = np.array([[1 if glab[i][c] == 1 else 0 for c in CATEGORIES] for i in range(n)])
    lab = {}
    for s in STRATS:
        pl = [label_report(r["pred"]) for r in data[s]]
        ppos = np.array([[1 if pl[i][c] == 1 else 0 for c in CATEGORIES] for i in range(n)])
        anym = np.array([[1 if (pl[i][c] != 0 or glab[i][c] != 0) else 0
                          for c in CATEGORIES] for i in range(n)])
        lab[s] = (ppos & gpos, ppos & ~gpos.astype(bool), ~ppos.astype(bool) & gpos, anym)

    def macro(w, s):
        tp = (lab[s][0] * w[:, None]).sum(0)
        fp = (lab[s][1] * w[:, None]).sum(0)
        fn = (lab[s][2] * w[:, None]).sum(0)
        present = (lab[s][3] * w[:, None]).sum(0) > 0
        pr = tp / np.maximum(tp + fp, 1)
        rc = tp / np.maximum(tp + fn, 1)
        f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-9)
        return f1[present].mean()

    rng = np.random.default_rng(0)
    W = rng.integers(0, n, size=(N_BOOT, n))
    out = {}
    # sanity: point estimates on the full set
    w1 = np.ones(n)
    for s in STRATS:
        m, c, r = cnt[s]
        print(f"point {s}: B1={bleu1(m, c, r, w1):.4f} CEma={macro(w1, s):.4f}")
    for a, b in PAIRS:
        d_b1 = np.array([bleu1(cnt[a][0], cnt[a][1], cnt[a][2], w) -
                         bleu1(cnt[b][0], cnt[b][1], cnt[b][2], w) for w in W])
        d_ce = np.array([macro(w, a) - macro(w, b) for w in W])
        win_b1 = float((d_b1 > 0).mean())
        win_ce = float((d_ce > 0).mean())
        out[f"{a}_vs_{b}"] = {
            "B1_delta_mean": round(float(d_b1.mean()), 4),
            "B1_frac_win": round(win_b1, 3),
            "B1_p_two_sided": round(min(win_b1, 1 - win_b1) * 2, 4),
            "CEma_delta_mean": round(float(d_ce.mean()), 4),
            "CEma_frac_win": round(win_ce, 3),
            "CEma_p_two_sided": round(min(win_ce, 1 - win_ce) * 2, 4)}
        print(f"{a} vs {b}: {out[f'{a}_vs_{b}']}", flush=True)
    json.dump(out, open("/home/deployer/otlora/results/bootstrap2.json", "w"), indent=1)
    print("WROTE results/bootstrap2.json")


if __name__ == "__main__":
    main()
