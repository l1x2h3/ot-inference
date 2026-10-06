#!/usr/bin/env python3
"""review5 offline analyses (no GPU): all operate on stored candidate sets.

E3  target-length selector: pick candidate closest to the train mean length
E4  same-length contrast: OT pick vs the *other* candidate closest in length,
    paired BLEU-1 / CEma with bootstrap CI (per-item, controls length)
E5  random multi-draw: distribution of corpus B-1 / CEma over 1000 random
    draws (subset and full), not a single lucky draw
E6  lastv mean length (collapse diagnosis)
E7  10k-resample paired bootstrap for the key pairs
Writes results/rev5_offline.json
"""
import json
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "/home/deployer/otlora/src")
from chexpert_label import label_report, CATEGORIES

D = "/home/deployer/otlora/data/processed/mimic_mlf"
REF_MEAN = 69.1  # words (length_analysis)
N = 800  # subset size


def peritem(texts, gts):
    m = np.zeros(len(texts)); c = np.zeros(len(texts)); r = np.zeros(len(texts))
    for i, (h, g) in enumerate(zip(texts, gts)):
        ch, cg = Counter(h.lower().split()), Counter(g.lower().split())
        m[i] = sum((ch & cg).values())
        c[i] = sum(ch.values()); r[i] = sum(cg.values())
    return m, c, r


def bleu1(m, c, r):
    M, C, R = m.sum(), c.sum(), r.sum()
    bp = 1.0 if C > R else np.exp(1 - R / max(C, 1e-9))
    return bp * M / max(C, 1e-9)


def labcounts(texts):
    n = len(texts); K = len(CATEGORIES)
    ppos = np.zeros((n, K)); gpos = np.zeros((n, K)); anym = np.zeros((n, K))
    return ppos, gpos, anym


def macro_from(pos, gt, anym):
    pos = pos.astype(bool); gt = gt.astype(bool)
    tp = (pos & gt).sum(0); fp = (pos & ~gt).sum(0)
    fn = (~pos & gt).sum(0)
    present = anym.sum(0) > 0
    pr = tp / np.maximum(tp + fp, 1); rc = tp / np.maximum(tp + fn, 1)
    f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-9)
    return f1[present].mean()


def full_labels(texts, gts_lab):
    n = len(texts); K = len(CATEGORIES)
    ppos = np.zeros((n, K)); g = np.zeros((n, K)); anym = np.zeros((n, K))
    for i, t in enumerate(texts):
        pl = label_report(t)
        for k, cat in enumerate(CATEGORIES):
            ppos[i, k] = 1 if pl[cat] == 1 else 0
            g[i, k] = 1 if gts_lab[i][cat] == 1 else 0
            anym[i, k] = 1 if (pl[cat] != 0 or gts_lab[i][cat] != 0) else 0
    return ppos, g, anym


def main():
    rows = [json.loads(l) for l in open(f"{D}/sft42_k8_cands.jsonl")]
    sub = rows[:N]
    ot = [json.loads(l) for l in open(f"{D}/sft42_k8_ot_pred.jsonl")]
    hybrid = [json.loads(l) for l in open(f"{D}/sft42_k8_hybrid_pred.jsonl")]
    gts = [r["gt"] for r in rows]
    out = {}

    # ---------- E3 target-length selector (subset + full) ----------
    for tag, rr in [("subset", sub), ("full", rows)]:
        picks = [min([r["greedy"]] + r["cands"],
                     key=lambda c: abs(len(c.split()) - REF_MEAN)) for r in rr]
        m, c, rw = peritem(picks, [r["gt"] for r in rr])
        gl = [label_report(r["gt"]) for r in rr]
        pp, g, am = full_labels(picks, gl)
        words = np.mean([len(p.split()) for p in picks])
        out[f"e3_tgtlen_{tag}"] = {"B1": round(bleu1(m, c, rw), 4),
                                   "CEma": round(float(macro_from(pp, g, am)), 4),
                                   "words": round(float(words), 1)}
        print("E3", tag, out[f"e3_tgtlen_{tag}"], flush=True)

    # ---------- E4 same-length contrast (subset; OT and hybrid) ----------
    for strat, preds in [("ot", ot), ("hybrid", hybrid)]:
        dm, dce = [], []
        for i, r in enumerate(sub):
            pool = [r["greedy"]] + r["cands"]
            pick = preds[i]["pred"]
            if pick not in pool:
                continue
            others = [c for c in pool if c != pick]
            if not others:
                continue
            w = len(pick.split())
            near = min(others, key=lambda c: abs(len(c.split()) - w))
            if len(near.split()) == w:
                d = 0  # same length pair
            a, b = pick, near
            ma, ca, ra = peritem([a], [r["gt"]]); mb, cb, rb = peritem([b], [r["gt"]])
            # per-item unigram F-ish proxy: use clipped precision + recall combo?
            # We keep corpus-level B1 for pairs via sums in the bootstrap below;
            # here store per-item m/c/r for both picks.
            dm.append((ma[0], ca[0], ra[0], mb[0], cb[0], rb[0]))
        A = np.array(dm)
        # labels
        gla = [label_report(sub[i]["gt"]) for i in range(len(dm))]
        picks_a = [ot[i]["pred"] for i in range(len(dm))]
        picks_b = []
        k = 0
        for i, r in enumerate(sub):
            pool = [r["greedy"]] + r["cands"]
            pick = preds[i]["pred"]
            if pick not in pool:
                continue
            others = [c for c in pool if c != pick]
            if not others:
                continue
            w = len(pick.split())
            picks_b.append(min(others, key=lambda c: abs(len(c.split()) - w)))
        pa, ga, ama = full_labels(picks_a, gla)
        pb, gb, amb = full_labels(picks_b, gla)
        # bootstrap over items
        rng = np.random.default_rng(0)
        n = len(dm)
        W = rng.integers(0, n, size=(2000, n))
        d_b1 = np.array([bleu1(A[w, 0], A[w, 1], A[w, 2]) - bleu1(A[w, 3], A[w, 4], A[w, 5])
                         for w in W])
        # CEma delta needs per-item label matrices; do slow loop with caching
        def macro_i(w, ppos, g, am):
            pw, gw = ppos[w].astype(bool), g[w].astype(bool)
            tp = (pw & gw).sum(0); fp = (pw & ~gw).sum(0)
            fn = (~pw & gw).sum(0)
            present = am[w].sum(0) > 0
            pr = tp / np.maximum(tp + fp, 1); rc = tp / np.maximum(tp + fn, 1)
            f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-9)
            return f1[present].mean()
        d_ce = np.array([macro_i(w, pa, ga, ama) - macro_i(w, pb, gb, amb) for w in W])
        out[f"e4_samelength_{strat}"] = {
            "n_pairs": n,
            "mean_dB1": round(float(d_b1.mean()), 5),
            "ci95_dB1": [round(float(np.percentile(d_b1, 2.5)), 5),
                          round(float(np.percentile(d_b1, 97.5)), 5)],
            "frac_dB1_pos": round(float((d_b1 > 0).mean()), 3),
            "mean_dCEma": round(float(d_ce.mean()), 5),
            "ci95_dCEma": [round(float(np.percentile(d_ce, 2.5)), 5),
                            round(float(np.percentile(d_ce, 97.5)), 5),
                            ],
            "frac_dCEma_pos": round(float((d_ce > 0).mean()), 3)}
        print("E4", strat, out[f"e4_samelength_{strat}"], flush=True)

    # ---------- E5 random multi-draw ----------
    for tag, rr in [("subset", sub), ("full", rows)]:
        rng = np.random.default_rng(1)
        b1s, ces = [], []
        gl = [label_report(r["gt"]) for r in rr]
        # precompute per-candidate m/c/r and labels
        cand_m, cand_c, cand_r, cand_pp, cand_am = [], [], [], [], []
        K = len(CATEGORIES)
        for r in rr:
            pool = [r["greedy"]] + r["cands"]
            ms, cs, rs_, pps, ams = [], [], [], [], []
            for c in pool:
                m, cc, rr_ = peritem([c], [r["gt"]])
                ms.append(m[0]); cs.append(cc[0]); rs_.append(rr_[0])
                pl = label_report(c)
                pv = np.array([1 if pl[cat] == 1 else 0 for cat in CATEGORIES])
                gv = np.array([1 if gl[len(cand_m)][cat] == 1 else 0 for cat in CATEGORIES])
                pps.append(pv)
                ams.append(((np.array([1 if pl[cat] != 0 else 0 for cat in CATEGORIES]) != 0) |
                            (gv != 0)).astype(int))
            cand_m.append(ms); cand_c.append(cs); cand_r.append(rs_)
            cand_pp.append(np.array(pps)); cand_am.append(np.array(ams))
        g_all = np.array([[1 if gl[i][cat] == 1 else 0 for cat in CATEGORIES]
                          for i in range(len(rr))])
        for it in range(300):
            sel = [rng.integers(0, len(cm)) for cm in cand_m]
            M = sum(cand_m[i][sel[i]] for i in range(len(rr)))
            C = sum(cand_c[i][sel[i]] for i in range(len(rr)))
            R = sum(cand_r[i][0] for i in range(len(rr)))
            b1s.append(bleu1(np.array([M]), np.array([C]), np.array([R])))
            ppos = np.array([cand_pp[i][sel[i]] for i in range(len(rr))])
            am = np.array([cand_am[i][sel[i]] for i in range(len(rr))])
            ces.append(macro_from(ppos, g_all, am))
        out[f"e5_random_dist_{tag}"] = {"B1_mean": round(float(np.mean(b1s)), 4),
                                        "B1_std": round(float(np.std(b1s)), 4),
                                        "CEma_mean": round(float(np.mean(ces)), 4),
                                        "CEma_std": round(float(np.std(ces)), 4),
                                        "n_draws": 300}
        print("E5", tag, out[f"e5_random_dist_{tag}"], flush=True)

    # ---------- E6 lastv mean length ----------
    lv = [json.loads(l) for l in open(f"{D}/sft42_k8_abl_lastv_pred.jsonl")]
    dfl = [json.loads(l) for l in open(f"{D}/sft42_k8_abl_default_pred.jsonl")]
    out["e6_lastv"] = {"lastv_words": round(float(np.mean([len(r["pred"].split()) for r in lv])), 1),
                       "default_words": round(float(np.mean([len(r["pred"].split()) for r in dfl])), 1)}
    print("E6", out["e6_lastv"])

    # ---------- E7 10k bootstrap ----------
    import importlib.util
    spec = importlib.util.spec_from_file_location("b3", "/home/deployer/otlora/src/bootstrap3.py")
    b3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module.__self__ if False else None
    # reuse logic inline instead of importing (it runs on import)
    STRATS = ["greedy", "random", "ot", "logprob", "hybrid"]
    data = {s: [json.loads(l) for l in open(f"{D}/sft42_full_{s}_pred.jsonl")] for s in STRATS}
    g2 = [r["gt"] for r in data["greedy"]]; n2 = len(g2)
    cnt = {s: peritem([r["pred"] for r in data[s]], g2) for s in STRATS}
    gl2 = [label_report(t) for t in g2]
    gpos2 = np.array([[1 if gl2[i][c] == 1 else 0 for c in CATEGORIES] for i in range(n2)])
    lab2 = {}
    for s in STRATS:
        pl = [label_report(r["pred"]) for r in data[s]]
        ppos = np.array([[1 if pl[i][c] == 1 else 0 for c in CATEGORIES] for i in range(n2)])
        anym = np.array([[1 if (pl[i][c] != 0 or gl2[i][c] != 0) else 0 for c in CATEGORIES]
                         for i in range(n2)])
        lab2[s] = (ppos, anym)

    def macro2(w, s):
        ppos, anym = lab2[s]
        tp = (ppos[w] & gpos2[w]).sum(0); fp = (ppos[w] & ~gpos2[w].astype(bool)).sum(0)
        fn = (~ppos[w].astype(bool) & gpos2[w]).sum(0)
        present = anym[w].sum(0) > 0
        pr = tp / np.maximum(tp + fp, 1); rc = tp / np.maximum(tp + fn, 1)
        f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-9)
        return f1[present].mean()

    rng = np.random.default_rng(0)
    W = rng.integers(0, n2, size=(10000, n2))
    for a, b in [("ot", "random"), ("hybrid", "logprob"), ("hybrid", "random"), ("hybrid", "greedy")]:
        db = np.array([bleu1(cnt[a][0][w], cnt[a][1][w], cnt[a][2][w]) -
                       bleu1(cnt[b][0][w], cnt[b][1][w], cnt[b][2][w]) for w in W])
        dc = np.array([macro2(w, a) - macro2(w, b) for w in W])
        out[f"e7_10k_{a}_vs_{b}"] = {
            "dB1": round(float(db.mean()), 4), "frac_B1_pos": round(float((db > 0).mean()), 4),
            "dCEma": round(float(dc.mean()), 4), "frac_CEma_pos": round(float((dc > 0).mean()), 4)}
        print("E7", a, b, out[f"e7_10k_{a}_vs_{b}"], flush=True)

    json.dump(out, open("/home/deployer/otlora/results/rev5_offline.json", "w"), indent=1)
    print("WROTE results/rev5_offline.json")


if __name__ == "__main__":
    main()
