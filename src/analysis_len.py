#!/usr/bin/env python3
"""Length / brevity-penalty / precision analysis for the selection strategies
(review4 item 7) plus pure-length selection baselines and finding precision.

Outputs:
  data/processed/mimic_mlf/sft42_full_{longest,medlen}_pred.jsonl (+ evals)
  results/length_analysis.json
"""
import json
import math
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from eval_metrics import compute_nlg  # noqa: E402
from chexpert_label import label_report, clinical_f1_from_pairs  # noqa: E402

D = "/home/deployer/otlora/data/processed/mimic_mlf"
OUT = "/home/deployer/otlora/results/length_analysis.json"
STRATS = ["greedy", "random", "mbr", "ot", "logprob", "hybrid"]


def words(t):
    return t.lower().split()


def unigram_stats(rows):
    """Corpus-level clipped unigram precision (BLEU-1 without BP) and BP."""
    num = den = C = R = 0
    for r in rows:
        h, g = Counter(words(r["pred"])), Counter(words(r["gt"]))
        num += sum((h & g).values())
        den += sum(h.values())
        C += sum(h.values())
        R += sum(g.values())
    prec = num / max(1, den)
    bp = 1.0 if C > R else math.exp(1 - R / C)
    return prec, bp, C / len(rows), R / len(rows)


def finding_pr(rows):
    """Micro positive-finding precision/recall from the rule labeler."""
    tp = fp = fn = 0
    for r in rows:
        p = {c for c, v in label_report(r["pred"]).items() if v == 1}
        g = {c for c, v in label_report(r["gt"]).items() if v == 1}
        tp += len(p & g)
        fp += len(p - g)
        fn += len(g - p)
    return tp / max(1, tp + fp), tp / max(1, tp + fn)


def main():
    res = {}

    # ---- per-strategy length / BP / precision / finding-PR ----
    for s in STRATS:
        rows = [json.loads(l) for l in open(f"{D}/sft42_full_{s}_pred.jsonl")]
        prec, bp, hyp_len, ref_len = unigram_stats(rows)
        fpr, frc = finding_pr(rows)
        res[s] = {"p1_nobp": round(prec, 4), "BP": round(bp, 4),
                  "hyp_words": round(hyp_len, 1), "ref_words": round(ref_len, 1),
                  "p1_x_bp": round(prec * bp, 4),
                  "finding_prec": round(fpr, 4), "finding_rec": round(frc, 4)}
        print(s, res[s], flush=True)

    # ---- pure-length baselines from the candidate pool ----
    cands_rows = [json.loads(l) for l in open(f"{D}/sft42_full_cands.jsonl")]
    longest, medlen = [], []
    for r in cands_rows:
        cs = [r["greedy"]] + r["cands"]
        lens = [len(words(c)) for c in cs]
        longest.append(cs[max(range(len(cs)), key=lambda i: lens[i])])
        med = sorted(lens)[len(lens) // 2]
        medlen.append(cs[min(range(len(cs)), key=lambda i: abs(lens[i] - med))])
    for strat, preds in [("longest", longest), ("medlen", medlen)]:
        p = f"{D}/sft42_full_{strat}_pred.jsonl"
        with open(p, "w") as f:
            for r, pr in zip(cands_rows, preds):
                f.write(json.dumps({"id": r["id"], "gt": r["gt"], "pred": pr}) + "\n")
        compute_nlg(p, p.replace(".jsonl", "_nlg.json"))
        clinical_f1_from_pairs([json.loads(l) for l in open(p)], p.replace(".jsonl", "_ce2.json"))
        rows = [json.loads(l) for l in open(p)]
        prec, bp, hyp_len, ref_len = unigram_stats(rows)
        fpr, frc = finding_pr(rows)
        n = json.load(open(p.replace(".jsonl", "_nlg.json")))
        c = json.load(open(p.replace(".jsonl", "_ce2.json")))
        res[strat] = {"p1_nobp": round(prec, 4), "BP": round(bp, 4),
                      "hyp_words": round(hyp_len, 1), "ref_words": round(ref_len, 1),
                      "B1": round(n["BLEU-1"], 4), "CEma": round(c["F1_macro"], 4),
                      "uniq": n["unique_reports"],
                      "finding_prec": round(fpr, 4), "finding_rec": round(frc, 4)}
        print(strat, res[strat], flush=True)

    # ---- OT-score vs candidate length (full test) ----
    import math
    sc = [json.loads(l) for l in open(f"{D}/sft42_full_scores.json")]
    pairs = []
    for r, srow in zip(cands_rows, sc):
        cs = [r["greedy"]] + r["cands"]
        for i, c in enumerate(cs):
            pairs.append((len(words(c)), srow["ot"][str(i)]))
    n = len(pairs)
    mx = sum(x for x, _ in pairs) / n
    my = sum(y for _, y in pairs) / n
    cov = sum((x - mx) * (y - my) for x, y in pairs) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x, _ in pairs) / n)
    sy = math.sqrt(sum((y - my) ** 2 for _, y in pairs) / n)
    res["ot_len_pearson"] = {"r": round(cov / (sx * sy), 4), "n": n}
    print("OT-vs-length (global):", res["ot_len_pearson"], flush=True)

    # per-item Spearman within candidate sets: does OT prefer longer candidates?
    from scipy.stats import spearmanr
    rhos = []
    for r, srow in zip(cands_rows, sc):
        cs = [r["greedy"]] + r["cands"]
        if len(cs) < 3:
            continue
        lens = [len(words(c)) for c in cs]
        ots = [-srow["ot"][str(i)] for i in range(len(cs))]  # higher = better
        rho = spearmanr(lens, ots).statistic
        if rho == rho:  # not NaN
            rhos.append(rho)
    res["ot_len_spearman_within"] = {"mean_rho": round(sum(rhos) / len(rhos), 4),
                                     "n_items": len(rhos)}
    print("OT-vs-length (within):", res["ot_len_spearman_within"], flush=True)

    json.dump(res, open(OUT, "w"), indent=1)
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
