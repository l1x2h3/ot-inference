#!/usr/bin/env python3
"""Evaluate the rerank_rev5 rescoring variants (nlg + clinical + words)."""
import json
import sys

sys.path.insert(0, "/home/deployer/otlora/src")
from eval_metrics import compute_nlg
from chexpert_label import clinical_f1_from_pairs

D = "/home/deployer/otlora/data/processed/mimic_mlf"
for v in ["uni_256", "sal_256", "sal_all", "sal_sh_all", "hyb_sh_all"]:
    p = f"{D}/sft42_k8_r5_{v}_pred.jsonl"
    try:
        rows = [json.loads(l) for l in open(p)]
    except FileNotFoundError:
        print(v, "missing"); continue
    compute_nlg(p, p.replace(".jsonl", "_nlg.json"))
    clinical_f1_from_pairs(rows, p.replace(".jsonl", "_ce2.json"))
    n = json.load(open(p.replace(".jsonl", "_nlg.json")))
    c = json.load(open(p.replace(".jsonl", "_ce2.json")))
    words = sum(len(r["pred"].split()) for r in rows) / len(rows)
    print(f"{v:10s} B1={n['BLEU-1']:.4f} CEma={c['F1_macro']:.4f} CEmi={c['F1_micro']:.4f} uniq={n['unique_reports']} words={words:.1f}")
