#!/usr/bin/env python3
"""Emit the corrected-scorer comparison table (800-item subset)."""
import json

D = "/home/deployer/otlora/data/processed/mimic_mlf"
ROWS = [
    ("uni\\_256", "uni_256"),
    ("sal\\_256", "sal_256"),
    ("sal\\_all", "sal_all"),
    ("sal\\_sh\\_all", "sal_sh_all"),
    ("hyb\\_sh\\_all", "hyb_sh_all"),
]
lines = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\caption{Corrected scorer (candidate-only window, fixed JL seed, true salience) on the 800-item subset: uniform vs.\ salience marginals, 256 vs.\ all (${\sim}324$) visual tokens, independent vs.\ shared projection. The deployed scorer (prompt-inclusive window, unseeded draw) reached B-1 0.263 / CE$_{ma}$ 0.177 on these items. Random-selection expectation: B-1 $0.232{\pm}0.003$, CE$_{ma}$ $0.170{\pm}0.008$.}\label{tab:r5var}",
         r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lcccc}", r"\toprule",
         r"Scorer & B-1 & CE$_{ma}$ & CEmi & Uniq \\", r"\midrule"]
for name, tag in ROWS:
    n = json.load(open(f"{D}/sft42_k8_r5_{tag}_pred_nlg.json"))
    c = json.load(open(f"{D}/sft42_k8_r5_{tag}_pred_ce2.json"))
    u = n["unique_reports"].split("/")[0]
    lines.append(f"{name} & {n['BLEU-1']:.3f} & {c['F1_macro']:.3f} & {c['F1_micro']:.3f} & {u} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open("/home/deployer/otlora/paper/tables/r5_variants.tex", "w").write("\n".join(lines) + "\n")
print("WROTE r5_variants.tex")
