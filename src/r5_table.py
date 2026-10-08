#!/usr/bin/env python3
"""Window/configuration comparison table (800-item subset)."""
import json

D = "/home/deployer/otlora/data/processed/mimic_mlf"
ROWS = [
    ("uni\\_256", "uni_256"),
    ("uni\\_sh", "uni_sh"),
    ("sal\\_256", "sal_256"),
    ("sal\\_all", "sal_all"),
    ("sal\\_sh\\_all", "sal_sh_all"),
    ("hyb\\_sh\\_all (blend)", "hyb_sh_all"),
]
def get(tag, v, f):
    if v == "uni_sh":
        tag = "r5old2" if tag == "r5old" else "r5b"
    p = f"{D}/sft42_k8_{tag}_{v}_pred_{f}.json"
    return json.load(open(p))
lines = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\caption{Scoring-window and configuration analysis on the 800-item subset (fixed JL seed). The \emph{anchored} window includes the decoder's image-token states ahead of the candidate (the deployed configuration); the \emph{candidate-only} window scores candidate tokens alone. Random-selection expectation: B-1 $0.232{\pm}0.003$, CE$_{ma}$ $0.170{\pm}0.008$.}\label{tab:r5var}",
         r"\setlength{\tabcolsep}{3.5pt}",
         r"\begin{tabular}{lcccc}", r"\toprule",
         r"Scorer & \multicolumn{2}{c}{Anchored window} & \multicolumn{2}{c}{Candidate-only} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r" & B-1 & CE$_{ma}$ & B-1 & CE$_{ma}$ \\", r"\midrule"]
for name, v in ROWS:
    a = get("r5old", v, "nlg"); ac = get("r5old", v, "ce2")
    c = get("r5", v, "nlg"); cc = get("r5", v, "ce2")
    lines.append(f"{name} & {a['BLEU-1']:.3f} & {ac['F1_macro']:.3f} & {c['BLEU-1']:.3f} & {cc['F1_macro']:.3f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open("/home/deployer/otlora/paper/tables/r5_variants.tex", "w").write("\n".join(lines) + "\n")
print("WROTE r5_variants.tex (two-window)")
