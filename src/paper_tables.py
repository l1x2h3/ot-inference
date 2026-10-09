#!/usr/bin/env python3
"""Aggregate all experiment outputs into paper-ready markdown tables.

Usage: python paper_tables.py  ->  writes results/TABLES.md
"""
import glob
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(__file__))
from chexpert_label import clinical_f1_from_pairs  # noqa: E402

D = "/home/deployer/otlora/data/processed"
OUT = "/home/deployer/otlora/results/TABLES.md"


def load(ds, tag):
    base = os.path.join(D, ds, f"{tag}_pred")
    if not os.path.exists(base + ".jsonl"):
        return None
    row = {"tag": tag}
    try:
        nlg = json.load(open(base + "_nlg.json"))
        row.update({k: nlg[k] for k in
                    ["BLEU-1", "BLEU-4", "ROUGE-L", "CIDEr", "distinct-2"]})
        row["uniq"] = int(nlg["unique_reports"].split("/")[0])
        row["uniq_n"] = int(nlg["unique_reports"].split("/")[1])
    except FileNotFoundError:
        return None
    ce_file = base + "_ce2.json"
    if os.path.exists(ce_file):
        ce = json.load(open(ce_file))
        row["CEma"], row["CEmi"] = ce["F1_macro"], ce["F1_micro"]
    else:
        rows = [json.loads(l) for l in open(base + ".jsonl")]
        ma, mi = clinical_f1_from_pairs(rows, ce_file)
        row["CEma"], row["CEmi"] = ma, mi
    return row


def fmt(row, keys):
    return " | ".join(f"{row[k]:.3f}" if isinstance(row[k], float) else str(row[k])
                      for k in keys)


def group_mean(rows, keys):
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if k in r]
        out[k] = statistics.mean(vals) if vals else float("nan")
        if len(vals) > 1:
            out[k + "_std"] = statistics.stdev(vals)
    return out


def main():
    KEYS = ["BLEU-1", "BLEU-4", "ROUGE-L", "CIDEr", "distinct-2", "CEma", "CEmi"]
    L = ["# OT-LoRA Experiment Tables (auto-generated)", ""]

    # --- Table 1: main table on MIMIC-MLF (3 seeds mean +- std) ---
    L += ["## Table 1: MIMIC-MLF test (full 4,596 unless noted)", "",
          "| System | " + " | ".join(KEYS) + " | uniq |",
          "|" + "---|" * (len(KEYS) + 2)]
    zs = load("mimic_mlf", "zeroshot_mimic")
    if zs:
        L.append(f"| Zero-shot | {fmt(zs, KEYS)} | {zs['uniq']} |")
    for name, group in [("SFT (LoRA)", ["sft_s42", "sft_1337", "sft_9233"]),
                        ("+ OT-uniform", ["otuni_s42", "otuni_s1337", "otuni_s9233"]),
                        ("+ OT-salience", ["otsal_s42", "otsal_s1337", "otsal_s9233"]),
                        ("+ OT-salience lam0.3", ["otsal_lam03"]),
                        ("+ OT-salience lam1.0", ["otsal_lam10"])]:
        rows = [r for r in (load("mimic_mlf", t) for t in group) if r]
        if not rows:
            continue
        m = group_mean(rows, KEYS)
        cells = []
        for k in KEYS:
            s = m.get(k + "_std")
            cells.append(f"{m[k]:.3f}" + (f" ±{s:.3f}" if s else ""))
        L.append(f"| {name} ({len(rows)} runs) | " + " | ".join(cells)
                 + f" | {int(statistics.mean([r['uniq'] for r in rows]))} |")

    # --- Table 2: reranking (800-item subset) ---
    L += ["", "## Table 2: Training-free reranking (MIMIC-MLF 800-item subset, k=8, SFT adapter)",
          "", "| Strategy | " + " | ".join(KEYS) + " | uniq/n |",
          "|" + "---|" * (len(KEYS) + 2)]
    for strat in ["greedy", "random", "ot", "ot_len", "logprob", "hybrid"]:
        r = load("mimic_mlf", f"sft42_k8_{strat}")
        if r:
            L.append(f"| {strat} | {fmt(r, KEYS)} | {r['uniq_n'] and str(r['uniq'])}/{r['uniq_n']} |")

    # --- Table 3: IU ablation ---
    L += ["", "## Table 3: IU small-data stress test (590 test)", "",
          "| Config | " + " | ".join(KEYS) + " | uniq |",
          "|" + "---|" * (len(KEYS) + 2)]
    for tag in ["zeroshot", "iu_sft", "iu_otuni_f", "iu_otsal_f", "iu_otgated_f"]:
        r = load("iu", tag)
        if r:
            L.append(f"| {tag} | {fmt(r, KEYS)} | {r['uniq']} |")

    open(OUT, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("\nWROTE", OUT)
    emit_latex()


def emit_latex():
    """Emit paper-ready LaTeX table fragments into paper/tables/."""
    os.makedirs("/home/deployer/otlora/paper/tables", exist_ok=True)
    KEYS = ["BLEU-1", "BLEU-4", "ROUGE-L", "CIDEr", "distinct-2", "CEma", "CEmi"]
    rg_path = "/home/deployer/otlora/results/radgraph_full.json"
    rg = {}
    if os.path.exists(rg_path):
        rg = json.load(open(rg_path))
    rg_old_path = "/home/deployer/otlora/results/radgraph_fullold.json"
    if os.path.exists(rg_old_path):
        rg_fullold = json.load(open(rg_old_path))
        rg.update({"sft42_full_r5fullold_uni_256": rg_fullold.get("ot"),
                   "sft42_full_r5fullold_hyb_sh_all": rg_fullold.get("hybrid")})
    ncol = len(KEYS) + 1 + (1 if rg else 0)
    colspec = "l" + "c" * ncol

    # reranking main table (whichever tag set exists: full > k8 subset)
    for tags, caption, fname in [
        (["sft42_full_greedy", "sft42_full_random", "sft42_full_longest",
          "sft42_full_medlen", "sft42_full_mbr",
          "sft42_full_r5fullold_uni_256",
          "sft42_full_logprob",
          "sft42_full_r5fullold_hyb_sh_all"],
         "OT-guided candidate selection on the full MIMIC-MLF test set (k=8). "
         "CE$_{ma}$/CE$_{mi}$: macro/micro clinical F1 over the 14 CheXpert "
         "categories (rule labeler, Sect.~4.1); F1$_{\\mathrm{Rad}}$: "
         "RadGraph micro-F1 (Sect.~4.2); Unique: distinct reports of 4{,}596. "
         "The random row is one uniform draw; over 300 draws the expectation is "
         "B-1 $0.2235{\pm}0.0013$, CE$_{ma}$ $0.167{\pm}0.003$.",
         "rerank_full.tex"),
        (["sft42_k8_greedy", "sft42_k8_random", "sft42_k8_mbr",
          "sft42_k8_cospool", "sft42_k8_ot",
          "sft42_k8_ot_len", "sft42_k8_logprob", "sft42_k8_hybrid"],
         "OT-guided candidate selection, 800-item MIMIC-MLF subset (k=8).",
         "rerank_subset.tex"),
    ]:
        pairs = [(t, r) for t, r in ((t, load("mimic_mlf", t)) for t in tags) if r]
        rows = [r for _, r in pairs]
        if len(rows) < 4:
            continue
        lines = ["\\begin{table*}[t]", "\\centering", "\\small",
                 f"\\caption{{{caption}}}\\label{{tab:rerank}}",
                 "\\setlength{\\tabcolsep}{4pt}",
                 f"\\begin{{tabular}}{{{colspec}}}",
                 "\\toprule",
                 "Strategy & " + " & ".join(KEYS)
                 + (" & F1$_{\\mathrm{Rad}}$" if rg else "") + " & Unique \\\\",
                 "\\midrule"]
        pretty = {"greedy": "Greedy (no reranking)", "random": "Random",
                  "longest": "Longest candidate", "medlen": "Median-length cand.",
                  "mbr": "MBR (1-gram consensus)", "cospool": "Pooled cosine",
                  "ot": "OT cost", "ot_len": "OT + length prior",
                  "logprob": "Mean log-prob", "hybrid": "OT + log-prob (hybrid)",
                  "sft42_full_r5fullold_uni_256": "OT cost",
                                    "sft42_full_r5fullold_hyb_sh_all": "OT + log-prob (hybrid)"}
        OURS = {"ot", "ot_len", "hybrid"}
        for t, r in pairs:
            key = t.split("_k8_")[-1] if "_k8_" in t else (t.split("_full_")[-1] if "_full_" in t else t)
            if t == "sft42_full_r5fullold_uni_256":
                key = "ot"
            elif t == "sft42_full_r5fullold_hyb_sh_all":
                key = "hybrid"
            name = pretty.get(key, t)
            shade = ""
            if key == "hybrid":
                shade = "\\rowcolor{orange!22} "
            elif key in OURS:
                shade = "\\rowcolor{blue!7} "
            cells = " & ".join(f"{r[k]:.3f}" if isinstance(r[k], float) else str(r[k])
                               for k in KEYS)
            rgk = t if t in rg else key
            if rg:
                cells += " & " + (f"{rg[rgk]['F1']:.3f}" if rgk in rg else "--")
            lines.append(f"{shade}{name} & {cells} & {r['uniq']} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
        open(f"/home/deployer/otlora/paper/tables/{fname}", "w").write("\n".join(lines))

    # training-time OT table
    lines = ["\\begin{table}[t]", "\\centering", "\\small",
             "\\caption{Training-time OT supervision under PEFT is absorbed: "
             "clinical F1 stays within noise across OT weights (MIMIC-MLF test). "
             "These rows use the training-evaluation decoder (220 max tokens, no "
             "repetition penalty), which yields a slightly stronger SFT greedy than "
             "the rerank pipeline of Table~1 (B-1 0.188 vs.\\ 0.172); within-table "
             "comparisons hold the decoder fixed.}\\label{tab:trainot}",
             "\\begin{tabular}{lcccc}", "\\toprule",
             "System & B-1 & B-4 & CE$_{ma}$ & CE$_{mi}$ \\\\", "\\midrule"]
    for t, name in [("zeroshot_mimic", "Zero-shot"),
                    ("sft_s42", "SFT (LoRA)"),
                    ("otsal_s42", "~~+ OT ($\\lambda{=}0.05$)"),
                    ("otsal_lam03", "~~+ OT ($\\lambda{=}0.3$)"),
                    ("otsal_lam10", "~~+ OT ($\\lambda{=}1.0$)")]:
        r = load("mimic_mlf", t)
        if r:
            lines.append(f"{name} & {r['BLEU-1']:.3f} & {r['BLEU-4']:.3f} & "
                         f"{r['CEma']:.3f} & {r['CEmi']:.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open("/home/deployer/otlora/paper/tables/train_ot.tex", "w").write("\n".join(lines))

    # OT variant comparison as selection scores
    vlines = ["\\begin{table}[t]", "\\centering", "\\small",
              "\\caption{OT formulation variants as reranking scores "
              "(same candidates, single scoring pass; MIMIC-MLF subset, $k{=}8$; "
              "* = our default).}\\label{tab:variants}",
              "\\setlength{\\tabcolsep}{3pt}",
              "\\begin{tabular}{lccccc}", "\\toprule",
              "Variant & B-1 & B-4 & CE$_{ma}$ & CE$_{mi}$ & Unique \\\\",
              "\\midrule"]
    vnames = [("bal_l2_sal", "Bal-$\\ell_2$-sal\\textsuperscript{*}"),
              ("unif", "Bal-$\\ell_2$-unif"),
              ("gated", "Bal-$\\ell_2$-gate"),
              ("unb_t08", "Unb-$\\tau$0.8"),
              ("unb_t05", "Unb-$\\tau$0.5"),
              ("cos", "Cosine")]
    vrows = [(n, load("mimic_mlf", f"sft42_k8_var_{k}")) for k, n in vnames]
    rnd = load("mimic_mlf", "sft42_k8_random")
    hyb = load("mimic_mlf", "sft42_k8_hybrid")
    have = [r for _, r in vrows if r]
    if have:
        if rnd:
            vlines.append(f"\\midrule\n\\multicolumn{{6}}{{l}}{{\\emph{{Reference baselines}}}} \\\\\n"
                          f"Random selection & {rnd['BLEU-1']:.3f} & {rnd['BLEU-4']:.3f} & "
                          f"{rnd['CEma']:.3f} & {rnd['CEmi']:.3f} & {rnd['uniq']} \\\\")
        for (n, r) in vrows:
            if r:
                shade = "\\rowcolor{orange!22} " if n.startswith("Bal-$\\ell_2$-sal") else ""
                vlines.append(f"{shade}{n} & {r['BLEU-1']:.3f} & {r['BLEU-4']:.3f} & "
                              f"{r['CEma']:.3f} & {r['CEmi']:.3f} & {r['uniq']} \\\\")
        if hyb:
            vlines.append(f"\\midrule\n\\rowcolor{{orange!22}} + log-prob hybrid & "
                          f"{hyb['BLEU-1']:.3f} & {hyb['BLEU-4']:.3f} & "
                          f"{hyb['CEma']:.3f} & {hyb['CEmi']:.3f} & {hyb['uniq']} \\\\")
    else:
        vlines.append("\\multicolumn{6}{l}{\\emph{(variant results pending; table auto-generated)}} \\\\")
    vlines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open("/home/deployer/otlora/paper/tables/ot_variants.tex", "w").write("\n".join(vlines))

    # ---- transfer table: RRG-2461 + IU X-Ray ----
    def c4(r):
        return f"{r['BLEU-1']:.3f} & {r['BLEU-4']:.3f} & {r['CEma']:.3f} & {r['uniq']}"
    pretty_t = {"greedy": "Greedy", "random": "Random", "ot": "OT cost",
                "ot_len": "OT + length prior", "logprob": "Mean log-prob",
                "hybrid": "OT + log-prob (hybrid)"}
    trows = []
    for s in ["greedy", "random", "ot", "ot_len", "logprob", "hybrid"]:
        rr, iu = load("rrg_test", f"rrg_sft42_{s}_clean"), load("iu", f"iu_sft42_{s}")
        if rr and iu:
            trows.append((s, rr, iu))
    if len(trows) >= 4:
        tl = ["\\begin{table*}[t]", "\\centering", "\\small",
              "\\caption{Cross-dataset transfer of candidate selection, with no retraining "
              "or re-tuning: the official MIMIC-CXR-RRG test split (2{,}097 of 2{,}461 studies; 364 with near-duplicate reports in the MIMIC-MLF training split excluded, Sect.~4.1) and IU X-Ray (590), same strategies as Table~\\ref{tab:rerank}.}\\label{tab:transfer}",
              "\\setlength{\\tabcolsep}{4.5pt}",
              "\\begin{tabular}{lcccccccc}", "\\toprule",
              " & \\multicolumn{4}{c}{MIMIC-CXR-RRG (2{,}097)} & "
              "\\multicolumn{4}{c}{IU X-Ray (590)} \\\\",
              "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
              "Strategy & B-1 & B-4 & CE$_{ma}$ & Uniq & B-1 & B-4 & CE$_{ma}$ & Uniq \\\\",
              "\\midrule"]
        for s, rr, iu in trows:
            shade = "\\rowcolor{orange!22} " if s == "hybrid" else (
                "\\rowcolor{blue!7} " if s in ("ot", "ot_len") else "")
            tl.append(f"{shade}{pretty_t[s]} & {c4(rr)} & {c4(iu)} \\\\")
        tl += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
        open("/home/deployer/otlora/paper/tables/transfer.tex", "w").write("\n".join(tl))

    # ---- robustness table: budget k / SFT seed / backbone ----
    def c3(r):
        return f"{r['BLEU-1']:.3f} & {r['CEma']:.3f} & {r['uniq']}"
    bl = ["\\begin{table}[t]", "\\centering", "\\small",
          "\\caption{Robustness of hybrid selection (MIMIC-MLF): candidate budget, "
          "held-out SFT seeds, and backbone scale. Unique-report counts are out of "
          "800 (budget, backbone) or 1{,}500 (seeds) images.}\\label{tab:robust}",
          "\\setlength{\\tabcolsep}{3.5pt}",
          "\\begin{tabular}{llccc}", "\\toprule",
          " & Setting & B-1 & CE$_{ma}$ & Uniq \\\\", "\\midrule"]
    blk = []
    for k in [4, 8, 16]:
        r = load("mimic_mlf", f"sft42_k{k}_hybrid")
        if r:
            blk.append(f" & $k{{=}}{k}$ & {c3(r)} \\\\")
    if blk:
        bl.append("\\multicolumn{5}{l}{\\emph{Candidate budget (800-image subset)}} \\\\")
        bl += blk + ["\\midrule"]
    blk = []
    for tag, nm in [("sfts1337_k8", "seed 1337"), ("sfts9233_k8", "seed 9233")]:
        r = load("mimic_mlf", f"{tag}_hybrid")
        if r:
            blk.append(f" & {nm} & {c3(r)} \\\\")
    if blk:
        bl.append("\\multicolumn{5}{l}{\\emph{Held-out SFT seeds (1{,}500 images)}} \\\\")
        bl += blk + ["\\midrule"]
    blk = []
    for s, nm in [("greedy", "7B greedy"), ("random", "7B random"),
                  ("logprob", "7B log-prob"), ("ot", "7B OT cost"),
                  ("hybrid", "7B hybrid")]:
        r = load("mimic_mlf", f"sft7b_k8_{s}")
        if r:
            shade = "\\rowcolor{orange!22} " if s == "hybrid" else (
                "\\rowcolor{blue!7} " if s == "ot" else "")
            blk.append(f"{shade} & {nm} & {c3(r)} \\\\")
    if blk:
        bl.append("\\multicolumn{5}{l}{\\emph{Backbone scale (Qwen2.5-VL-7B, subset)}} \\\\")
        bl += blk
    blk = []
    for s, nm in [("greedy", "LLaVA greedy"), ("random", "LLaVA random"),
                  ("ot", "LLaVA OT cost"), ("logprob", "LLaVA log-prob"),
                  ("hybrid", "LLaVA hybrid")]:
        r = load("mimic_mlf", f"llava_k8_{s}")
        if r:
            shade = "\\rowcolor{orange!22} " if s == "hybrid" else (
                "\\rowcolor{blue!7} " if s == "ot" else "")
            blk.append(f"{shade} & {nm} & {c3(r)} \\\\")
    if blk:
        bl.append("\\multicolumn{5}{l}{\\emph{Second family (LLaVA-1.5-7B, subset)}} \\\\")
        bl += blk
    bl += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open("/home/deployer/otlora/paper/tables/robustness.tex", "w").write("\n".join(bl))


if __name__ == "__main__":
    main()
