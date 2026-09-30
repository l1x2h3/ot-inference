#!/usr/bin/env python3
"""Generate all paper figures from real experiment outputs -> paper/figs/*.pdf"""
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGS = "paper/figs"
os.makedirs(FIGS, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
D = "data/processed"


def parse_ot_log(path, pattern):
    steps, ots = [], []
    for line in open(path, errors="ignore"):
        m = pattern.search(line)
        if m:
            steps.append(int(m.group(1)))
            ots.append(float(m.group(2)))
    return steps, ots


# ---------------- Fig A: projection collapse vs anchored geometry ----------------
rx = re.compile(r"step(\d+) lm=[\d.]+ ot=([\d.]+)")
L = "results/logs"
s1, o1 = parse_ot_log(f"{L}/learned_collapse.log", rx)
fig, ax = plt.subplots(figsize=(3.4, 2.9))
ax.plot(s1, o1, label="learned projections (collapses)", color="#c0392b",
        lw=2.0, ls="--", dashes=(4, 2))
runs = [
    (f"{L}/otsal_s42.log", r"fixed JL, $\lambda{=}0.05$", "#08306b"),
    (f"{L}/otsal_s1337.log", r"fixed JL, $\lambda{=}0.05$, seed 1337", "#2171b5"),
    (f"{L}/otuni_s42.log", r"fixed JL, uniform marginals", "#6baed6"),
    (f"{L}/otsal_lam03.log", r"fixed JL, $\lambda{=}0.3$", "#fd8d3c"),
    (f"{L}/otsal_lam10.log", r"fixed JL, $\lambda{=}1.0$", "#d94801"),
]
for path, name, color in runs:
    s, o = parse_ot_log(path, rx)
    if s:
        ax.plot(s, o, label=name, color=color, lw=1.1, alpha=0.9)
ax.set_yscale("log")
ax.set_xlabel("optimizer step")
ax.set_ylabel("OT loss")
ax.legend(fontsize=6.2, frameon=False, loc="lower left", ncol=1,
          handlelength=1.6, labelspacing=0.25)
fig.savefig(f"{FIGS}/ot_curves.pdf")

# ---------------- Fig B: lambda absorption ----------------
vals = {}
for tag, name in [("sft_s42", "SFT"), ("otsal_s42", "0.05"), ("otsal_lam03", "0.3"),
                  ("otsal_lam10", "1.0")]:
    p = f"{D}/mimic_mlf/{tag}_pred_ce2.json"
    if os.path.exists(p):
        vals[name] = json.load(open(p))["F1_macro"]
if len(vals) >= 3:
    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    names = list(vals)
    colors = ["#7f8c8d"] + ["#1a6faf"] * (len(names) - 1)
    bars = ax.bar(names, [vals[n] for n in names], color=colors, width=0.55)
    ax.set_xlabel(r"OT weight $\lambda$")
    ax.set_ylabel("clinical macro-F1")
    ax.set_ylim(0.10, 0.17)
    ax.set_title("(a) Supervision is absorbed", fontsize=9)
    for b, n in zip(bars, names):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.002,
                f"{vals[n]:.3f}", ha="center", fontsize=7.5)
    fig.savefig(f"{FIGS}/lambda.pdf")

# ---------------- Fig C: reranking lifts diversity + clinical ----------------
strategies, b1, cema, uniq = [], [], [], []
for s in ["greedy", "random", "ot", "logprob", "hybrid"]:
    p = f"{D}/mimic_mlf/sft42_full_{s}_pred_nlg.json"
    if os.path.exists(p):
        d = json.load(open(p))
        ce = json.load(open(f"{D}/mimic_mlf/sft42_full_{s}_pred_ce2.json"))
        strategies.append(s)
        b1.append(d["BLEU-1"])
        cema.append(ce["F1_macro"])
        uniq.append(int(d["unique_reports"].split("/")[0]) / 4596)
fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.5))
colors = ["#7f8c8d", "#95a5a6", "#e67e22", "#2980b9", "#c0392b"]
for ax, ys, ylab in zip(axes, [b1, cema, uniq],
                        ["BLEU-1", "clinical macro-F1", "unique-report ratio"]):
    ax.bar(strategies, ys, color=colors[:len(ys)], width=0.62)
    ax.set_ylabel(ylab)
    ax.set_ylim(0, max(ys) * 1.22)
    for i, v in enumerate(ys):
        ax.text(i, v + max(ys) * 0.03, f"{v:.3f}" if v < 1 else f"{v:.2f}",
                ha="center", va="bottom", fontsize=6.5, rotation=90)
    ax.tick_params(axis="x", labelsize=7.5, rotation=20)
fig.suptitle("OT-guided candidate selection", fontsize=9, y=1.04)
fig.tight_layout()
fig.savefig(f"{FIGS}/rerank_bars.pdf")

# ---------------- Fig D: per-label clinical F1 delta (hybrid vs greedy) ----------------
gh = json.load(open(f"{D}/mimic_mlf/sft42_k8_hybrid_pred_ce2.json"))["per_label_F1"]
gg = json.load(open(f"{D}/mimic_mlf/sft42_k8_greedy_pred_ce2.json"))["per_label_F1"]
labels = sorted(gh, key=lambda k: gh[k] - gg[k])
delta = [gh[k] - gg[k] for k in labels]
fig, ax = plt.subplots(figsize=(4.6, 2.5))
xs = range(len(labels))
ax.bar(xs, delta, color=["#c0392b" if d < 0 else "#1a6faf" for d in delta], width=0.62)
ax.set_xticks(list(xs))
ax.set_xticklabels(labels, fontsize=7, rotation=45,
                   ha="right", rotation_mode="anchor")
ax.axhline(0, color="k", lw=0.6)
ax.set_ylabel(r"$\Delta$ per-label F1")
fig.tight_layout()
fig.savefig(f"{FIGS}/perlabel.pdf")

print("FIGS_DONE", os.listdir(FIGS))
