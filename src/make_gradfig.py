#!/usr/bin/env python3
"""Panel (c) for the analysis figure: gradient-space absorption diagnostic."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rs = json.load(open("results/grad_diag.json"))
cos = [r["cos"] for r in rs]
ratio = [r["ratio"] for r in rs]
steps = list(range(len(rs)))

fig, ax = plt.subplots(figsize=(2.4, 2.6))
ax.scatter(steps, cos, s=9, color="#1a6faf", alpha=0.85, linewidths=0)
ax.axhline(0, color="k", lw=0.6)
ax.axhline(sum(cos) / len(cos), color="#c0392b", lw=1.1, ls="--",
           label="mean $+0.03$")
ax.set_xlabel("probe batch")
ax.set_ylabel(r"$\cos(\nabla\mathcal{L}_{\mathrm{LM}},\,\nabla\mathcal{L}_{\mathrm{OT}})$")
ax.set_ylim(-0.12, 0.2)
ax.legend(fontsize=6.5, frameon=False, loc="upper left")
ax.annotate(r"$\|\nabla\mathcal{L}_{\mathrm{OT}}\| \approx 0.11\,\|\nabla\mathcal{L}_{\mathrm{LM}}\|$"
            "\n" r"($\lambda{=}1$; all 7 LoRA modules)",
            xy=(0.97, 0.05), xycoords="axes fraction", ha="right", fontsize=6)
ax.set_title("(b) Gradients near-orthogonal", fontsize=9)
fig.savefig("paper/figs/grad_diag.pdf",
            bbox_inches="tight", dpi=200)
print("GRADFIG_DONE")
