#!/usr/bin/env python3
"""Case study as LaTeX table rows: images as cells, colored report text in
p{}-columns (LaTeX handles alignment). Writes figs/case_*.png + tables/case_rows.tex
"""
import json
import sys

import torch

sys.path.insert(0, "src")
from ot_loss import CAOTLoss  # noqa: E402
from salience import entity_spans  # noqa: E402
from chexpert_label import label_report  # noqa: E402

PROMPT = ("You are a radiologist. Read the chest X-ray and write the findings "
          "section of the radiology report.")


def latexify(w):
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("#", r"\#"), ("$", r"\$"), ("_", r"\_")]:
        w = w.replace(a, b)
    return w


def colored_tex(text, gt_text, max_words=45):
    """Report text with \\textcolor-marked entity words."""
    spans = entity_spans(text)
    gt_labels = label_report(gt_text)
    words = text.split()[:max_words]
    offsets, cur = [], 0
    for w in words:
        offsets.append((cur, cur + len(w)))
        cur += len(w) + 1
    out = []
    for w, (a, b) in zip(words, offsets):
        hit = next((s for s, e in spans if not (b <= s or a >= e)), None)
        if hit is not None:
            e2 = text.find(" ", hit)
            phrase = text[hit:e2] if e2 > 0 else text[hit:]
            lab = label_report(phrase)
            supported = any(v == 1 for v in lab.values()) and any(
                gt_labels.get(k, 0) == 1 for k, v in lab.items() if v == 1)
            color = "cgreen" if supported else "cred"
            out.append(r"\textcolor{%s}{%s}" % (color, latexify(w)))
        else:
            out.append(latexify(w))
    tex = " ".join(out)
    if len(text.split()) > max_words:
        tex += r"~[\ldots]"
    return tex


@torch.no_grad()
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel

    D = "data/processed/mimic_mlf"
    FIGS = "paper/figs"
    cands = {r["id"]: r for r in map(json.loads, open(f"{D}/sft42_k8_cands.jsonl"))}
    hyb = {r["id"]: r for r in map(json.loads, open(f"{D}/sft42_k8_hybrid_pred.jsonl"))}
    items = {it["id"]: it for it in map(json.loads, open(f"{D}/test.jsonl"))}

    chosen = []
    for cid, r in cands.items():
        h = hyb[cid]["pred"]
        if h == r["greedy"]:
            continue
        gl, hl = label_report(r["gt"]), label_report(h)
        gt_pos = {k for k, v in gl.items() if v == 1}
        hyb_pos = {k for k, v in hl.items() if v == 1}
        if len(gt_pos) >= 2 and len(hyb_pos & gt_pos) >= 2:
            chosen.append((cid, len(hyb_pos & gt_pos)))
        if len(chosen) >= 12:
            break
    chosen = [c for c, _ in sorted(chosen, key=lambda x: -x[1])[:3]]
    print("cases:", chosen)

    mp = "models/Qwen2.5-VL-3B-Instruct"
    processor = AutoProcessor.from_pretrained(mp, max_pixels=512 * 512)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        mp, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model = PeftModel.from_pretrained(model, "results/sft_s42/adapter_ep1")
    model.eval()
    ot = CAOTLoss(d_vision=model.config.hidden_size, d_text=model.config.hidden_size).cuda()

    rows = []
    for ci, cid in enumerate(chosen):
        r, item = cands[cid], items[cid]
        img = Image.open(item["image_path"]).convert("RGB")
        pick = hyb[cid]["pred"]
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": item["image_path"]},
            {"type": "text", "text": PROMPT}]}]
        prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = processor(text=[prompt], images=[img], return_tensors="pt").to("cuda:0")
        full = processor(text=[prompt + pick + "<|im_end|>"], images=[img],
                         return_tensors="pt").to("cuda:0")
        out = model(input_ids=full.input_ids, attention_mask=full.attention_mask,
                    pixel_values=full.pixel_values, image_grid_thw=full.image_grid_thw,
                    output_hidden_states=True)
        plen = enc.input_ids.shape[1]
        tail = out.hidden_states[-1][0, plen:].float()
        ids = full.input_ids[0, plen:]
        vout = model.base_model.model.visual(full.pixel_values, grid_thw=full.image_grid_thw).float()
        per = int((full.image_grid_thw.prod(dim=1) // 4).sum())
        v = vout[:per]
        _, info = ot(v.unsqueeze(0), tail.unsqueeze(0))
        plan = info["plan"][0].cpu()
        thw = full.image_grid_thw[0]
        h, w = int(thw[1]) // 2, int(thw[2]) // 2

        spans = entity_spans(pick)
        ents = [pick[s:e].split()[0] for s, e in spans if e - s > 3][:4]
        ents = list(dict.fromkeys(ents))[:2] or ["lungs"]
        toks = [processor.tokenizer.decode([i]) for i in ids.tolist()]

        # three separate images per case, dashed gray frames
        paths = []
        for k in range(3):
            fig, ax = plt.subplots(figsize=(1.8, 1.8))
            ax.imshow(img, cmap="gray")
            if k > 0:
                word = ents[k - 1]
                mass = torch.zeros(per)
                for ti, t in enumerate(toks):
                    if word[:4].lower() in t.lower():
                        mass += plan[:, ti]
                if mass.sum() <= 0:
                    mass = plan.sum(dim=1)
                m = mass[:h * w].reshape(1, 1, h, w)
                m = torch.nn.functional.interpolate(m, size=(img.height, img.width),
                                                    mode="bilinear")[0, 0]
                ax.imshow(m.numpy(), alpha=0.55, cmap="inferno")
                ax.set_title(r'"%s"' % word[:12], fontsize=6, color="#8e44ad", pad=1.5)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_linestyle((0, (4, 3)))
                sp.set_color("#999999"); sp.set_linewidth(0.8)
            p = f"{FIGS}/case_{ci}_{k}.png"
            fig.savefig(p, bbox_inches="tight", dpi=300, pad_inches=0.02)
            plt.close(fig)
            paths.append(p)

        labels = ["Reference", "Greedy", "Hybrid (ours)"]
        texts = [r["gt"], r["greedy"], pick]
        for k in range(3):
            rows.append(
                r"\begin{minipage}[c]{2.15cm}\centering "
                + r"\includegraphics[width=2.0cm]{figs/case_%d_%d.png}" % (ci, k)
                + r"\end{minipage} & \begin{minipage}[c]{9.9cm}\scriptsize\raggedright "
                + r"\textbf{%s:} " % labels[k] + colored_tex(texts[k], r["gt"])
                + r"\end{minipage}"
                + (r" \\[4pt]" if k < 2 else r" \\"))
        rows.append(r"\cmidrule{1-2}")

    # emit two tables: first case for the main text, remaining for the appendix
    wrap = ("\\begin{tabular}{@{}c@{\\hspace{6pt}}p{9.9cm}@{}}\n%s\\end{tabular}")
    main_body = wrap % "\n".join(rows[:3]) + "\n"          # case 1: 3 rows
    app_body = wrap % "\n".join(rows[4:11]) + "\n"         # cases 2-3 + middle cmidrule
    open("paper/tables/case_main.tex", "w").write(main_body)
    open("paper/tables/case_app.tex", "w").write(app_body)
    open("paper/tables/case_rows.tex", "w").write(
        wrap % "\n".join(rows[:-1]) + "\n")                # legacy: all three
    print("CASE_ROWS_DONE")


if __name__ == "__main__":
    main()
