#!/usr/bin/env python3
"""Case-study figure: image + per-entity transport heatmaps + colored reports.

Style follows RRG qualitative figures (R2GenCMN/METransformer): for each case
show the X-ray, attention-style overlays locating where entity tokens
transport their mass, and the GT / greedy / hybrid-selected reports with
clinical terms color-coded (green: supported by GT; red: hallucinated).
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
GREEN, RED, GRAY = "#1e8449", "#c0392b", "#555555"


def color_words(ax, text, gt_text, y, size=6.2):
    """Draw text word-by-word with entity coloring vs GT."""
    spans = entity_spans(text)
    gt_labels = label_report(gt_text)
    words = text.split()
    # char offsets per word
    offsets, cur = [], 0
    for w in words:
        offsets.append((cur, cur + len(w)))
        cur += len(w) + 1
    x = 0.01
    for w, (a, b) in zip(words, offsets):
        hit = next((s for s, e in spans if not (b <= s or a >= e)), None)
        if hit is not None:
            phrase = text[hit:text.find(" ", hit)] if text.find(" ", hit) > 0 else text[hit:]
            lab = label_report(phrase)
            supported = any(v == 1 for v in lab.values()) and any(
                gt_labels.get(k, 0) == 1 for k, v in lab.items() if v == 1)
            color = GREEN if supported else RED
        else:
            color = GRAY
        ax.text(x, y, w + " ", fontsize=size, color=color, family="serif",
                va="top", wrap=False)
        x += 0.0085 * min(len(w) + 1, 14)
        if x > 0.97:
            x = 0.01
            y -= 0.035
    return y


def _draw_colored(ax, text, gt_text, x0, x1, y0, dy, size=5.4):
    """Draw wrapped, entity-colored text within column [x0, x1]; return final y."""
    spans = entity_spans(text)
    gt_labels = label_report(gt_text)
    words = text.split()
    offsets, cur = [], 0
    for w in words:
        offsets.append((cur, cur + len(w)))
        cur += len(w) + 1
    max_chars = max(24, int((x1 - x0) / 0.0068))
    lines, line = [], ""
    for w in words:
        if len(line) + len(w) + 1 > max_chars:
            lines.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    if line:
        lines.append(line)
    y = y0
    wi = 0
    for ln in lines:
        x = x0
        for w in ln.split():
            a, b = offsets[wi]
            wi += 1
            hit = next((s for s, e in spans if not (b <= s or a >= e)), None)
            if hit is not None:
                e2 = text.find(" ", hit)
                phrase = text[hit:e2] if e2 > 0 else text[hit:]
                lab = label_report(phrase)
                supported = any(v == 1 for v in lab.values()) and any(
                    gt_labels.get(k, 0) == 1 for k, v in lab.items() if v == 1)
                color = GREEN if supported else RED
            else:
                color = GRAY
            ax.text(x, y, w, fontsize=size, color=color, va="top", family="serif")
            x += 0.0068 * (len(w) + 1)
        y -= dy
    return y


@torch.no_grad()
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel

    D = "data/processed/mimic_mlf"
    cands = {r["id"]: r for r in map(json.loads, open(f"{D}/sft42_k8_cands.jsonl"))}
    hyb = {r["id"]: r for r in map(json.loads, open(f"{D}/sft42_k8_hybrid_pred.jsonl"))}
    items = {it["id"]: it for it in map(json.loads, open(f"{D}/test.jsonl"))}

    # choose 3 informative cases: hybrid differs from greedy + abnormal GT
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

    fig = plt.figure(figsize=(7.6, 3 * 3.1))
    gs = fig.add_gridspec(6, 3, height_ratios=[1.1, 0.85, 1.1, 0.85, 1.1, 0.85],
                          hspace=0.18, wspace=0.06)
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
        plan = info["plan"][0].cpu()  # [Nv, T]
        thw = full.image_grid_thw[0]
        h, w = int(thw[1]) // 2, int(thw[2]) // 2

        # pick two entity words from the generated text
        spans = entity_spans(pick)
        ents = [pick[s:e].split()[0] for s, e in spans if e - s > 3][:4]
        ents = list(dict.fromkeys(ents))[:2] or ["lungs"]

        toks = [processor.tokenizer.decode([i]) for i in ids.tolist()]
        for k in range(3):
            ax = fig.add_subplot(gs[2 * ci, k])
            ax.imshow(img, cmap="gray")
            ax.axis("off")
            if k > 0:
                word = ents[k - 1]
                mass = torch.zeros(per)
                for ti, t in enumerate(toks):
                    if word[:4].lower() in t.lower():
                        mass += plan[:, ti]
                if mass.sum() <= 0:  # fallback: most-coupled token
                    mass = plan.sum(dim=1)
                m = mass[:h * w].reshape(1, 1, h, w)
                m = torch.nn.functional.interpolate(m, size=(img.height, img.width),
                                                    mode="bilinear")[0, 0]
                ax.imshow(m.numpy(), alpha=0.55, cmap="inferno")
                ax.set_title(f'"{word}" mass', fontsize=7, color="#8e44ad", pad=2)

        # dedicated text band BELOW the image band (no overlap)
        tax = fig.add_subplot(gs[2 * ci + 1, :])
        tax.axis("off")
        tax.set_xlim(0, 1); tax.set_ylim(0, 1)
        trunc = lambda s, n=58: " ".join(s.split()[:n]) + (" [...]" if len(s.split()) > n else "")
        # left column: reference; right column: greedy vs hybrid
        tax.text(0.005, 0.97, "Reference:", fontsize=6.2, fontweight="bold", va="top")
        _draw_colored(tax, trunc(r["gt"]), r["gt"], x0=0.005, x1=0.475,
                      y0=0.90, dy=0.115, size=5.4)
        y = 0.97
        for name, txt, bold in [("Greedy:", r["greedy"], False),
                                ("Hybrid (ours):", pick, False)]:
            tax.text(0.525, y, name, fontsize=6.2, fontweight="bold" if bold else "normal",
                     va="top")
            y = _draw_colored(tax, trunc(txt), r["gt"], x0=0.525, x1=0.995,
                              y0=y - 0.06, dy=0.115, size=5.4)
            y -= 0.05
    fig.savefig("paper/figs/case_study.pdf", bbox_inches="tight", dpi=220)
    print("CASE_STUDY_DONE")


if __name__ == "__main__":
    main()
