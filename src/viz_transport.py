#!/usr/bin/env python3
"""Qualitative figure: image with per-patch transport mass overlay + top-coupled tokens."""
import json
import sys

import torch

sys.path.insert(0, "/home/deployer/otlora/src")
from ot_loss import CAOTLoss  # noqa: E402
from salience import token_salience  # noqa: E402

PROMPT = ("You are a radiologist. Read the chest X-ray and write the findings "
          "section of the radiology report.")


@torch.no_grad()
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel

    mp = "/home/deployer/otlora/models/Qwen2.5-VL-3B-Instruct"
    ad = "/home/deployer/otlora/results/sft_s42/adapter_ep1"
    processor = AutoProcessor.from_pretrained(mp, max_pixels=512 * 512)
    processor.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        mp, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model = PeftModel.from_pretrained(model, ad)
    model.eval()

    # pick an abnormal example: GT mentions pleural effusion
    items = [json.loads(l) for l in open(
        "/home/deployer/otlora/data/processed/mimic_mlf/test.jsonl")]
    it = next(x for x in items if "pleural effusion" in x["report"].lower()
              and "no " not in x["report"].lower()[:200])
    img = Image.open(it["image_path"]).convert("RGB")

    msgs = [{"role": "user", "content": [
        {"type": "image", "image": it["image_path"]},
        {"type": "text", "text": PROMPT}]}]
    prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = processor(text=[prompt], images=[img], return_tensors="pt").to("cuda:0")
    gen = model.generate(**enc, max_new_tokens=200, do_sample=False,
                         repetition_penalty=1.05, min_new_tokens=45)
    cand = processor.decode(gen[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
    print("GEN:", cand[:150])

    # forward on prompt+cand to get hidden states
    full = processor(text=[prompt + cand + "<|im_end|>"], images=[img],
                     return_tensors="pt").to("cuda:0")
    plen = enc.input_ids.shape[1]
    out = model(input_ids=full.input_ids, attention_mask=full.attention_mask,
                pixel_values=full.pixel_values, image_grid_thw=full.image_grid_thw,
                output_hidden_states=True)
    tail = out.hidden_states[-1][0, plen:].float()  # [T, d]
    ids = full.input_ids[0, plen:]
    sal = torch.tensor(token_salience(processor.tokenizer, ids.tolist(), cand)).cuda()

    visual = model.base_model.model.visual
    vout = visual(full.pixel_values, grid_thw=full.image_grid_thw).float()
    per = int((full.image_grid_thw.prod(dim=1) // 4).sum())
    v = vout[:per]

    ot = CAOTLoss(d_vision=v.size(1), d_text=tail.size(1), token_marginal="salience").cuda()
    _, info = ot(v.unsqueeze(0), tail.unsqueeze(0),
                 token_salience=sal.unsqueeze(0))
    plan = info["plan"][0].cpu()  # [Nv, T]

    # NOTE (review4 item 13): with balanced OT the per-patch row sums equal the
    # uniform marginal a_i and the per-token column sums equal b_t BY
    # CONSTRUCTION -- plotting those shows marginals, not structure. We plot
    # row-normalized coupling profiles instead: where each token's mass comes
    # from, and how spatially concentrated (entropy) that profile is.
    col = plan / (plan.sum(dim=0, keepdim=True) + 1e-9)      # [Nv, T] profiles
    ent = -(col * (col + 1e-12).log()).sum(dim=0)            # spatial entropy per token

    # spatial layout: reconstruct grid from grid_thw (assume single image, t*h/2 x w/2)
    thw = full.image_grid_thw[0]
    h, w = int(thw[1]) // 2, int(thw[2]) // 2

    # token axis -> words (merge BPE fragments; subwords otherwise rank high)
    toks = [processor.tokenizer.decode([i]) for i in ids.tolist()]
    tok_ent = ent.tolist()
    words, went, tidx, cur, curm, curt = [], [], [], "", [], []
    for ti, (tk, e) in enumerate(zip(toks, tok_ent)):
        if "<|" in tk or ">" == tk.strip():  # special tokens (e.g. <|im_end|>)
            continue
        if (tk.startswith(" ") or tk.startswith("\n")) and cur.strip():
            words.append(cur.strip())
            # a word's concentration = entropy of its averaged profile
            prof = col[:, curt].mean(dim=1)
            went.append(float(-(prof * (prof + 1e-12).log()).sum()))
            cur, curm, curt = "", [], []
        cur += tk
        curm.append(e)
        curt.append(ti)
    if cur.strip():
        prof = col[:, curt].mean(dim=1)
        went.append(float(-(prof * (prof + 1e-12).log()).sum()))

    # most-localized ENTITY words only (review5: "its"/"not" previously ranked
    # high because any frequent subword has a peaked-by-construction profile).
    # Entity = words inside a matched CheXpert phrase span (char-aligned), NOT
    # phrase components (splitting phrases pollutes the lexicon with "is/the").
    from salience import entity_spans
    STOP = {"is", "are", "the", "a", "an", "of", "and", "or", "with", "without",
            "its", "not", "no", "in", "on", "to", "as", "at", "by", "for",
            "patient", "remains", "again", "demonstrates"}
    # word char offsets while rebuilding the joined text
    joined, offs, o = "", [], 0
    for wd in words:
        joined += (" " if joined else "") + wd
        offs.append((o, o + len(wd)))
        o += len(wd) + 1
    span_words = set()
    for s, e in entity_spans(joined):
        for i, (ws, we) in enumerate(offs):
            if ws < e and we > s:
                span_words.add(i)
    cand_idx = [i for i in (span_words | {i for i, wd in enumerate(words)
                                          if len(wd) > 6 and wd.lower().strip(".,") not in STOP})
                if words[i].lower().strip(".,") not in STOP]
    cand_idx = sorted(cand_idx)
    top = sorted(cand_idx, key=lambda i: went[i])[:10]

    # overlay: the first clinical entity in the generated text (deterministic)
    focus = cand_idx[0] if cand_idx else min(range(len(words)), key=lambda i: went[i])
    cols = [t for t, tk in enumerate(toks) if words[focus][:4].lower() in tk.lower()]
    prof = col[:, cols].mean(dim=1) if cols else col.mean(dim=1)
    prof = prof / (prof.sum() + 1e-9)
    mass = prof[:h * w].reshape(1, 1, h, w)
    mass = torch.nn.functional.interpolate(mass, size=(img.height, img.width),
                                           mode="bilinear")[0, 0].cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(6.2, 2.2), gridspec_kw={"width_ratios": [0.8, 0.8, 1.45]})
    fig.subplots_adjust(wspace=0.16)  # keep the two X-ray panels close together
    axes[0].imshow(img, cmap="gray"); axes[0].set_title("Input", fontsize=7.5, pad=3)
    axes[1].imshow(img, cmap="gray")
    axes[1].imshow(mass, alpha=0.5, cmap="inferno")
    axes[1].set_title('Coupling of "%s"' % words[focus][:12], fontsize=7.5, pad=3)
    sel = top[:8]
    names = [words[i][:14] for i in sel]
    vals = [went[i] for i in sel]  # spatial entropy: lower = more localized
    vmax = max(vals) if max(vals) > 1e-9 else 1.0
    order = sorted(range(len(names)), key=lambda j: vals[j])
    axes[2].barh([names[j] for j in order], [vals[j] for j in order],
                 color=plt.cm.inferno([0.35 + 0.6 * vals[j] / vmax for j in order]),
                 height=0.62, edgecolor="none")
    axes[2].set_title("Most localized tokens", fontsize=7.5, pad=3)
    axes[2].set_xlabel("profile entropy (lower = more localized)", fontsize=7)
    axes[2].tick_params(axis="y", labelsize=7.5, length=0)
    axes[2].tick_params(axis="x", labelsize=6.5)
    axes[2].grid(axis="x", alpha=0.25, lw=0.5)
    for sp in ["top", "right", "left"]:
        axes[2].spines[sp].set_visible(False)
    axes[2].spines["bottom"].set_linewidth(0.6)
    for ax in axes[:2]:
        ax.axis("off")
    fig.canvas.draw()
    pos1 = axes[1].get_position()   # image axes (aspect-locked) defines the band
    pos2 = axes[2].get_position()
    x0 = pos1.x1 + 0.115            # gutter between images and bar chart = word labels
    axes[2].set_position([x0, pos1.y0, min(pos2.x1, 0.995) - x0, pos1.height])
    fig.savefig("/home/deployer/otlora/paper/figs/transport_viz.pdf", bbox_inches="tight", dpi=200)
    print("VIZ_DONE")


if __name__ == "__main__":
    main()
