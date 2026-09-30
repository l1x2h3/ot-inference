#!/usr/bin/env python3
"""Qualitative figure: image with per-patch transport mass overlay + top-coupled tokens."""
import json
import sys

import torch

sys.path.insert(0, "src")
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

    mp = "models/Qwen2.5-VL-3B-Instruct"
    ad = "results/sft_s42/adapter_ep1"
    processor = AutoProcessor.from_pretrained(mp, max_pixels=512 * 512)
    processor.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        mp, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model = PeftModel.from_pretrained(model, ad)
    model.eval()

    # pick an abnormal example: GT mentions pleural effusion
    items = [json.loads(l) for l in open(
        "data/processed/mimic_mlf/test.jsonl")]
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
    patch_mass = plan.sum(dim=1)  # mass received per patch

    # spatial layout: reconstruct grid from grid_thw (assume single image, t*h/2 x w/2)
    thw = full.image_grid_thw[0]
    h, w = int(thw[1]) // 2, int(thw[2]) // 2
    mass = patch_mass[:h * w].reshape(1, 1, h, w)
    mass = torch.nn.functional.interpolate(mass, size=(img.height, img.width),
                                           mode="bilinear")[0, 0].cpu().numpy()

    # top WORDS by received mass (merge BPE fragments; subwords rank high otherwise)
    tok_mass = plan.sum(dim=0).cpu()
    toks = [processor.tokenizer.decode([i]) for i in ids.tolist()]
    words, wmass, cur, curm = [], [], "", []
    for tk, m in zip(toks, tok_mass.tolist()):
        if "<|" in tk or ">" == tk.strip():  # special tokens (e.g. <|im_end|>)
            continue
        if (tk.startswith(" ") or tk.startswith("\n")) and cur.strip():
            words.append(cur.strip())
            wmass.append(sum(curm))
            cur, curm = "", []
        cur += tk
        curm.append(m)
    if cur.strip():
        words.append(cur.strip())
        wmass.append(sum(curm))
    top = sorted(range(len(words)), key=lambda i: -wmass[i])[:10]

    fig, axes = plt.subplots(1, 3, figsize=(6.2, 2.2), gridspec_kw={"width_ratios": [0.8, 0.8, 1.45]})
    fig.subplots_adjust(wspace=0.16)  # keep the two X-ray panels close together
    axes[0].imshow(img, cmap="gray"); axes[0].set_title("Input", fontsize=7.5, pad=3)
    axes[1].imshow(img, cmap="gray")
    axes[1].imshow(mass, alpha=0.5, cmap="inferno")
    axes[1].set_title("OT mass", fontsize=7.5, pad=3)
    sel = [i for i in top if len(words[i]) > 2][:8]
    names = [words[i][:14] for i in sel]
    vals = [float(wmass[i]) for i in sel]
    order = sorted(range(len(names)), key=lambda j: vals[j])
    axes[2].barh([names[j] for j in order], [vals[j] for j in order],
                 color=plt.cm.inferno([0.35 + 0.6 * vals[j] / max(vals) for j in order]),
                 height=0.62, edgecolor="none")
    axes[2].set_title("Top tokens", fontsize=7.5, pad=3)
    axes[2].tick_params(axis="y", labelsize=7.5, length=0)
    axes[2].tick_params(axis="x", labelsize=6.5)
    axes[2].grid(axis="x", alpha=0.25, lw=0.5)
    for sp in ["top", "right", "left"]:
        axes[2].spines[sp].set_visible(False)
    axes[2].spines["bottom"].set_linewidth(0.6)
    for yj, j in enumerate(order):
        axes[2].text(vals[j] + 0.02 * max(vals), yj, f"{vals[j]:.2f}",
                     va="center", fontsize=6)
    for ax in axes[:2]:
        ax.axis("off")
    fig.canvas.draw()
    pos1 = axes[1].get_position()   # image axes (aspect-locked) defines the band
    pos2 = axes[2].get_position()
    x0 = pos1.x1 + 0.115            # gutter between images and bar chart = word labels
    axes[2].set_position([x0, pos1.y0, min(pos2.x1, 0.995) - x0, pos1.height])
    fig.savefig("paper/figs/transport_viz.pdf", bbox_inches="tight", dpi=200)
    print("VIZ_DONE")


if __name__ == "__main__":
    main()
