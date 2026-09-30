#!/usr/bin/env python3
"""Horizontal comparison of OT variants as reranking scores (shared forward pass).

Variants (all training-free, same candidates):
  bal_l2_sal   balanced entropic OT, l2_mean cost, salience marginals  (default)
  unif         balanced, uniform token marginals
  gated        hard-gated marginals (entity tokens only)
  unb_t08      unbalanced (tau=0.8)
  unb_t05      unbalanced (tau=0.5)
  cos          cosine cost (salience marginals)
Each variant -> argmin-cost pick; also writes per-candidate scores for reuse.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from ot_loss import CAOTLoss  # noqa: E402

PROMPT = ("You are a radiologist. Read the chest X-ray and write the findings "
          "section of the radiology report.")

VARIANTS = {
    "bal_l2_sal": dict(token_marginal="salience", cost_type="l2_mean", balanced=True),
    "unif": dict(token_marginal="uniform", cost_type="l2_mean", balanced=True),
    "gated": dict(token_marginal="gated", cost_type="l2_mean", balanced=True),
    "unb_t08": dict(token_marginal="salience", cost_type="l2_mean", balanced=False, unbalanced_tau=0.8),
    "unb_t05": dict(token_marginal="salience", cost_type="l2_mean", balanced=False, unbalanced_tau=0.5),
    "cos": dict(token_marginal="salience", cost_type="cosine", balanced=True),
}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="models/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--dataset", default="mimic_mlf")
    ap.add_argument("--data_root", default="data/processed")
    ap.add_argument("--cands_tag", required=True)
    args = ap.parse_args()

    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel

    processor = AutoProcessor.from_pretrained(args.model_path, max_pixels=512 * 512)
    tokenizer = processor.tokenizer
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    d_model = model.config.hidden_size

    ots = {name: CAOTLoss(d_vision=d_model, d_text=d_model, **kw).cuda()
           for name, kw in VARIANTS.items()}

    base = os.path.join(args.data_root, args.dataset)
    rows = [json.loads(l) for l in open(os.path.join(base, f"{args.cands_tag}_cands.jsonl"))]
    id2item = {it["id"]: it for it in map(
        json.loads, open(os.path.join(base, "test.jsonl")))}

    picks = {k: [] for k in VARIANTS}
    picks["hybrid"] = []
    logprobs_all = []
    for s in range(0, len(rows), 4):
        chunk = rows[s:s + 4]
        seq_texts, owners, cand_idx = [], [], []
        for bi, r in enumerate(chunk):
            cands = [r["greedy"]] + r["cands"]
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": id2item[r["id"]]["image_path"]},
                {"type": "text", "text": PROMPT}]}]
            prompt = processor.apply_chat_template(msgs, tokenize=False,
                                                   add_generation_prompt=True)
            for ci, c in enumerate(cands):
                seq_texts.append(prompt + c + "<|im_end|>")
                owners.append(bi)
                cand_idx.append(ci)
        imgs = [Image.open(id2item[chunk[o]["id"]]["image_path"]).convert("RGB")
                for o in owners]
        # NOTE: to keep hidden states aligned with the real chat format we reuse
        # the chat template as in rerank_ot; plain concat keeps positions similar.
        enc = processor(text=seq_texts, images=imgs, padding=True, truncation=True,
                        max_length=768, return_tensors="pt").to("cuda:0")
        out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                    pixel_values=enc.pixel_values, image_grid_thw=enc.image_grid_thw,
                    output_hidden_states=True)
        tok_lens = enc.attention_mask.sum(dim=1)
        logits = out.logits.float()
        lp = []
        for i in range(len(seq_texts)):
            n = int(tok_lens[i])
            start = max(1, n - 200)
            lg = logits[i, start - 1:n - 1]
            tg = enc.input_ids[i, start:n]
            lp.append(torch.log_softmax(lg, dim=-1).gather(1, tg.unsqueeze(1)).mean().item())
        hidden = out.hidden_states[-1].float()
        visual = model.base_model.model.visual
        vout = visual(enc.pixel_values, grid_thw=enc.image_grid_thw).float()
        per = (enc.image_grid_thw.prod(dim=1) // 4).tolist()
        vfeats = torch.split(vout, per, dim=0)

        import statistics
        for bi, r in enumerate(chunk):
            cands = [r["greedy"]] + r["cands"]
            idx = [si for si in range(len(seq_texts)) if owners[si] == bi]
            scores = {v: {} for v in VARIANTS}
            for si in idx:
                n = int(tok_lens[si])
                tail = hidden[si, max(0, n - 200):n]
                v = vfeats[owners[si]][:256]
                if tail.size(0) < 4 or v.size(0) < 4:
                    for name in VARIANTS:
                        scores[name][cand_idx[si]] = 1e9
                    continue
                for name, crit in ots.items():
                    lv, _ = crit(v.unsqueeze(0), tail.unsqueeze(0))
                    scores[name][cand_idx[si]] = float(lv)
            for name in VARIANTS:
                best = min(scores[name], key=scores[name].get)
                picks[name].append(cands[best])
            lps = [lp[si] for si in idx]
            ml, sl = statistics.mean(lps), (statistics.pstdev(lps) or 1e-9)
            otv = [scores["bal_l2_sal"][cand_idx[si]] for si in idx]
            mo, so = statistics.mean(otv), (statistics.pstdev(otv) or 1e-9)
            hyb = {cand_idx[si]: 0.5 * (-(scores["bal_l2_sal"][cand_idx[si]] - mo) / so)
                   + 0.5 * ((lp[si] - ml) / sl) for si in idx}
            picks["hybrid"].append(cands[max(hyb, key=hyb.get)])
        if (s // 4) % 25 == 0:
            print(f"scored {s + len(chunk)}/{len(rows)}", flush=True)

    for strat, preds in picks.items():
        with open(os.path.join(base, f"{args.cands_tag}_var_{strat}_pred.jsonl"), "w") as f:
            for r, p in zip(rows, preds):
                f.write(json.dumps({"id": r["id"], "gt": r["gt"], "pred": p}) + "\n")
        print("WROTE", strat)


if __name__ == "__main__":
    main()
