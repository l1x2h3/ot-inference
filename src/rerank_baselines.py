#!/usr/bin/env python3
"""Non-OT selection baselines from an existing candidate file (review control):
is Sinkhorn OT actually needed, or does a simple similarity do the job?

  mbr       - MBD-style: pick the candidate with max mean 1-gram-F1 to the
              other candidates (consensus decoding, no image, no model)
  cos_pool  - cosine between mean-pooled vision patch features and mean-pooled
              candidate token hidden states (same frozen features OT uses,
              but no transport geometry, no marginals, no Sinkhorn)

Writes {tag}_mbr_pred.jsonl / {tag}_cospool_pred.jsonl (+ eval JSONs).
"""
import argparse
import json
import os
import sys
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(__file__))

PROMPT = ("You are a radiologist. Read the chest X-ray and write the findings "
          "section of the radiology report.")


def ngram_f1(a, b):
    ca, cb = Counter(a.lower().split()), Counter(b.lower().split())
    if not ca or not cb:
        return 0.0
    ov = sum((ca & cb).values())
    p, r = ov / sum(ca.values()), ov / sum(cb.values())
    return 2 * p * r / (p + r) if p + r else 0.0


def mbr_pick(cands):
    best, best_s = 0, -1
    for i, c in enumerate(cands):
        s = sum(ngram_f1(c, d) for j, d in enumerate(cands) if j != i)
        if s > best_s:
            best, best_s = i, s
    return best


@torch.no_grad()
def cosine_pass(rows, id2item, model, processor, args):
    """Score every (item, candidate) by pooled cosine; also returns per-cand
    mean log-prob so len-normalized logprob comes for free."""
    from PIL import Image
    picks_cos, picks_lp = [], []
    for s in range(0, len(rows), args.chunk):
        chunk = rows[s:s + args.chunk]
        texts, imgs, owners, cand_idx = [], [], [], []
        for bi, r in enumerate(chunk):
            cands = [r["greedy"]] + r["cands"]
            for ci, c in enumerate(cands):
                msgs = [{"role": "user", "content": [
                    {"type": "image", "image": r.get("image_path", "")},
                    {"type": "text", "text": PROMPT}]}]
                prompt = processor.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True)
                texts.append(prompt + c + "<|im_end|>")
                owners.append(bi)
                cand_idx.append(ci)
                imgs.append(Image.open(id2item[r["id"]]["image_path"]).convert("RGB"))
        enc = processor(text=texts, images=imgs, padding=True, truncation=True,
                        max_length=768, return_tensors="pt").to("cuda:0")
        out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                    pixel_values=enc.pixel_values, image_grid_thw=enc.image_grid_thw,
                    output_hidden_states=True)
        hidden = out.hidden_states[-1].float()
        lens = enc.attention_mask.sum(dim=1)
        grid = enc.image_grid_thw
        per = (grid.prod(dim=1) // 4).tolist()
        vout = model.base_model.model.visual(enc.pixel_values, grid_thw=grid).float()
        vfeats = torch.split(vout, per, dim=0)

        cos_s, lp_s = {}, {}
        for si in range(len(texts)):
            n = int(lens[si])
            tail = hidden[si, max(0, n - 200):n]
            hbar = tail.mean(dim=0)
            vbar = vfeats[owners[si]][:256].mean(dim=0)
            key = (owners[si], cand_idx[si])
            cos_s[key] = torch.nn.functional.cosine_similarity(
                hbar, vbar, dim=0).item()
            lg = out.logits[si, max(0, n - 200) - 1:n - 1].float()
            tg = enc.input_ids[si, max(0, n - 200):n]
            lps = torch.log_softmax(lg, dim=-1).gather(1, tg.unsqueeze(1)).squeeze(1)
            lp_s[key] = lps.mean().item()
        for bi, r in enumerate(chunk):
            cands = [r["greedy"]] + r["cands"]
            ks = [(o, c) for (o, c) in cos_s if o == bi]
            picks_cos.append(cands[max(ks, key=lambda k: cos_s[k])[1]])
            picks_lp.append(cands[max(ks, key=lambda k: lp_s[k])[1]])
        if (s // args.chunk) % 20 == 0:
            print(f"cos scored {s + len(chunk)}/{len(rows)}", flush=True)
    return picks_cos, picks_lp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="models/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--adapter", default="results/sft_s42/adapter_ep1")
    ap.add_argument("--dataset", default="mimic_mlf")
    ap.add_argument("--data_root", default="data/processed")
    ap.add_argument("--split", default="test")
    ap.add_argument("--cands_tag", default="sft42_k8")
    ap.add_argument("--chunk", type=int, default=4)
    args = ap.parse_args()

    base = os.path.join(args.data_root, args.dataset)
    rows = [json.loads(l) for l in open(os.path.join(base, f"{args.cands_tag}_cands.jsonl"))]
    id2item = {it["id"]: it for it in map(
        json.loads, open(os.path.join(base, f"{args.split}.jsonl")))}

    # 1) MBR (pure text consensus, CPU)
    mbr = []
    for r in rows:
        cands = [r["greedy"]] + r["cands"]
        mbr.append(cands[mbr_pick(cands)])
    print("MBR_DONE", len(mbr))

    # 2) pooled cosine + len-norm logprob (GPU)
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel
    processor = AutoProcessor.from_pretrained(args.model_path, max_pixels=512 * 512)
    processor.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    cospool, lp_len = cosine_pass(rows, id2item, model, processor, args)

    for strat, preds in [("mbr", mbr), ("cospool", cospool), ("lp_len", lp_len)]:
        p = os.path.join(base, f"{args.cands_tag}_{strat}_pred.jsonl")
        with open(p, "w") as f:
            for r, pr in zip(rows, preds):
                f.write(json.dumps({"id": r["id"], "gt": r["gt"], "pred": pr}) + "\n")
        print("WROTE", strat)

    from eval_metrics import compute_nlg
    from chexpert_label import clinical_f1_from_pairs
    for strat in ["mbr", "cospool", "lp_len"]:
        p = os.path.join(base, f"{args.cands_tag}_{strat}_pred.jsonl")
        nj = os.path.join(base, f"{args.cands_tag}_{strat}_pred_nlg.json")
        cj = os.path.join(base, f"{args.cands_tag}_{strat}_pred_ce2.json")
        compute_nlg(p, nj)
        clinical_f1_from_pairs([json.loads(l) for l in open(p)], cj)
        n, c = json.load(open(nj)), json.load(open(cj))
        print("EVAL", strat, "B1=%.3f CEma=%.3f uniq=%s"
              % (n["BLEU-1"], c["F1_macro"], n["unique_reports"]))


if __name__ == "__main__":
    main()
