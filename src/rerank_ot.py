#!/usr/bin/env python3
"""Phase 2 of OT reranking: score candidates with fixed-projection OT alignment
and write best-candidate prediction files for several reranking strategies.

Strategies compared (all training-free):
  greedy            - the model's greedy output (no reranking)
  random            - uniform random candidate (sanity floor)
  ot                - min transport cost (vision patches <-> candidate token states)
  ot_len            - OT cost + length prior toward the corpus median length
  logprob           - mean token log-prob (standard reranker)
"""
import argparse
import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from ot_loss import CAOTLoss  # noqa: E402

PROMPT = ("You are a radiologist. Read the chest X-ray and write the findings "
          "section of the radiology report.")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="models/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--dataset", default="mimic_mlf")
    ap.add_argument("--data_root", default="data/processed")
    ap.add_argument("--split", default="test")
    ap.add_argument("--cands_tag", required=True, help="tag used by sample_k.py")
    ap.add_argument("--median_len", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=4)
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
    ot = CAOTLoss(d_vision=d_model, d_text=d_model, token_marginal="salience").cuda()

    src = os.path.join(args.data_root, args.dataset, f"{args.cands_tag}_cands.jsonl")
    rows = [json.loads(l) for l in open(src)]
    rng = random.Random(0)

    picks = {k: [] for k in ["greedy", "random", "ot", "ot_len", "logprob", "hybrid"]}
    score_dump = []
    for s in range(0, len(rows), args.chunk):
        chunk = rows[s:s + args.chunk]
        # flatten: item x candidate -> batch of sequences
        seq_texts, seq_imgs, owners, cand_idx = [], [], [], []
        for bi, r in enumerate(chunk):
            cands = [r["greedy"]] + r["cands"]
            for ci, c in enumerate(cands):
                msgs = [{"role": "user", "content": [
                    {"type": "image", "image": r.get("image_path", "")},
                    {"type": "text", "text": PROMPT}]}]
                prompt = processor.apply_chat_template(msgs, tokenize=False,
                                                       add_generation_prompt=True)
                seq_texts.append(prompt + c + "<|im_end|>")
                owners.append(bi)
                cand_idx.append(ci)
        # images: need per-row PIL for processor (paths not in cands file; use id lookup)
        # -> sample_k did not store image paths; rebuild from split file
        id2item = {it["id"]: it for it in map(json.loads, open(
            os.path.join(args.data_root, args.dataset, f"{args.split}.jsonl")))}
        seq_imgs = [Image.open(id2item[chunk[o]["id"]]["image_path"]).convert("RGB")
                    for o in owners]

        enc = processor(text=seq_texts, images=seq_imgs, padding=True,
                        truncation=True, max_length=768, return_tensors="pt").to("cuda:0")
        out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                    pixel_values=enc.pixel_values, image_grid_thw=enc.image_grid_thw,
                    output_hidden_states=True)
        # log-prob of candidate tokens (shifted)
        logits = out.logits  # bf16; per-slice fp32 below
        lp_sum, lp_cnt = torch.zeros(len(seq_texts)), torch.zeros(len(seq_texts))
        tok_lens = enc.attention_mask.sum(dim=1)
        # approximate: score whole right part after left padding
        for i in range(len(seq_texts)):
            n = int(tok_lens[i])
            # last tok_lens-? — candidate region is roughly the tail; use
            # everything after prompt: we don't store prompt len here, so use
            # the tail = min(cand tokens est) — approximate with last 200 tokens
            start = max(1, n - 200)
            lg = logits[i, start - 1:n - 1].float()
            tg = enc.input_ids[i, start:n]
            lps = torch.log_softmax(lg, dim=-1).gather(1, tg.unsqueeze(1)).squeeze(1)
            lp_sum[i] = lps.mean()
            lp_cnt[i] = 1

        # vision features per unique image (per chunk row)
        uniq = {o: seq_imgs[si] for si, o in enumerate(owners)}
        grid = enc.image_grid_thw
        per = (grid.prod(dim=1) // 4).tolist()
        visual = model.base_model.model.visual
        vout = visual(enc.pixel_values, grid_thw=grid).float()
        vfeats = torch.split(vout, per, dim=0)

        # OT score per sequence (text states = tail tokens' last hidden)
        hidden = out.hidden_states[-1].float()
        ot_scores = torch.zeros(len(seq_texts))
        for si in range(len(seq_texts)):
            n = int(tok_lens[si])
            tail = hidden[si, max(0, n - 200):n]  # candidate region (approx)
            v = vfeats[owners[si]][:256]
            if tail.size(0) < 4 or v.size(0) < 4:
                ot_scores[si] = 1e9
                continue
            loss, _ = ot(v.unsqueeze(0), tail.unsqueeze(0))
            ot_scores[si] = loss.item()

        for bi, r in enumerate(chunk):
            cands = [r["greedy"]] + r["cands"]
            idx = [si for si in range(len(seq_texts)) if owners[si] == bi]
            ot_s = {cand_idx[si]: float(ot_scores[si]) for si in idx}
            lp_s = {cand_idx[si]: lp_sum[si].item() for si in idx}
            # z-normalize the two signals within the candidate set, then blend
            import statistics
            ots = list(ot_s.values()); lps = list(lp_s.values())
            mo, so = statistics.mean(ots), (statistics.pstdev(ots) or 1e-9)
            ml, sl = statistics.mean(lps), (statistics.pstdev(lps) or 1e-9)
            hyb = {c: 0.5 * (-(ot_s[c] - mo) / so) + 0.5 * ((lp_s[c] - ml) / sl)
                   for c in ot_s}
            picks["greedy"].append(cands[0])
            picks["random"].append(rng.choice(cands[1:]) if len(cands) > 1 else cands[0])
            ot_best = min(ot_s, key=ot_s.get)
            picks["ot"].append(cands[ot_best])
            ot_len_best = min(ot_s, key=lambda c: ot_s[c]
                              + 0.01 * abs(len(cands[c].split()) - args.median_len))
            picks["ot_len"].append(cands[ot_len_best])
            picks["logprob"].append(cands[max(lp_s, key=lp_s.get)])
            picks["hybrid"].append(cands[max(hyb, key=hyb.get)])
            score_dump.append({"id": r["id"], "ot": ot_s, "lp": lp_s})
        if (s // 4) % 20 == 0:
            print(f"scored {s + len(chunk)}/{len(rows)}", flush=True)

    base = os.path.join(args.data_root, args.dataset)
    with open(os.path.join(base, f"{args.cands_tag}_scores.json"), "w") as f:
        for r in score_dump:
            f.write(json.dumps(r) + "\n")
    for strat, preds in picks.items():
        with open(os.path.join(base, f"{args.cands_tag}_{strat}_pred.jsonl"), "w") as f:
            for r, p in zip(rows, preds):
                f.write(json.dumps({"id": r["id"], "gt": r["gt"], "pred": p}) + "\n")
        print("WROTE", strat)


if __name__ == "__main__":
    main()
