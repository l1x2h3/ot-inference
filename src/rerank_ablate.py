#!/usr/bin/env python3
"""Geometry ablations for the OT selection score (review4 item 10), 800-item subset.

Variants (same candidates, one forward pass, training-free):
  sharedp     V = vision-merger output (LLM input layer), H = last-layer tail,
              SINGLE shared JL matrix P for both modalities (vs. independent
              P_v, P_t in the default). Independent random matrices make the
              cost a random bilinear form; a shared P preserves genuine
              v-h overlap.
  lastv       V = last-layer hidden states at the image-token positions
              (same layer as H), shared P. Removes the layer mismatch.
  otnorm      default geometry, but scores are divided by the candidate's
              token count before ranking (length-normalization control).

Also reports the OT-score/candidate-length correlation.
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


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="/home/deployer/otlora/models/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--adapter", default="/home/deployer/otlora/results/sft_s42/adapter_ep1")
    ap.add_argument("--dataset", default="mimic_mlf")
    ap.add_argument("--data_root", default="/home/deployer/otlora/data/processed")
    ap.add_argument("--cands_tag", default="sft42_k8")
    args = ap.parse_args()

    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel

    processor = AutoProcessor.from_pretrained(args.model_path, max_pixels=512 * 512)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    d_model = model.config.hidden_size
    img_tok = getattr(model.config, "image_token_id", 151655)

    ots = {
        "default": CAOTLoss(d_vision=d_model, d_text=d_model, token_marginal="salience",
                            cost_type="l2_mean", balanced=True, proj_mode="fixed").cuda(),
        "sharedp": CAOTLoss(d_vision=d_model, d_text=d_model, token_marginal="salience",
                            cost_type="l2_mean", balanced=True, proj_mode="fixed_shared").cuda(),
        "lastv": CAOTLoss(d_vision=d_model, d_text=d_model, token_marginal="salience",
                          cost_type="l2_mean", balanced=True, proj_mode="fixed_shared").cuda(),
    }

    base = os.path.join(args.data_root, args.dataset)
    rows = [json.loads(l) for l in open(os.path.join(base, f"{args.cands_tag}_cands.jsonl"))]
    id2item = {it["id"]: it for it in map(json.loads, open(os.path.join(base, "test.jsonl")))}

    picks = {k: [] for k in ["default", "sharedp", "lastv", "otnorm"]}
    len_ot_pairs = []  # (n_tokens, ot_cost) for correlation
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
        enc = processor(text=seq_texts, images=imgs, padding=True, truncation=True,
                        max_length=768, return_tensors="pt").to("cuda:0")
        out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                    pixel_values=enc.pixel_values, image_grid_thw=enc.image_grid_thw,
                    output_hidden_states=True)
        tok_lens = enc.attention_mask.sum(dim=1)
        hidden = out.hidden_states[-1].float()
        visual = model.base_model.model.visual
        vout = visual(enc.pixel_values, grid_thw=enc.image_grid_thw).float()
        per = (enc.image_grid_thw.prod(dim=1) // 4).tolist()
        vfeats = torch.split(vout, per, dim=0)

        import statistics
        for bi, r in enumerate(chunk):
            cands = [r["greedy"]] + r["cands"]
            idx = [si for si in range(len(seq_texts)) if owners[si] == bi]
            scores = {v: {} for v in ots}
            nt = {}
            for si in idx:
                n = int(tok_lens[si])
                tail = hidden[si, max(0, n - 200):n]
                v_in = vfeats[owners[si]][:256]
                # last-layer states at the image-token positions
                img_pos = (enc.input_ids[si] == img_tok).nonzero(as_tuple=True)[0]
                v_last = hidden[si][img_pos[:256]] if img_pos.numel() >= 4 else v_in
                nt[cand_idx[si]] = tail.size(0)
                if tail.size(0) < 4 or v_in.size(0) < 4 or v_last.size(0) < 4:
                    for name in ots:
                        scores[name][cand_idx[si]] = 1e9
                    continue
                lv, _ = ots["default"](v_in.unsqueeze(0), tail.unsqueeze(0))
                scores["default"][cand_idx[si]] = float(lv)
                lv, _ = ots["sharedp"](v_in.unsqueeze(0), tail.unsqueeze(0))
                scores["sharedp"][cand_idx[si]] = float(lv)
                lv, _ = ots["lastv"](v_last.unsqueeze(0), tail.unsqueeze(0))
                scores["lastv"][cand_idx[si]] = float(lv)
                len_ot_pairs.append((tail.size(0), scores["default"][cand_idx[si]]))
            for name in ["default", "sharedp", "lastv"]:
                best = min(scores[name], key=scores[name].get)
                picks[name].append(cands[best])
            norm = {ci: scores["default"][ci] / max(1, nt[ci]) for ci in scores["default"]}
            picks["otnorm"].append(cands[min(norm, key=norm.get)])
        if (s // 4) % 25 == 0:
            print(f"scored {s + len(chunk)}/{len(rows)}", flush=True)

    for strat, preds in picks.items():
        with open(os.path.join(base, f"{args.cands_tag}_abl_{strat}_pred.jsonl"), "w") as f:
            for r, p in zip(rows, preds):
                f.write(json.dumps({"id": r["id"], "gt": r["gt"], "pred": p}) + "\n")
        print("WROTE", strat)

    import math
    n = len(len_ot_pairs)
    mx = sum(x for x, _ in len_ot_pairs) / n
    my = sum(y for _, y in len_ot_pairs) / n
    cov = sum((x - mx) * (y - my) for x, y in len_ot_pairs) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x, _ in len_ot_pairs) / n)
    sy = math.sqrt(sum((y - my) ** 2 for _, y in len_ot_pairs) / n)
    print(f"OT-vs-length Pearson r = {cov / (sx * sy):+.4f}  (n={n})")
    json.dump({"pearson_r": cov / (sx * sy), "n": n},
              open(os.path.join(base, f"{args.cands_tag}_abl_corr.json"), "w"))


if __name__ == "__main__":
    main()
