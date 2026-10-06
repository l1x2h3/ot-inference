#!/usr/bin/env python3
"""review5 rescoring sweep (one shared forward pass over the 800-item subset).

Fixes three deployed-configuration issues flagged in review5:
  - true salience marginals (the deployed scorer never passed token_salience,
    so all published rows effectively used uniform marginals)
  - ALL visual tokens (deployed scorer kept only the first 256 of ~324;
    on chest X-rays the dropped band is the lung bases)
  - shared projection as default candidate (better AND principled)

Variants written (argmin-score picks; hybrid = z-blend with mean log-prob):
  uni_256   uniform marginal,  256 vtok, independent P   (= deployed default)
  sal_256   salience marginal, 256 vtok, independent P
  sal_all   salience marginal, all vtok,  independent P
  sal_sh_all salience marginal, all vtok, shared P
  hyb_sh_all hybrid of sal_sh_all with log-prob
Per-candidate scores + lengths are dumped for offline analysis.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from ot_loss import CAOTLoss  # noqa: E402
from salience import token_salience  # noqa: E402

PROMPT = ("You are a radiologist. Read the chest X-ray and write the findings "
          "section of the radiology report.")
IMGTOK = 151655


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="/home/deployer/otlora/models/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--adapter", default="/home/deployer/otlora/results/sft_s42/adapter_ep1")
    ap.add_argument("--dataset", default="mimic_mlf")
    ap.add_argument("--data_root", default="/home/deployer/otlora/data/processed")
    ap.add_argument("--cands_tag", default="sft42_k8")
    ap.add_argument("--out_tag", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--old_window", action="store_true",
                    help="deployed tail window max(0,n-200) incl. prompt tokens")
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

    torch.manual_seed(args.seed)
    variants = {
        "uni_256": dict(token_marginal="uniform", proj_mode="fixed"),
        "sal_256": dict(token_marginal="salience", proj_mode="fixed"),
        "sal_all": dict(token_marginal="salience", proj_mode="fixed"),
        "sal_sh_all": dict(token_marginal="salience", proj_mode="fixed_shared"),
    }
    ots = {k: CAOTLoss(d_vision=d_model, d_text=d_model, **kw).cuda()
           for k, kw in variants.items()}

    base = os.path.join(args.data_root, args.dataset)
    rows = [json.loads(l) for l in open(os.path.join(base, f"{args.cands_tag}_cands.jsonl"))]
    id2item = {it["id"]: it for it in map(json.loads, open(os.path.join(base, "test.jsonl")))}

    keys = list(variants) + ["hyb_sh_all"]
    picks = {k: [] for k in keys}
    dump = []
    for s in range(0, len(rows), 2):
        chunk = rows[s:s + 2]
        seq_texts, owners, cand_idx, cand_text = [], [], [], []
        for bi, r in enumerate(chunk):
            cands = [r["greedy"]] + r["cands"]
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": id2item[r["id"]]["image_path"]},
                {"type": "text", "text": PROMPT}]}]
            prompt = processor.apply_chat_template(msgs, tokenize=False,
                                                   add_generation_prompt=True)
            for ci, c in enumerate(cands):
                seq_texts.append(prompt + c + "<|im_end|>")
                owners.append(bi); cand_idx.append(ci); cand_text.append(c)
        imgs = [Image.open(id2item[chunk[o]["id"]]["image_path"]).convert("RGB")
                for o in owners]
        enc = processor(text=seq_texts, images=imgs, padding=True, truncation=True,
                        max_length=768, return_tensors="pt").to("cuda:0")
        out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                    pixel_values=enc.pixel_values, image_grid_thw=enc.image_grid_thw,
                    output_hidden_states=True)
        tok_lens = enc.attention_mask.sum(dim=1)
        logits = out.logits.float()
        hidden = out.hidden_states[-1].float()
        vout = model.base_model.model.visual(
            enc.pixel_values, grid_thw=enc.image_grid_thw).float()
        per = (enc.image_grid_thw.prod(dim=1) // 4).tolist()
        vfeats = torch.split(vout, per, dim=0)

        import statistics
        for bi, r in enumerate(chunk):
            cands = [r["greedy"]] + r["cands"]
            idx = [si for si in range(len(seq_texts)) if owners[si] == bi]
            scores = {v: {} for v in variants}
            lp_s, ntok = {}, {}
            # prompt token length: identical across this item's sequences
            p0 = idx[0]
            n0 = int(tok_lens[p0])
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": id2item[r["id"]]["image_path"]},
                {"type": "text", "text": PROMPT}]}]
            pr_text = processor.apply_chat_template(msgs, tokenize=False,
                                                    add_generation_prompt=True)
            # count prompt tokens via image-token expansion in the encoded seq:
            img_pos = (enc.input_ids[p0] == IMGTOK).nonzero(as_tuple=True)[0]
            plen = int(img_pos[-1].item()) + 1 if img_pos.numel() else n0 - 200

            for si in idx:
                n = int(tok_lens[si])
                start = max(0, n - 200) if args.old_window else max(plen, n - 200)
                tail_ids = enc.input_ids[si, start:n]
                tail = hidden[si, start:n]
                lg = logits[si, start - 1:n - 1]
                tg = enc.input_ids[si, start:n]
                lp = torch.log_softmax(lg, dim=-1).gather(1, tg.unsqueeze(1)).mean().item()
                sal = torch.tensor(token_salience(
                    processor.tokenizer, tail_ids.tolist(), cand_text[si]),
                    device="cuda:0", dtype=torch.float32)
                v_all = vfeats[owners[si]]
                ntok[cand_idx[si]] = int(tail.size(0))
                if tail.size(0) < 4 or v_all.size(0) < 4:
                    for name in variants:
                        scores[name][cand_idx[si]] = 1e9
                    lp_s[cand_idx[si]] = lp
                    continue
                for name, kw in variants.items():
                    v_in = v_all[:256] if name.endswith("256") else v_all
                    sal_in = None if name.startswith("uni") else sal.unsqueeze(0)
                    with torch.no_grad():
                        lv, _ = ots[name](v_in.unsqueeze(0), tail.unsqueeze(0),
                                          token_salience=sal_in)
                    scores[name][cand_idx[si]] = float(lv)
                lp_s[cand_idx[si]] = lp
            for name in variants:
                best = min(scores[name], key=scores[name].get)
                picks[name].append(cands[best])
            otv = [scores["sal_sh_all"][ci] for ci in sorted(scores["sal_sh_all"])]
            lps = [lp_s[ci] for ci in sorted(lp_s)]
            mo, so = statistics.mean(otv), (statistics.pstdev(otv) or 1e-9)
            ml, sl = statistics.mean(lps), (statistics.pstdev(lps) or 1e-9)
            hyb = {ci: 0.5 * (-(scores["sal_sh_all"][ci] - mo) / so)
                   + 0.5 * ((lp_s[ci] - ml) / sl) for ci in lp_s}
            picks["hyb_sh_all"].append(cands[max(hyb, key=hyb.get)])
            dump.append({"id": r["id"],
                         "scores": {k: {str(ci): v for ci, v in scores[k].items()}
                                    for k in variants},
                         "lp": {str(ci): v for ci, v in lp_s.items()},
                         "ntok": {str(ci): v for ci, v in ntok.items()},
                         "nv": int(v_all.size(0))})
        if (s // 4) % 25 == 0:
            print(f"scored {s + len(chunk)}/{len(rows)}", flush=True)

    ot = args.out_tag or "r5"
    for strat, preds in picks.items():
        with open(os.path.join(base, f"{args.cands_tag}_{ot}_{strat}_pred.jsonl"), "w") as f:
            for r, p in zip(rows, preds):
                f.write(json.dumps({"id": r["id"], "gt": r["gt"], "pred": p}) + "\n")
        print("WROTE", strat)
    with open(os.path.join(base, f"{args.cands_tag}_{ot}_scores.jsonl"), "w") as f:
        for d in dump:
            f.write(json.dumps(d) + "\n")
    print("WROTE scores dump")


if __name__ == "__main__":
    main()
