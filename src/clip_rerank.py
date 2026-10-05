#!/usr/bin/env python3
"""External-aligner baseline (review4 item 9): rerank candidates by a
pretrained medical CLIP's image-text similarity (X-REM's recipe).
BiomedCLIP is not reachable from this network; we use PubMedCLIP
(flaviagiammarino/pubmed-clip-vit-base-patch32, transformers CLIPModel).
800-item subset, same candidates. Writes {tag}_medclip_pred.jsonl (+evals).
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="flaviagiammarino/pubmed-clip-vit-base-patch32")
    ap.add_argument("--dataset", default="mimic_mlf")
    ap.add_argument("--data_root", default="/home/deployer/otlora/data/processed")
    ap.add_argument("--cands_tag", default="sft42_k8")
    args = ap.parse_args()

    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(args.model, torch_dtype=torch.float16).cuda().eval()
    processor = CLIPProcessor.from_pretrained(args.model)

    base = os.path.join(args.data_root, args.dataset)
    rows = [json.loads(l) for l in open(os.path.join(base, f"{args.cands_tag}_cands.jsonl"))]
    id2item = {it["id"]: it for it in map(json.loads, open(os.path.join(base, "test.jsonl")))}

    picks = []
    for s in range(0, len(rows), 2):
        chunk = rows[s:s + 2]
        imgs, txts, cand_ids = [], [], []
        for bi, r in enumerate(chunk):
            cands = [r["greedy"]] + r["cands"]
            img = Image.open(id2item[r["id"]]["image_path"]).convert("RGB")
            for ci, c in enumerate(cands):
                imgs.append(img)
                txts.append(c[:300])
                cand_ids.append((bi, ci))
        enc = processor(text=txts, images=imgs, return_tensors="pt", padding=True,
                        truncation=True, max_length=77).to("cuda:0")
        out = model(**enc)
        sims = out.logits_per_image.diag()  # [B] aligned (image_i, text_i)
        for bi, r in enumerate(chunk):
            cands = [r["greedy"]] + r["cands"]
            idx = [si for si, (b, _) in enumerate(cand_ids) if b == bi]
            best = max(idx, key=lambda si: sims[si].item())
            picks.append(cands[cand_ids[best][1]])
        if (s // 2) % 50 == 0:
            print(f"scored {s + len(chunk)}/{len(rows)}", flush=True)

    p = os.path.join(base, f"{args.cands_tag}_medclip_pred.jsonl")
    with open(p, "w") as f:
        for r, pr in zip(rows, picks):
            f.write(json.dumps({"id": r["id"], "gt": r["gt"], "pred": pr}) + "\n")

    from eval_metrics import compute_nlg
    from chexpert_label import clinical_f1_from_pairs
    compute_nlg(p, p.replace(".jsonl", "_nlg.json"))
    clinical_f1_from_pairs([json.loads(l) for l in open(p)], p.replace(".jsonl", "_ce2.json"))
    n, c = json.load(open(p.replace(".jsonl", "_nlg.json"))), \
        json.load(open(p.replace(".jsonl", "_ce2.json")))
    print(f"MEDCLIP B1={n['BLEU-1']:.3f} CEma={c['F1_macro']:.3f} uniq={n['unique_reports']}")


if __name__ == "__main__":
    main()
