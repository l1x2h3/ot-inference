#!/usr/bin/env python3
"""Inference-time selection on LLaVA-1.5-7B (second family): sample k=8 + greedy,
score candidates with the same frozen-geometry OT + log-prob hybrid, evaluate.
One script = the whole minimal loop on the 800-item subset."""
import argparse
import json
import os
import random
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from llava_common import D_ROOT, PROMPT, MODEL_PATH, vision_feats  # noqa: E402
from ot_loss import CAOTLoss  # noqa: E402

TAG = "llava_k8"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="results/llava_sft/adapter_ep2")
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--skip_sample", action="store_true")
    args = ap.parse_args()

    from PIL import Image
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from peft import PeftModel

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    tok = processor.tokenizer

    items = [json.loads(l) for l in open(os.path.join(D_ROOT, "mimic_mlf", "test.jsonl"))][:args.n]
    prompt = f"USER: <image>\n{PROMPT} ASSISTANT:"

    # ---- phase 1: greedy + k samples ----
    cands_file = os.path.join(D_ROOT, "mimic_mlf", f"{TAG}_cands.jsonl")
    if args.skip_sample or os.path.exists(cands_file):
        rows = [json.loads(l) for l in open(cands_file)]
    else:
        rows = []
        for s in range(0, len(items), args.bs):
            batch = items[s:s + args.bs]
            imgs = [Image.open(x["image_path"]).convert("RGB") for x in batch]
            enc = processor(text=[prompt] * len(batch), images=imgs,
                            padding=True, return_tensors="pt").to("cuda:0")
            g = model.generate(**enc, max_new_tokens=220, do_sample=False,
                               repetition_penalty=1.05)
            smp = model.generate(**enc, max_new_tokens=220, do_sample=True,
                                 temperature=0.8, top_p=0.95, num_return_sequences=args.k)
            for bi, x in enumerate(batch):
                greedy = tok.decode(g[bi, enc.input_ids.shape[1]:], skip_special_tokens=True)
                cs = []
                for ki in range(args.k):
                    j = bi * args.k + ki
                    cs.append(tok.decode(smp[j, enc.input_ids.shape[1]:],
                                         skip_special_tokens=True))
                rows.append({"id": x["id"], "gt": x["report"], "greedy": greedy,
                             "cands": cs, "image_path": x["image_path"]})
            if (s // args.bs) % 10 == 0:
                print(f"sampled {s + len(batch)}/{len(items)}", flush=True)
        with open(cands_file, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    # ---- phase 2: score all candidates (OT + mean log-prob) ----
    id2img = {x["id"]: x["image_path"] for x in items}
    ot = CAOTLoss(d_vision=1024, d_text=4096, token_marginal="salience").cuda()
    rng = random.Random(0)
    picks = {k: [] for k in ["greedy", "random", "ot", "logprob", "hybrid"]}
    for s in range(0, len(rows), args.chunk):
        chunk = rows[s:s + args.chunk]
        texts, imgs, owners, cidx = [], [], [], []
        for bi, r in enumerate(chunk):
            cands = [r["greedy"]] + r["cands"]
            for ci, c in enumerate(cands):
                texts.append(prompt + c + "</s>")
                owners.append(bi)
                cidx.append(ci)
                imgs.append(Image.open(id2img[r["id"]]).convert("RGB"))
        enc = processor(text=texts, images=imgs, padding=True, truncation=True,
                        max_length=768, return_tensors="pt").to("cuda:0")
        out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                    pixel_values=enc.pixel_values, output_hidden_states=True)
        hidden = out.hidden_states[-1].float()
        vfeats = vision_feats(model, enc.pixel_values)  # list [576,1024] per seq
        lens = enc.attention_mask.sum(dim=1)
        ot_s, lp_s = {}, {}
        for si in range(len(texts)):
            n = int(lens[si])
            tail = hidden[si, max(0, n - 200):n]
            lg = out.logits[si, max(0, n - 200) - 1:n - 1].float()
            tg = enc.input_ids[si, max(0, n - 200):n]
            lps = torch.log_softmax(lg, dim=-1).gather(1, tg.unsqueeze(1)).squeeze(1)
            key = (owners[si], cidx[si])
            lp_s[key] = lps.mean().item()
            if tail.size(0) >= 4:
                loss, _ = ot(vfeats[si].unsqueeze(0), tail.unsqueeze(0))
                ot_s[key] = loss.item()
            else:
                ot_s[key] = 1e9
        for bi, r in enumerate(chunk):
            cands = [r["greedy"]] + r["cands"]
            ks = [(o, c) for (o, c) in ot_s if o == bi]
            ots = [ot_s[k] for k in ks]; lps = [lp_s[k] for k in ks]
            mo, so = statistics.mean(ots), (statistics.pstdev(ots) or 1e-9)
            ml, sl = statistics.mean(lps), (statistics.pstdev(lps) or 1e-9)
            hyb = {k: 0.5 * (-(ot_s[k] - mo) / so) + 0.5 * ((lp_s[k] - ml) / sl)
                   for k in ks}
            picks["greedy"].append(cands[0])
            picks["random"].append(rng.choice(cands[1:]))
            picks["ot"].append(cands[min(ks, key=lambda k: ot_s[k])[1]])
            picks["logprob"].append(cands[max(ks, key=lambda k: lp_s[k])[1]])
            picks["hybrid"].append(cands[max(hyb, key=hyb.get)[1]])
        if (s // args.chunk) % 20 == 0:
            print(f"scored {s + len(chunk)}/{len(rows)}", flush=True)

    from eval_metrics import compute_nlg
    from chexpert_label import clinical_f1_from_pairs
    base = os.path.join(D_ROOT, "mimic_mlf")
    for strat, preds in picks.items():
        p = os.path.join(base, f"{TAG}_{strat}_pred.jsonl")
        with open(p, "w") as f:
            for r, pr in zip(rows, preds):
                f.write(json.dumps({"id": r["id"], "gt": r["gt"], "pred": pr}) + "\n")
        nj = p.replace(".jsonl", "_nlg.json")
        cj = p.replace(".jsonl", "_ce2.json")
        compute_nlg(p, nj)
        clinical_f1_from_pairs([json.loads(l) for l in open(p)], cj)
        n, c = json.load(open(nj)), json.load(open(cj))
        print(f"LLAVA_EVAL {strat:8s} B1={n['BLEU-1']:.3f} B4={n['BLEU-4']:.3f} "
              f"CEma={c['F1_macro']:.3f} uniq={n['unique_reports']}", flush=True)
    print("LLAVA_SELECT_DONE")


if __name__ == "__main__":
    main()
