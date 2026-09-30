#!/usr/bin/env python3
"""Phase 1 of OT reranking: sample k candidate reports per test image (+greedy).

Usage: python sample_k.py --adapter .../adapter_ep1 --dataset mimic_mlf \
           --max_items 800 --k 8 --tag sft42_k8
"""
import argparse
import json
import os

import torch

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
    ap.add_argument("--max_items", type=int, default=800)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel

    processor = AutoProcessor.from_pretrained(args.model_path, max_pixels=512 * 512)
    processor.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    items = [json.loads(l) for l in open(
        os.path.join(args.data_root, args.dataset, f"{args.split}.jsonl"))]
    if args.max_items:
        items = items[:args.max_items]

    out_path = os.path.join(args.data_root, args.dataset, f"{args.tag}_cands.jsonl")
    with open(out_path, "w") as out:
        for s in range(0, len(items), args.bs):
            chunk = items[s:s + args.bs]
            imgs = [Image.open(it["image_path"]).convert("RGB") for it in chunk]
            msgs = [[{"role": "user", "content": [
                {"type": "image", "image": it["image_path"]},
                {"type": "text", "text": PROMPT}]}] for it in chunk]
            texts = [processor.apply_chat_template(m, tokenize=False,
                                                   add_generation_prompt=True) for m in msgs]
            enc = processor(text=texts, images=imgs, padding=True,
                            return_tensors="pt").to("cuda:0")
            # greedy
            gen = model.generate(**enc, max_new_tokens=240, do_sample=False,
                                 repetition_penalty=1.05)
            greedy = processor.batch_decode(gen[:, enc.input_ids.shape[1]:],
                                            skip_special_tokens=True)
            # k samples
            gen_s = model.generate(**enc, max_new_tokens=240, do_sample=True,
                                   temperature=0.8, top_p=0.95, num_return_sequences=args.k,
                                   repetition_penalty=1.05)
            cands = processor.batch_decode(gen_s[:, enc.input_ids.shape[1]:],
                                           skip_special_tokens=True)
            for bi, it in enumerate(chunk):
                ks = cands[bi * args.k:(bi + 1) * args.k]
                out.write(json.dumps({"id": it["id"], "gt": it["report"],
                                      "greedy": greedy[bi], "cands": ks}) + "\n")
            if (s // args.bs) % 10 == 0:
                print(f"sampled {s + len(chunk)}/{len(items)}", flush=True)
    print("WROTE", out_path)


if __name__ == "__main__":
    main()
