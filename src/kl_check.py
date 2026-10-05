#!/usr/bin/env python3
"""Direct inference-time evidence for 'absorbed': per-token KL between the
SFT and SFT+OT checkpoints on teacher-forced reference reports (review4 item 12).
If the OT loss had changed the output distribution, KL would be non-trivial.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))

PROMPT = ("You are a radiologist. Read the chest X-ray and write the findings "
          "section of the radiology report.")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="/home/deployer/otlora/models/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--adapter_a", default="/home/deployer/otlora/results/sft_s42/adapter_ep1")
    ap.add_argument("--adapter_b", default="/home/deployer/otlora/results/otsal_s42/adapter_ep1")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--out", default="/home/deployer/otlora/results/kl_check.json")
    args = ap.parse_args()

    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel

    base = "/home/deployer/otlora/data/processed/mimic_mlf"
    items = [json.loads(l) for l in open(f"{base}/test.jsonl")][: args.n]
    gtfield = "report" if "report" in items[0] else "gt"
    rows = [{"id": it["id"], "gt": it[gtfield], "image_path": it["image_path"]}
            for it in items if it.get(gtfield)][: args.n]

    processor = AutoProcessor.from_pretrained(args.model_path, max_pixels=512 * 512)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model = PeftModel.from_pretrained(model, args.adapter_a, adapter_name="a")
    model.load_adapter(args.adapter_b, adapter_name="b")
    model.eval()

    kls, jsds, top1 = [], [], []
    for s in range(0, len(rows), args.chunk):
        chunk = rows[s:s + args.chunk]
        texts = []
        for r in chunk:
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": r["image_path"]},
                {"type": "text", "text": PROMPT}]}]
            prompt = processor.apply_chat_template(msgs, tokenize=False,
                                                   add_generation_prompt=True)
            texts.append(prompt + r["gt"] + "<|im_end|>")
        enc = processor(text=texts,
                        images=[Image.open(r["image_path"]).convert("RGB") for r in chunk],
                        padding=True, truncation=True, max_length=768,
                        return_tensors="pt").to("cuda:0")

        logits = {}
        for name in ["a", "b"]:
            model.set_adapter(name)
            out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                        pixel_values=enc.pixel_values, image_grid_thw=enc.image_grid_thw)
            logits[name] = out.logits.float()

        lens = enc.attention_mask.sum(dim=1)
        for i in range(len(chunk)):
            n = int(lens[i])
            start = max(1, n - 220)
            la = logits["a"][i, start - 1:n - 1]
            lb = logits["b"][i, start - 1:n - 1]
            pa, pb = torch.log_softmax(la, -1), torch.log_softmax(lb, -1)
            kls.append((pa.exp() * (pa - pb)).sum(-1).mean().item())
            m = 0.5 * (pa.exp() + pb.exp())
            jsds.append((0.5 * (pa.exp() * (pa - m.log())).sum(-1)
                         + 0.5 * (pb.exp() * (pb - m.log())).sum(-1)).mean().item())
            top1.append((la.argmax(-1) == lb.argmax(-1)).float().mean().item())
        if (s // args.chunk) % 25 == 0:
            print(f"{s + len(chunk)}/{len(rows)} KL={sum(kls)/len(kls):.5f}", flush=True)

    out = {"n": len(kls), "mean_KL_a_to_b": sum(kls) / len(kls),
           "max_KL": max(kls), "mean_JSD": sum(jsds) / len(jsds),
           "top1_agreement": sum(top1) / len(top1)}
    json.dump(out, open(args.out, "w"), indent=1)
    print("KL_CHECK", out)


if __name__ == "__main__":
    main()
