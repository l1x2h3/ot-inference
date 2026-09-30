#!/usr/bin/env python3
"""SFT (LoRA) for LLaVA-1.5-7B on MIMIC-MLF - second-family replication.
Mirrors train_lora_ot.py's SFT path with LLaVA-specific plumbing."""
import argparse
import json
import math
import os
import sys

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(__file__))
from llava_common import D_ROOT, PROMPT, MODEL_PATH  # noqa: E402


class JsonlImg(Dataset):
    def __init__(self, path):
        self.items = [json.loads(l) for l in open(path)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def build_collate(processor, tokenizer):
    from PIL import Image

    def collate(batch):
        from PIL import Image as Im
        imgs, prompts, replies = [], [], []
        for ex in batch:
            imgs.append(Im.open(ex["image_path"]).convert("RGB"))
            prompts.append(f"USER: <image>\n{PROMPT} ASSISTANT:")
            replies.append(ex["report"])
        full = processor(text=[p + r + "</s>" for p, r in zip(prompts, replies)],
                         images=imgs, padding=True, truncation=True,
                         max_length=768, return_tensors="pt")
        ponly = processor(text=prompts, images=imgs, padding=True,
                          return_tensors="pt")
        labels = torch.full_like(full.input_ids, -100)
        pad_id = processor.tokenizer.pad_token_id
        for bi in range(len(batch)):
            plen = int(ponly.input_ids[bi].ne(pad_id).sum())
            flen = int(full.input_ids[bi].ne(pad_id).sum())
            labels[bi, plen:flen] = full.input_ids[bi, plen:flen]
        return {"input_ids": full.input_ids, "attention_mask": full.attention_mask,
                "pixel_values": full.pixel_values, "labels": labels}
    return collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="results/llava_sft")
    ap.add_argument("--max_items", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from peft import LoraConfig, get_peft_model

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model.config.use_cache = False
    lcfg = LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
        target_modules=r"^language_model\.model\.layers\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
        task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    ds = JsonlImg(os.path.join(D_ROOT, "mimic_mlf", "train.jsonl"))
    if args.max_items:
        ds.items = ds.items[:args.max_items]
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=4,
                    collate_fn=build_collate(processor, tokenizer), drop_last=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    steps = math.ceil(len(dl) / args.accum) * args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    model.train()
    gstep = 0
    for ep in range(args.epochs):
        for it, batch in enumerate(dl):
            out = model(input_ids=batch["input_ids"].cuda(),
                        attention_mask=batch["attention_mask"].cuda(),
                        pixel_values=batch["pixel_values"].to("cuda:0", torch.bfloat16),
                        labels=batch["labels"].cuda())
            (out.loss / args.accum).backward()
            if (it + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                gstep += 1
                if gstep % 5 == 0:
                    print(f"ep{ep} step{gstep}/{steps} lm={out.loss.item():.4f}",
                          flush=True)
        model.save_pretrained(os.path.join(args.out_dir, f"adapter_ep{ep + 1}"))
        print(f"SAVED ep{ep + 1}", flush=True)
    print("TRAIN_LLAVA_DONE")


if __name__ == "__main__":
    main()
