#!/usr/bin/env python3
"""OT-LoRA training: Qwen2.5-VL + LoRA + decoder-side salience-weighted CA-OT.

Usage (single A100):
  python train_lora_ot.py --dataset mimic_mlf --use_ot --token_marginal salience
  python train_lora_ot.py --dataset mimic_mlf            # plain SFT baseline
"""
import argparse
import json
import math
import os
import sys

import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from ot_loss import CAOTLoss, ratio_capped_weight  # noqa: E402
from salience import token_salience  # noqa: E402

IM_START = "<|im_start|>"


class JsonlImgDataset(Dataset):
    def __init__(self, path):
        self.items = [json.loads(l) for l in open(path)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def build_collate(processor, tokenizer, use_salience, max_pixels):
    from PIL import Image

    def collate(batch):
        pil_images, prompt_parts, reply_parts = [], [], []
        for ex in batch:
            msgs = [
                {"role": "user", "content": [
                    {"type": "image", "image": ex["image_path"]},
                    {"type": "text", "text": "You are a radiologist. Read the chest X-ray and write the findings section of the radiology report."},
                ]},
            ]
            prompt = processor.apply_chat_template(msgs, tokenize=False,
                                                   add_generation_prompt=True)
            prompt_parts.append(prompt)
            reply_parts.append(ex["report"])
            pil_images.append(Image.open(ex["image_path"]).convert("RGB"))
        # full sequences: prompt + reply + <|im_end|>
        full_enc = processor(text=[p + r + "<|im_end|>" for p, r in zip(prompt_parts, reply_parts)],
                             images=pil_images, padding=True, truncation=True,
                             max_length=768, return_tensors="pt")
        # prompt-only encoding to locate the supervised span (includes image tokens)
        prompt_enc = processor(text=prompt_parts, images=pil_images, padding=True,
                               return_tensors="pt")
        input_ids, labels = full_enc.input_ids, torch.full_like(full_enc.input_ids, -100)
        sal_t = None
        saliences = []
        for bi, (p, r) in enumerate(zip(prompt_parts, reply_parts)):
            plen = int(prompt_enc.input_ids[bi].ne(processor.tokenizer.pad_token_id).sum()) \
                if processor.tokenizer.pad_token_id is not None else int(prompt_enc.input_ids.shape[1])
            rlen = len(tokenizer(r + "<|im_end|>", add_special_tokens=False).input_ids)
            end = min(plen + rlen, input_ids.shape[1])
            labels[bi, plen:end] = input_ids[bi, plen:end]
            if use_salience:
                ids = input_ids[bi, plen:end].tolist()
                saliences.append(torch.tensor(token_salience(tokenizer, ids, r)))
        if use_salience and saliences:
            maxr = max(s.numel() for s in saliences)
            sal_t = torch.ones(len(batch), maxr)
            for bi, s in enumerate(saliences):
                sal_t[bi, :s.numel()] = s
        return {"input_ids": input_ids, "attention_mask": full_enc.attention_mask,
                "pixel_values": full_enc.pixel_values, "image_grid_thw": full_enc.image_grid_thw,
                "labels": labels, "salience": sal_t}
    return collate


@torch.no_grad()
def vision_patch_feats(model, pixel_values, image_grid_thw):
    """Frozen vision tower output for each image: list of [n_i, D] tensors."""
    if hasattr(model, "get_vision_tower"):
        visual = model.get_vision_tower()
    elif hasattr(model, "base_model"):  # peft-wrapped ForConditionalGeneration
        visual = model.base_model.model.visual
    else:
        visual = model.visual
    out = visual(pixel_values, grid_thw=image_grid_thw)  # [total_patches, D]
    per = (image_grid_thw.prod(dim=1) // 4).tolist()  # 2x2 merge
    return list(torch.split(out.float(), per, dim=0))


def report_token_hiddens(hidden, labels):
    """Last-layer states at report-token positions. hidden [B,L,D] -> list of [n_i, D]."""
    outs = []
    for b in range(hidden.size(0)):
        idx = (labels[b] != -100).nonzero(as_tuple=True)[0]
        outs.append(hidden[b, idx].float())
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/processed")
    ap.add_argument("--dataset", default="mimic_mlf", choices=["mimic_mlf", "iu"])
    ap.add_argument("--model_path", default="models/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--out_dir", default="results/run")
    ap.add_argument("--use_ot", action="store_true")
    ap.add_argument("--token_marginal", default="salience",
                    choices=["uniform", "salience", "gated"])
    ap.add_argument("--proj_mode", default="fixed", choices=["fixed", "learned"])
    ap.add_argument("--ot_weight", type=float, default=0.05)
    ap.add_argument("--rho_cap", type=float, default=0.5)
    ap.add_argument("--lam_local", type=float, default=0.7)
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_items", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import LoraConfig, get_peft_model

    processor = AutoProcessor.from_pretrained(
        args.model_path, max_pixels=512 * 512, padding_side="right")
    tokenizer = processor.tokenizer
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="cuda:0")
    model.config.use_cache = False
    lcfg = LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
        target_modules=r"^model\.layers\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
        task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    d_model = model.config.hidden_size if hasattr(model.config, "hidden_size") \
        else model.config.text_config.hidden_size
    ot_crit = CAOTLoss(d_vision=d_model, d_text=d_model,
                       lam_local=args.lam_local, lam_global=1 - args.lam_local,
                       epsilon=args.epsilon, token_marginal=args.token_marginal,
                       proj_mode=args.proj_mode).cuda()

    split_dir = os.path.join(args.data_root, args.dataset)
    train_ds = JsonlImgDataset(os.path.join(split_dir, "train.jsonl"))
    if args.max_items:
        train_ds.items = train_ds.items[:args.max_items]
    collate = build_collate(processor, tokenizer, use_salience=args.use_ot, max_pixels=None)
    dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=4,
                    collate_fn=collate, drop_last=True)
    steps_per_epoch = math.ceil(len(dl) / args.accum)

    params = [p for p in model.parameters() if p.requires_grad]
    groups = [{"params": params, "lr": args.lr}]
    ot_params = [p for p in ot_crit.parameters() if p.requires_grad]
    if ot_params:
        groups.append({"params": ot_params, "lr": 1e-3})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps_per_epoch)

    model.train()
    gstep = 0
    for ep in range(args.epochs):
        for it, batch in enumerate(dl):
            pixel_values = batch["pixel_values"].to("cuda:0", torch.bfloat16)
            grid = batch["image_grid_thw"].to("cuda:0")
            out = model(input_ids=batch["input_ids"].cuda(),
                        attention_mask=batch["attention_mask"].cuda(),
                        pixel_values=pixel_values, image_grid_thw=grid,
                        labels=batch["labels"].cuda(), output_hidden_states=True)
            lm_loss = out.loss
            tot_ot = None
            if args.use_ot:
                with torch.no_grad():
                    vfeats = vision_patch_feats(model, pixel_values, grid)
                hiddens = report_token_hiddens(out.hidden_states[-1], batch["labels"])
                losses = []
                for b, (v, t) in enumerate(zip(vfeats, hiddens)):
                    if v.size(0) < 2 or t.size(0) < 2:
                        continue
                    sal = batch["salience"][b, :t.size(0)].cuda()
                    l, _ = ot_crit(v.unsqueeze(0), t.unsqueeze(0),
                                   token_salience=sal.unsqueeze(0))
                    losses.append(l)
                if losses:
                    tot_ot = torch.stack(losses).mean()
                    w = ratio_capped_weight(tot_ot, lm_loss, args.ot_weight, args.rho_cap)
                    loss = lm_loss + w * tot_ot
                else:
                    loss = lm_loss
            else:
                loss = lm_loss
            (loss / args.accum).backward()
            if (it + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(params + list(ot_crit.parameters()), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                gstep += 1
                if gstep % 5 == 0:
                    msg = f"ep{ep} step{gstep} lm={lm_loss.item():.4f}"
                    if tot_ot is not None:
                        msg += f" ot={tot_ot.item():.4f} w={w.item():.4f}"
                    print(msg, flush=True)
        model.save_pretrained(os.path.join(args.out_dir, f"adapter_ep{ep}"))
        torch.save(ot_crit.state_dict(), os.path.join(args.out_dir, f"ot_heads_ep{ep}.pt"))
    json.dump(vars(args), open(os.path.join(args.out_dir, "args.json"), "w"), indent=2)
    print("TRAINING_DONE", args.out_dir)


if __name__ == "__main__":
    main()
