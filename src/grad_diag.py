#!/usr/bin/env python3
"""Gradient-space diagnostic for the 'supervision is absorbed' claim.

For each probe batch, backprop L_LM and L_OT separately through the LoRA
parameters and record
  - cos(g_LM, g_OT) over the full LoRA gradient vector,
  - ||g_OT|| / ||g_LM|| (and lambda-scaled versions),
  - per-module (q/k/v/o/gate/up/down_proj) cosine and norm shares.

If g_OT is near-orthogonal to g_LM and lambda*||g_OT|| << ||g_LM||, the
alignment signal cannot displace parameters in any direction the LM
objective cares about: absorption is a mechanism, not an interpretation.

Usage:
  python grad_diag.py --adapter results/sft_s42/adapter_ep1 --steps 40
"""
import argparse
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from ot_loss import CAOTLoss  # noqa: E402
from train_lora_ot import (JsonlImgDataset, build_collate,  # noqa: E402
                           report_token_hiddens, vision_patch_feats)

MODULE_RX = re.compile(r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="models/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--adapter", default="results/sft_s42/adapter_ep1")
    ap.add_argument("--dataset", default="mimic_mlf")
    ap.add_argument("--data_root", default="data/processed")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--lam_local", type=float, default=0.7)
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/grad_diag.json")
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    from torch.utils.data import DataLoader
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel

    processor = AutoProcessor.from_pretrained(args.model_path, max_pixels=512 * 512,
                                              padding_side="right")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="cuda:0")
    model.config.use_cache = False
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    model.eval()  # disable LoRA dropout; grads still flow to LoRA params

    d_model = model.config.hidden_size
    ot_crit = CAOTLoss(d_vision=d_model, d_text=d_model, lam_local=args.lam_local,
                       epsilon=args.epsilon, token_marginal="salience",
                       proj_mode="fixed").cuda()

    train_ds = JsonlImgDataset(os.path.join(args.data_root, args.dataset, "train.jsonl"))
    collate = build_collate(processor, processor.tokenizer, use_salience=True, max_pixels=None)
    dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=2,
                    collate_fn=collate, drop_last=True)

    # fixed param order: trainable (LoRA) params only
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    names = [n for n, _ in named]

    def gather():
        return torch.cat([p.grad.detach().float().flatten()
                          for (_, p) in named if p.grad is not None])

    records = []
    done = 0
    for batch in dl:
        pixel_values = batch["pixel_values"].to("cuda:0", torch.bfloat16)
        grid = batch["image_grid_thw"].to("cuda:0")
        model.zero_grad(set_to_none=True)
        out = model(input_ids=batch["input_ids"].cuda(),
                    attention_mask=batch["attention_mask"].cuda(),
                    pixel_values=pixel_values, image_grid_thw=grid,
                    labels=batch["labels"].cuda(), output_hidden_states=True)
        lm_loss = out.loss
        lm_loss.backward(retain_graph=True)
        g_lm = gather()
        model.zero_grad(set_to_none=True)

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
        if not losses:
            continue
        ot_loss = torch.stack(losses).mean()
        ot_loss.backward()
        g_ot = gather()
        model.zero_grad(set_to_none=True)

        cos = torch.nn.functional.cosine_similarity(
            g_lm, g_ot, dim=0).item()
        n_lm, n_ot = g_lm.norm().item(), g_ot.norm().item()
        rec = {"step": done, "lm": lm_loss.item(), "ot": ot_loss.item(),
               "cos": cos, "ratio": n_ot / max(n_lm, 1e-12)}

        # per-module cosine / norm shares
        per = {}
        off = 0
        for (n, p) in named:
            k = MODULE_RX.search(n)
            if not k:
                continue
            mod = k.group(1).replace("_proj", "")
            d = p.numel()
            a, b = g_lm[off:off + d], g_ot[off:off + d]
            ent = per.setdefault(mod, {"dot": 0.0, "lm2": 0.0, "ot2": 0.0})
            ent["dot"] += float(a @ b)
            ent["lm2"] += float(a @ a)
            ent["ot2"] += float(b @ b)
            off += d
        rec["per_module"] = {
            m: {"cos": e["dot"] / max((e["lm2"] * e["ot2"]) ** 0.5, 1e-12),
                "ot_share": e["ot2"] ** 0.5, "lm_share": e["lm2"] ** 0.5}
            for m, e in per.items()}
        records.append(rec)
        done += 1
        print(f"{done}/{args.steps} lm={lm_loss.item():.3f} ot={ot_loss.item():.3f} "
              f"cos={cos:+.3f} ratio={n_ot / max(n_lm, 1e-12):.4f}", flush=True)
        if done >= args.steps:
            break

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(records, open(args.out, "w"), indent=1)
    cos_all = [r["cos"] for r in records]
    ratio_all = [r["ratio"] for r in records]
    print("DIAG_DONE mean|cos|=%.4f mean_cos=%.4f mean_ratio=%.4f "
          "lam1.0_share=%.4f" % (
              sum(abs(c) for c in cos_all) / len(cos_all),
              sum(cos_all) / len(cos_all),
              sum(ratio_all) / len(ratio_all),
              sum(ratio_all) / len(ratio_all)))


if __name__ == "__main__":
    main()
