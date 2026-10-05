#!/usr/bin/env python3
"""grad_diag v2: adds the reference baseline review4 item 12 asks for --
the cosine between LM gradients of DIFFERENT batches. In tens-of-millions-
dimensional parameter space random vectors are near-orthogonal anyway, so
cos(g_LM, g_OT)=0.03 is only meaningful against cos(g_LM^t, g_LM^{t-1}).
Writes results/grad_diag2.json (keeps all v1 fields).
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
    ap.add_argument("--model_path", default="/home/deployer/otlora/models/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--adapter", default="/home/deployer/otlora/results/sft_s42/adapter_ep1")
    ap.add_argument("--dataset", default="mimic_mlf")
    ap.add_argument("--data_root", default="/home/deployer/otlora/data/processed")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--out", default="/home/deployer/otlora/results/grad_diag2.json")
    args = ap.parse_args()
    torch.manual_seed(42)

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
    model.eval()

    d_model = model.config.hidden_size
    ot_crit = CAOTLoss(d_vision=d_model, d_text=d_model, lam_local=0.7,
                       epsilon=0.1, token_marginal="salience",
                       proj_mode="fixed").cuda()

    train_ds = JsonlImgDataset(os.path.join(args.data_root, args.dataset, "train.jsonl"))
    collate = build_collate(processor, processor.tokenizer, use_salience=True, max_pixels=None)
    dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=2,
                    collate_fn=collate, drop_last=True)

    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    def gather():
        return torch.cat([p.grad.detach().float().flatten()
                          for (_, p) in named if p.grad is not None])

    records = []
    prev_g_lm = None
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
            l, _ = ot_crit(v.unsqueeze(0), t.unsqueeze(0), token_salience=sal.unsqueeze(0))
            losses.append(l)
        if not losses:
            continue
        ot_loss = torch.stack(losses).mean()
        ot_loss.backward()
        g_ot = gather()
        model.zero_grad(set_to_none=True)

        cos = torch.nn.functional.cosine_similarity(g_lm, g_ot, dim=0).item()
        rec = {"step": done, "lm": lm_loss.item(), "ot": ot_loss.item(),
               "cos": cos, "ratio": g_ot.norm().item() / max(g_lm.norm().item(), 1e-12)}
        if prev_g_lm is not None:
            rec["cos_lm_lm"] = torch.nn.functional.cosine_similarity(
                g_lm, prev_g_lm, dim=0).item()
        prev_g_lm = g_lm
        records.append(rec)
        done += 1
        print(f"{done}/{args.steps} lm={lm_loss.item():.3f} ot={ot_loss.item():.3f} "
              f"cos={cos:+.3f} cos_lmlm={rec.get('cos_lm_lm', float('nan')):+.3f}",
              flush=True)
        if done >= args.steps:
            break

    json.dump(records, open(args.out, "w"), indent=1)
    import statistics
    lmlm = [r["cos_lm_lm"] for r in records if "cos_lm_lm" in r]
    print(f"DIAG2_DONE mean_cos_lm_lm={statistics.mean(lmlm):+.4f} "
          f"mean_cos_lmot={statistics.mean(r['cos'] for r in records):+.4f} "
          f"mean_ratio={statistics.mean(r['ratio'] for r in records):.4f}")


if __name__ == "__main__":
    main()
