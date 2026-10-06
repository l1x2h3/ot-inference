#!/usr/bin/env python3
"""Probe: dump per-candidate (n_j, OT score c_j) for ~40 subset items to
diagnose why argmin(c) == argmin(c/n) in 800/800 cases (review5 item 5)."""
import json
import sys

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

sys.path.insert(0, "/home/deployer/otlora/src")
from ot_loss import CAOTLoss

BASE = "/home/deployer/otlora"
CKPT = f"{BASE}/results/sft_s42/adapter_ep1"
IMGTOK = 151655


def main():
    rows = [json.loads(l) for l in open(
        f"{BASE}/data/processed/mimic_mlf/sft42_k8_cands.jsonl")][:40]
    meta = {r["id"]: r for r in [json.loads(l) for l in open(
        f"{BASE}/data/processed/mimic_mlf/test.jsonl")]}
    processor = AutoProcessor.from_pretrained("/home/deployer/otlora/models/Qwen2.5-VL-3B-Instruct", max_pixels=512 * 512)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "/home/deployer/otlora/models/Qwen2.5-VL-3B-Instruct", torch_dtype=torch.bfloat16,
        device_map="cuda:0")
    model = PeftModel.from_pretrained(model, CKPT)
    model.eval()
    PROMPT = ("You are a radiologist. Read the chest X-ray and write the findings "
          "section of the radiology report.")

    ot = CAOTLoss(d_vision=2048, d_text=2048, token_marginal="salience")
    ot = ot.to("cuda:0")

    out = []
    for r in rows[:40]:
        cands = [r["greedy"]] + r["cands"]
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": meta[r["id"]]["image_path"]},
            {"type": "text", "text": PROMPT}]}]
        prompt = processor.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True)
        seqs = [prompt + c + "<|im_end|>" for c in cands]
        img = Image.open(meta[r["id"]]["image_path"]).convert("RGB")
        enc = processor(text=seqs, images=[img] * len(seqs), padding=True,
                        truncation=True, max_length=768, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            o = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                      pixel_values=enc.pixel_values, image_grid_thw=enc.image_grid_thw,
                      output_hidden_states=True)
        n = int(enc.attention_mask.sum(dim=1)[0])
        hidden = o.hidden_states[-1].float()
        vout = model.base_model.model.visual(
            enc.pixel_values, grid_thw=enc.image_grid_thw).float()
        per = (enc.image_grid_thw.prod(dim=1) // 4).tolist()
        vfeats = torch.split(vout, per, dim=0)
        img_tok = enc.input_ids[0] == IMGTOK
        item = {"id": r["id"], "cands": []}
        for j, c in enumerate(cands):
            nj = int(enc.attention_mask.sum(dim=1)[j])
            tail = hidden[j, max(0, nj - 200):nj]
            v_in = vfeats[j][:256]
            if tail.size(0) < 4 or v_in.size(0) < 4:
                continue
            from salience import token_salience
            ids = enc.input_ids[j, max(0, nj - 200):nj].tolist()
            sal = torch.tensor(token_salience(processor.tokenizer, ids, c),
                               device="cuda:0", dtype=torch.float32)
            with torch.no_grad():
                lv, _ = ot(v_in.unsqueeze(0), tail.unsqueeze(0),
                           token_salience=sal.unsqueeze(0))
            item["cands"].append({"j": j, "n_tok": int(tail.size(0),
                                ), "words": len(c.split()), "score": float(lv)})
        sc = [c for c in item["cands"]]
        if len(sc) >= 4:
            am = min(sc, key=lambda c: c["score"])
            amn = min(sc, key=lambda c: c["score"] / max(1, c["n_tok"]))
            amw = min(sc, key=lambda c: c["score"] / max(1, c["words"]))
            item["argmin_score"] = am["j"]
            item["argmin_per_tok"] = amn["j"]
            item["argmin_per_word"] = amw["j"]
            spread = max(c["score"] for c in sc) - min(c["score"] for c in sc)
            mean = sum(c["score"] for c in sc) / len(sc)
            item["rel_spread"] = spread / mean
        out.append(item)
    json.dump(out, open(f"{BASE}/results/probe_otnorm.json", "w"), indent=1)
    same = sum(1 for it in out if it.get("argmin_score") == it.get("argmin_per_tok"))
    samew = sum(1 for it in out if it.get("argmin_score") == it.get("argmin_per_word"))
    sp = [it["rel_spread"] for it in out if "rel_spread" in it]
    print(f"items={len(out)} argmin(score)==argmin(score/tok): {same}, ==argmin(score/word): {samew}")
    print(f"relative within-set score spread: mean={sum(sp)/len(sp):.3f} min={min(sp):.3f} max={max(sp):.3f}")


if __name__ == "__main__":
    main()
