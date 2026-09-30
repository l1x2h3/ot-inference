#!/usr/bin/env python3
"""Generate reports with a trained/zero-shot model and evaluate.

NLG: BLEU-1..4, ROUGE-L, CIDEr (pycocoevalcap), distinct-2.
Clinical: CheXbert 14-label CE metrics when weights available, otherwise the
rule labeler (CheXpert-style patterns, same as LOTUS codebase).

Usage:
  python generate_eval.py --adapter results/run/adapter_ep1 --split test --dataset mimic_mlf
  python generate_eval.py --zero_shot --split test --dataset rrg_test
"""
import argparse
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))

PROMPT = ("You are a radiologist. Read the chest X-ray and write the findings "
          "section of the radiology report.")


# ---------------- generation ----------------

@torch.no_grad()
def generate(model_path, adapter, test_items, out_json, max_new=220, bs=8,
             min_new=0, num_beams=1):
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(model_path, max_pixels=512 * 512)
    processor.tokenizer.padding_side = "left"  # decoder-only batch generation
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda:0")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    preds = []
    for s in range(0, len(test_items), bs):
        chunk = test_items[s:s + bs]
        imgs = [Image.open(it["image_path"]).convert("RGB") for it in chunk]
        msgs = [[{"role": "user", "content": [
            {"type": "image", "image": it["image_path"]},
            {"type": "text", "text": PROMPT}]}] for it in chunk]
        texts = [processor.apply_chat_template(m, tokenize=False,
                                               add_generation_prompt=True) for m in msgs]
        enc = processor(text=texts, images=imgs,
                        padding=True, return_tensors="pt").to("cuda:0")
        gen_kwargs = dict(max_new_tokens=max_new, do_sample=False,
                          repetition_penalty=1.05)
        if min_new:
            gen_kwargs["min_new_tokens"] = min_new
        if num_beams > 1:
            gen_kwargs["num_beams"] = num_beams
        gen = model.generate(**enc, **gen_kwargs)
        gen = gen[:, enc.input_ids.shape[1]:]
        outs = processor.batch_decode(gen, skip_special_tokens=True)
        preds.extend(outs)
        if (s // bs) % 10 == 0:
            print(f"gen {s + len(chunk)}/{len(test_items)}", flush=True)
    with open(out_json, "w") as f:
        for it, p in zip(test_items, preds):
            f.write(json.dumps({"id": it["id"], "gt": it["report"], "pred": p}) + "\n")
    print("WROTE", out_json)


# ---------------- NLG metrics ----------------

def compute_nlg(pred_file, out_file):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge

    rows = [json.loads(l) for l in open(pred_file)]
    gts = {i: [r["gt"].lower()] for i, r in enumerate(rows)}
    res = {i: [r["pred"].lower()] for i, r in enumerate(rows)}
    bleu, _ = Bleu(4).compute_score(gts, res)
    rouge, _ = Rouge().compute_score(gts, res)
    cider, _ = Cider().compute_score(gts, res)

    def distinct_n(texts, n=2):
        total, uniq = 0, set()
        for t in texts:
            toks = t.lower().split()
            grams = [" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)]
            total += len(grams); uniq.update(grams)
        return len(uniq) / max(1, total)

    metrics = {f"BLEU-{i + 1}": bleu[i] for i in range(4)}
    metrics["ROUGE-L"] = rouge
    metrics["CIDEr"] = cider
    metrics["distinct-2"] = distinct_n([r["pred"] for r in rows])
    metrics["distinct-3"] = distinct_n([r["pred"] for r in rows], 3)
    uniq_reports = len({r["pred"] for r in rows})
    metrics["unique_reports"] = f"{uniq_reports}/{len(rows)}"
    json.dump(metrics, open(out_file, "w"), indent=2)
    print(json.dumps(metrics, indent=2))


# ---------------- clinical metrics (rule labeler; CheXbert upgrade path) ----------------

LABELS = ["Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Lesion", "Lung Opacity",
          "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
          "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding"]

NEG_HINT = re.compile(r"\b(no|not|without|free of|absent|resolution of|resolved|negative for)\b",
                      re.IGNORECASE)


def rule_label(text):
    """14-dim label vector in {-1, 0, 1}: 1 positive, -1 negative, 0 unmentioned.
    Simplified CheXpert-style matcher consistent with the LOTUS codebase."""
    t = text.lower()
    out = {}
    for lab in LABELS:
        key = lab.lower()
        pats = [key.replace(" ", r"\s+")]
        if lab == "Lung Opacity":
            pats += [r"opacit", r"haz", r"infiltrat", r"ground.?glass"]
        if lab == "Pleural Effusion":
            pats += [r"effusion", r"subpulmonic fluid"]
        if lab == "Support Devices":
            pats += [r"\bdevice", r"endotracheal", r"\bng tube\b", r"nasogastric",
                     r"chest tube", r"central (venous )?catheter", r"central line",
                     r"picc", r"pacemaker", r"defibrillator"]
        if lab == "No Finding":
            pats = [r"no (acute|active|cardiopulmonary|significant)", r"normal limits",
                    r"clear(ly)? (lungs|of)", r"unremarkable"]
        hit = None
        for p in pats:
            m = re.search(p, t)
            if m:
                window = t[max(0, m.start() - 60):m.start()]
                hit = -1 if NEG_HINT.search(window) else 1
                break
        out[lab] = 0 if hit is None else hit
    return out


def clinical_f1(pred_file, out_file):
    """Upgraded: official CheXpert mention lists (chexpert_label module)."""
    from chexpert_label import clinical_f1_from_pairs
    rows = [json.loads(l) for l in open(pred_file)]
    macro, micro = clinical_f1_from_pairs(rows, out_file)
    print(f"chexpert-official CE: macro={macro:.3f} micro={micro:.3f} -> {out_file}")
    return


def clinical_f1_legacy(pred_file, out_file):
    rows = [json.loads(l) for l in open(pred_file)]
    per_label = {lab: {"tp": 0, "fp": 0, "fn": 0} for lab in LABELS}
    for r in rows:
        pg, gg = rule_label(r["pred"]), rule_label(r["gt"])
        for lab in LABELS:
            p, g = pg[lab], gg[lab]
            if p == 1 and g == 1:
                per_label[lab]["tp"] += 1
            elif p == 1 and g != 1:
                per_label[lab]["fp"] += 1
            elif p != 1 and g == 1:
                per_label[lab]["fn"] += 1
    f1s = {}
    for lab, c in per_label.items():
        prec = c["tp"] / max(1, c["tp"] + c["fp"])
        rec = c["tp"] / max(1, c["tp"] + c["fn"])
        f1s[lab] = 2 * prec * rec / max(1e-9, prec + rec)
    present = [f1s[l] for l in LABELS if sum(per_label[l].values()) > 0]
    macro = sum(present) / max(1, len(present))
    tp = sum(per_label[l]["tp"] for l in LABELS)
    micro = 2 * tp / max(1, 2 * tp + sum(per_label[l]["fp"] for l in LABELS)
                         + sum(per_label[l]["fn"] for l in LABELS))
    out = {"F1_macro": macro, "F1_micro": micro, "per_label_F1": f1s,
           "counts": {k: dict(v) for k, v in per_label.items()}}
    json.dump(out, open(out_file, "w"), indent=2)
    print(f"rule-labeler CE: macro={macro:.3f} micro={micro:.3f} -> {out_file}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="models/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--zero_shot", action="store_true")
    ap.add_argument("--dataset", default="mimic_mlf", choices=["mimic_mlf", "iu", "rrg_test"])
    ap.add_argument("--data_root", default="data/processed")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max_items", type=int, default=0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--min_new", type=int, default=0)
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--tag", default="pred")
    args = ap.parse_args()
    split_file = os.path.join(args.data_root, args.dataset, f"{args.split}.jsonl")
    items = [json.loads(l) for l in open(split_file)]
    if args.max_items:
        items = items[:args.max_items]
    out_dir = os.path.dirname(split_file)
    pred_file = os.path.join(out_dir, f"{args.tag}_pred.jsonl")
    generate(args.model_path, args.adapter or None, items, pred_file, bs=args.bs,
             min_new=args.min_new, num_beams=args.num_beams)
    compute_nlg(pred_file, pred_file.replace(".jsonl", "_nlg.json"))
    clinical_f1(pred_file, pred_file.replace(".jsonl", "_ce.json"))


if __name__ == "__main__":
    main()
