#!/usr/bin/env python3
"""Prepare datasets for OT-LoRA: extract parquet images to disk + build JSONL.

Outputs per split: images/ (jpg) + ann.json lines {"id", "image_path", "report"}.
Runs with system python3 (pyarrow+pillow only, no torch).
"""
import argparse
import io
import json
import os

import pyarrow.parquet as pq
from PIL import Image


def prepare_mimic_mlf(data_dir, out_dir, min_len=30):
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    splits = {
        "train": ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"],
        "val": ["validation-00000-of-00001.parquet"],
        "test": ["test-00000-of-00001.parquet"],
    }
    for split, files in splits.items():
        table = pq.read_table([os.path.join(data_dir, f) for f in files])
        n, skipped = 0, 0
        with open(os.path.join(out_dir, f"{split}.jsonl"), "w") as out:
            for i, row in enumerate(table.to_pylist()):
                img_bytes, report = row["image"]["bytes"], row["reports"].strip()
                if len(report) < min_len or not img_bytes:
                    skipped += 1
                    continue
                img_id = f"{split}_{i:06d}.jpg"
                path = os.path.join(out_dir, "images", img_id)
                if not os.path.exists(path):
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    img.save(path, "JPEG", quality=92)
                out.write(json.dumps({"id": img_id, "image_path": path,
                                      "report": report}) + "\n")
                n += 1
        print(f"MIMIC-MLF {split}: {n} examples ({skipped} skipped)")


def prepare_iu(data_dir, out_dir):
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    # zip nests images under mnt/.../iu_xray/images/<study>/<0|1>.png; flatten first
    nested = os.path.join(out_dir, "mnt")
    if os.path.exists(nested) and not os.path.exists(os.path.join(out_dir, "images", ".flattened")):
        src_root = None
        for root, dirs, files in os.walk(nested):
            if dirs and dirs[0].startswith("CXR") and any(f.endswith(".png") for f in files):
                src_root = root
                break
        if src_root is None:
            # find the deepest dir that directly contains CXR* study folders
            candidates = [root for root, dirs, _ in os.walk(nested)
                          if any(d.startswith("CXR") for d in dirs)]
            src_root = max(candidates, key=len)
        os.system(f"mv '{src_root}'/* {os.path.join(out_dir, 'images')}/ && rm -rf {nested}"
                  f" && touch {os.path.join(out_dir, 'images', '.flattened')}")
    for split in ["train", "val", "test"]:
        src = os.path.join(data_dir, f"{split}.jsonl")
        n, miss = 0, 0
        with open(src) as fin, open(os.path.join(out_dir, f"{split}.jsonl"), "w") as out:
            for line in fin:
                item = json.loads(line)
                # images: ['/iu_xray/image/<study>/<0|1>.png']; 0 = frontal
                imgs = item.get("images") or []
                study = imgs[0].split("/")[-2] if imgs else None
                frontal = os.path.join(out_dir, "images", study, "0.png") if study else None
                report = item.get("response", "").strip()
                if not frontal or not os.path.exists(frontal) or len(report) < 20:
                    miss += 1
                    continue
                out.write(json.dumps({"id": f"{study}_0", "image_path": frontal,
                                      "report": report}) + "\n")
                n += 1
        print(f"IU {split}: {n} examples ({miss} missing/short)")


def prepare_rrg_test(data_dir, out_dir, max_rows=2461):
    """MIMIC-CXR-RRG official test (findings_section config) -> external eval set."""
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    n = 0
    with open(os.path.join(out_dir, "test.jsonl"), "w") as out:
        for f in sorted(os.listdir(data_dir)):
            if not f.endswith(".parquet"):
                continue
            table = pq.read_table(os.path.join(data_dir, f), columns=["main_image", "findings_section"])
            for i, row in enumerate(table.to_pylist()):
                img_bytes, findings = row["main_image"]["bytes"], row["findings_section"]
                if not img_bytes or not findings or len(findings.strip()) < 30:
                    continue
                img_id = f"rrg_{f.split('-')[1]}_{i:05d}.jpg"
                path = os.path.join(out_dir, "images", img_id)
                if not os.path.exists(path):
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    img.save(path, "JPEG", quality=92)
                out.write(json.dumps({"id": img_id, "image_path": path,
                                      "report": findings.strip()}) + "\n")
                n += 1
                if n >= max_rows:
                    break
            if n >= max_rows:
                break
    print(f"RRG-2461 test: {n} examples")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", required=True,
                    choices=["mimic_mlf", "iu", "rrg_test", "all"])
    ap.add_argument("--data-root", default="data")
    args = ap.parse_args()
    if args.which in ("mimic_mlf", "all"):
        prepare_mimic_mlf(f"{args.data_root}/mimic_mlf", f"{args.data_root}/processed/mimic_mlf")
    if args.which in ("iu", "all"):
        prepare_iu(f"{args.data_root}/iu_xray", f"{args.data_root}/processed/iu")
    if args.which in ("rrg_test", "all"):
        prepare_rrg_test(f"{args.data_root}/mimic_rrg_test", f"{args.data_root}/processed/rrg_test")
