#!/usr/bin/env python3
"""Shared plumbing for the LLaVA-1.5-7B second-family replication.

Differences from the Qwen path: chat format (USER:/ASSISTANT:), no
image_grid_thw (fixed 576 CLIP patches per image), vision tower accessed
as model.vision_tower / model.base_model.model.vision_tower under PEFT.
"""
import json
import os

PROMPT = ("You are a radiologist. Read the chest X-ray and write the findings "
          "section of the radiology report.")
MODEL_PATH = "models/llava-1.5-7b-hf"
D_ROOT = "data/processed"


def llava_prompt():
    return f"USER: <image>\n{PROMPT} ASSISTANT:"


def load_test_items(dataset="mimic_mlf", n=800):
    items = [json.loads(l) for l in open(os.path.join(D_ROOT, dataset, "test.jsonl"))]
    return items[:n]


def vision_feats(model, pixel_values):
    """LLaVA-1.5 CLIP tower output: [B, 576, 1024] -> list per image [576, 1024]."""
    visual = model.base_model.model.vision_tower if hasattr(model, "base_model") \
        else model.vision_tower
    feats = visual(pixel_values).last_hidden_state  # [B, 576, 1024]
    return [f.float() for f in feats]
