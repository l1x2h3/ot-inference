#!/usr/bin/env python3
"""CheXpert-style rule labeler upgraded with the OFFICIAL mention phrase lists
(stanfordmlgroup/chexpert-labeler phrases/), with simplified context rules:
mention + negation window -> negative; mention + uncertainty -> unmentioned(0);
else positive. Unmention phrases are excluded first.

This is the same class of "CheXpert-style" labeler as the LOTUS codebase, but
with official mention coverage. Uncertainty is mapped to 0 (blank) following
common RRG practice.
"""
import os
import re

_HERE = os.path.dirname(__file__)
PHRASES = os.path.join(_HERE, "chexpert_phrases")

CATEGORIES = ["Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Lesion",
              "Lung Opacity", "Edema", "Consolidation", "Pneumonia",
              "Atelectasis", "Pneumothorax", "Pleural Effusion",
              "Pleural Other", "Fracture", "Support Devices", "No Finding"]

_file2cat = {
    "enlarged_cardiomediastinum.txt": "Enlarged Cardiomediastinum",
    "cardiomegaly.txt": "Cardiomegaly", "lung_lesion.txt": "Lung Lesion",
    "lung_opacity.txt": "Lung Opacity", "edema.txt": "Edema",
    "consolidation.txt": "Consolidation", "pneumonia.txt": "Pneumonia",
    "atelectasis.txt": "Atelectasis", "pneumothorax.txt": "Pneumothorax",
    "pleural_effusion.txt": "Pleural Effusion", "pleural_other.txt": "Pleural Other",
    "fracture.txt": "Fracture", "support_devices.txt": "Support Devices",
    "no_finding.txt": "No Finding",
}


def _load(kind):
    out = {}
    d = os.path.join(PHRASES, kind)
    for fn in os.listdir(d):
        if not fn.endswith(".txt"):
            continue
        cat = _file2cat.get(fn)
        if cat is None:
            continue
        with open(os.path.join(d, fn)) as f:
            pats = [l.strip() for l in f if l.strip()]
        out[cat] = pats
    return out


MENTION = _load("mention")
UNMENTION = _load("unmention")

NEG_WORDS = re.compile(
    r"\b(no|not|without|free of|absent|resolution of|resolved|negative for|"
    r"declines|declined|denied|rules? out|rule out|excluded?|nor)\b", re.I)
UNCERT_WORDS = re.compile(
    r"\b(may|might|possible|possibly|questionable|cannot exclude|can't exclude|"
    r"cannot be excluded|concerning for|suspicious for|suggests?|likely|"
    r"probable|borderline|equivocal|uncertain)\b", re.I)
# Normal-descriptor context ("heart size is normal" -> negative), replacing the
# ML polarity stage of the official pipeline with a same-sentence rule.
NORMAL_WORDS = re.compile(
    r"\b(normal|normally|unremarkable|within normal limits|clear(ly)?|"
    r"intact|stable|unchanged|is not enlarged|no enlargement)\b", re.I)


def _escape(p):
    return re.escape(p.strip())


def _compile(cat_pats):
    # longest first so specific phrases win
    pats = sorted(cat_pats, key=len, reverse=True)
    return re.compile("|".join(_escape(p) for p in pats), re.I)


MENTION_RX = {c: _compile(ps) for c, ps in MENTION.items() if ps}
UNMENTION_RX = {c: _compile(ps) for c, ps in UNMENTION.items() if ps}


def _mask_unmentions(text, cat):
    """Blank out unmention phrase spans so they can't trigger mentions."""
    rx = UNMENTION_RX.get(cat)
    if rx is None:
        return text
    return rx.sub(" ", text)


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def label_report(text):
    """Return dict cat -> {1,-1,0}. Context rules are applied within the
    sentence containing the mention (no cross-sentence negation)."""
    out = {}
    sents = _SENT_SPLIT.split((text or "").replace("\n", " "))
    for cat in CATEGORIES:
        out[cat] = 0
        m = MENTION_RX.get(cat)
        if m is None:
            continue
        for s in sents:
            s = _mask_unmentions(s, cat)
            hit = m.search(s)
            if hit is None:
                continue
            window = s[:hit.start()]
            if NEG_WORDS.search(window):
                out[cat] = -1
            elif NORMAL_WORDS.search(s):
                out[cat] = -1
            elif UNCERT_WORDS.search(window):
                out[cat] = 0
            else:
                out[cat] = 1
            break  # first mention sentence wins
    return out


def clinical_f1_from_pairs(rows, out_file):
    """rows: list of dicts with 'gt','pred'. Writes per-label F1 JSON."""
    per_label = {c: {"tp": 0, "fp": 0, "fn": 0} for c in CATEGORIES}
    for r in rows:
        pg, gg = label_report(r["pred"]), label_report(r["gt"])
        for c in CATEGORIES:
            p, g = pg[c], gg[c]
            if p == 1 and g == 1:
                per_label[c]["tp"] += 1
            elif p == 1 and g != 1:
                per_label[c]["fp"] += 1
            elif p != 1 and g == 1:
                per_label[c]["fn"] += 1
    f1s = {}
    for c, d in per_label.items():
        prec = d["tp"] / max(1, d["tp"] + d["fp"])
        rec = d["tp"] / max(1, d["tp"] + d["fn"])
        f1s[c] = 2 * prec * rec / max(1e-9, prec + rec)
    present = [c for c in CATEGORIES if sum(per_label[c].values()) > 0]
    macro = sum(f1s[c] for c in present) / max(1, len(present))
    tp = sum(per_label[c]["tp"] for c in CATEGORIES)
    denom = 2 * tp + sum(per_label[c]["fp"] for c in CATEGORIES) \
        + sum(per_label[c]["fn"] for c in CATEGORIES)
    micro = 2 * tp / max(1, denom)
    out = {"F1_macro": macro, "F1_micro": micro, "per_label_F1": f1s,
           "counts": {k: dict(v) for k, v in per_label.items()},
           "labeler": "chexpert-official-mentions + simplified context rules"}
    import json
    json.dump(out, open(out_file, "w"), indent=2)
    return macro, micro


if __name__ == "__main__":
    demo = ("There is a moderate left pleural effusion. No pneumothorax. "
            "Endotracheal tube tip satisfactory. Cardiomegaly is present.")
    print(label_report(demo))
    assert label_report(demo)["Pleural Effusion"] == 1
    assert label_report(demo)["Pneumothorax"] == -1
    assert label_report(demo)["Support Devices"] == 1
    assert label_report(demo)["Cardiomegaly"] == 1
    print("chexpert_label self-test OK")
