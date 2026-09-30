#!/usr/bin/env python3
"""Token salience for OT marginals: mark tokens inside clinical entity phrases.

Entity lexicon = the 14 CheXpert finding categories (keyword phrases), reused
from the LOTUS codebase's rule labeler so the salience definition stays
consistent with the published work's clinical categories.
"""
import re

# Salient phrases per CheXpert category (lowercase; regex-escaped where needed)
CATEGORY_PHRASES = {
    "Enlarged Cardiomediastinum": ["enlarged cardiomediastinum", "widened mediastinum",
                                   "cardiomediastinal enlargement", "enlarged cardiac silhouette",
                                   "mediastinal widening"],
    "Cardiomegaly": ["cardiomegaly", "cardiac enlargement", "enlarged heart",
                     "heart size is enlarged", "cardiac silhouette is enlarged"],
    "Lung Lesion": ["lung lesion", "mass in the lung", "pulmonary mass", "nodule",
                    "nodules", "lung mass", "cavitary lesion"],
    "Lung Opacity": ["opacity", "opacities", "haziness", "hazy", "ground glass",
                     "ground-glass", "infiltrate", "infiltrates", "airspace disease"],
    "Edema": ["edema", "pulmonary edema", "congestion", "congestive heart failure",
              "fluid overload", "vascular congestion"],
    "Consolidation": ["consolidation", "consolidations"],
    "Pneumonia": ["pneumonia", "pneumonitis"],
    "Atelectasis": ["atelectasis", "atelectases", "collapse of the lung",
                    "subsegmental atelectasis"],
    "Pneumothorax": ["pneumothorax", "pneumothoraces"],
    "Pleural Effusion": ["pleural effusion", "effusion", "effusions",
                         "fluid in the pleural space", "subpulmonic effusion"],
    "Pleural Other": ["pleural thickening", "pleural calcification", "pneumoperitoneum",
                      "pleural abnormality"],
    "Fracture": ["fracture", "fractures", "rib fracture", "compression fracture",
                 "displaced rib"],
    "Support Devices": ["endotracheal tube", "central venous catheter", "central line",
                        "picc line", "picc", "nasogastric tube", "ng tube",
                        "chest tube", "pacemaker", "defibrillator", "swan ganz",
                        "swan-ganz", "ij catheter", "jugular catheter", "dialysis catheter",
                        "support device", "device", "line is", "tube is", "catheter"],
    "No Finding": ["no acute cardiopulmonary abnormality", "no acute abnormality",
                   "no active disease", "normal limits", "clear lungs",
                   "lungs are clear", "no pneumothorax", "no pleural effusion",
                   "unremarkable"],
}

_ALL = sorted({p for v in CATEGORY_PHRASES.values() for p in v},
              key=len, reverse=True)  # longest-first so spans prefer long matches
_PATTERN = re.compile("|".join(re.escape(p) for p in _ALL), re.IGNORECASE)


def entity_spans(text: str):
    """Char-level spans of salient phrases in *text*."""
    return [m.span() for m in _PATTERN.finditer(text or "")]


def token_salience(tokenizer, report_ids, report, soft_weight=2.0, base=1.0):
    """Per-token salience over the assistant (report) token ids.

    A token is salient if any char of its (decoded) surface falls inside an
    entity span of *report*. Returns list[float] aligned with report_ids.
    """
    spans = entity_spans(report)
    if not spans:
        return [base] * len(report_ids)
    offsets = []  # approximate char offsets via cumulative decode lengths
    # decode token-by-token to map token -> chars precisely
    cursor, tok_range = 0, []
    for tid in report_ids:
        s = tokenizer.decode([tid], skip_special_tokens=True)
        tok_range.append((cursor, cursor + max(1, len(s))))
        cursor += len(s)
    out = []
    for (a, b) in tok_range:
        hit = any(not (b <= s or a >= e) for s, e in spans)
        out.append(soft_weight if hit else base)
    return out
