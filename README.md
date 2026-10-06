# Transport, Don't Train

Code for *"Transport, Don't Train: Optimal Transport as an Inference-Time
Selection Signal for Radiology Report Generation"*.

## One-line summary

Under MLLM PEFT, decoder-side OT **training losses** collapse (learned
projections) or are absorbed by the LM objective (fixed projections) — but the
**same transport geometry, used as an inference-time candidate-selection
score**, improves over greedy decoding by up to +53% BLEU-1, +48% clinical
macro-F1, and 4.7x output diversity on the full MIMIC-MLF test set,
training-free. Relative to the random-selection and log-probability selection
baselines, the OT signal adds +0.038 BLEU-1 / +0.019 macro-F1 over random and
the hybrid +0.014 BLEU-1 over log-probability (paired bootstrap p <= 0.001).

**Honest accounting (revision 2).** Most of the aggregate lexical/clinical
gain of sampling-plus-selection is a *length* effect: greedy under-generates
(40 vs 69 words), a pure target-length picker reaches B-1 0.267 on the full
test set, and precision-vs-length interpolation explains every learned
selector to +/-0.001. What survives same-length contrasts is a small but
robust OT residual (+0.011 BLEU-1, 95% CI [+0.005, +0.017]); the SFT-vs-SFT+OT
next-token KL (1.55) is of the same order as seed-to-seed KL (1.23). The
deployed scorer used uniform token marginals and the first 256 visual tokens;
`rerank_rev5.py` re-scores with true salience marginals, all visual tokens,
and a shared projection.

## Layout

```
src/
  ot_loss.py          anchored (fixed JL projection) CA-OT loss, Sinkhorn
  salience.py         CheXpert-category token salience for OT marginals
  chexpert_label.py   clinical labeler (official mention lists + context rules)
  train_lora_ot.py    Qwen2.5-VL + LoRA SFT / + OT supervision (train-time arm)
  sample_k.py         sample k candidates + greedy (inference arm, phase 1)
  rerank_ot.py        OT / log-prob / hybrid reranking (phase 2)
  rerank_variants.py  OT formulation comparison (balanced/unbalanced/cosine/...)
  eval_metrics.py     NLG (BLEU/ROUGE/CIDEr/distinct) + clinical F1
  paper_tables.py     aggregate result JSONs into summary tables
  make_figures.py     analysis figures from training logs and eval outputs
  viz_transport.py    per-entity transport-mass overlays
scripts/              dataset download / preparation / experiment launchers
results/              aggregated metrics (TABLES.md) and key result JSONs
```

## Reproduce

1. `bash scripts/download_data.sh` (MIMIC-MLF / IU X-Ray / MIMIC-CXR-RRG test
   via hf-mirror) and `bash scripts/download_models.sh` (Qwen2.5-VL-3B/7B,
   CheXbert weights source).
2. `python scripts/prepare_data.py --which all`
3. Train baseline: `python src/train_lora_ot.py --dataset mimic_mlf`
   Train + OT: add `--use_ot --token_marginal salience`
4. Rerank: `python src/sample_k.py --adapter <ckpt> --k 8 --tag run1` then
   `python src/rerank_ot.py --adapter <ckpt> --cands_tag run1`
5. Evaluate: `python src/eval_metrics.py --adapter <ckpt> --tag run1`

See `EXPERIMENT_PLAN.md` for the full experimental matrix, and
`results/TABLES.md` for aggregated numbers. Run everything from the
repository root (paths in defaults are relative).

## Requirements

Python 3.10; see `requirements.txt`. Single A100-80GB suffices for all 3B
experiments (LoRA trains in ~4.5h; reranking adds ~1.5x greedy inference cost).

## Datasets note

MIMIC-MLF is a community redistribution of MIMIC-CXR (patient-level splits);
MIMIC-CXR-RRG test is the official RRG split; IU X-Ray via public mirror.
No data is transmitted to external services (PhysioNet DUA compliance).
