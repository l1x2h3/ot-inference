#!/bin/bash
# One-shot bootstrap after torch lands: stack -> selftest -> Qwen -> smoke run
set -e
PY=python3
PIP=pip3
IDX=https://pypi.tuna.tsinghua.edu.cn/simple

echo '=== [1/4] install LLM stack ==='
$PIP install -q -i $IDX "transformers==4.49.0" "peft==0.15.2" accelerate datasets \
    sentencepiece protobuf qwen-vl-utils nltk scikit-learn pandas pycocoevalcap

echo '=== [2/4] OT loss self-test (GPU) ==='
cd src && $PY ot_loss.py

echo '=== [3/4] download Qwen2.5-VL-3B + CheXbert ==='
bash scripts/download_models.sh

echo '=== [4/4] IU smoke: 60 samples, 1 epoch, SFT vs SFT+OT ==='
cd src
$PY train_lora_ot.py --dataset iu --max_items 60 --epochs 1 --bs 2 --accum 2 \
    --out_dir results/smoke_sft
$PY train_lora_ot.py --dataset iu --max_items 60 --epochs 1 --bs 2 --accum 2 \
    --use_ot --token_marginal salience \
    --out_dir results/smoke_ot
echo 'BOOTSTRAP_ALL_DONE'
