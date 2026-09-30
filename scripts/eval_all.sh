#!/bin/bash
# Morning full-eval batch. Usage: bash eval_all.sh <gpu> <config1> [config2 ...]
# Each config = results/<name>/adapter_ep1, evaluated on MIMIC test (full 4596)
# with greedy + min_new=45. External sets (rrg_test/iu) for headline configs.
PY=python3
SRC=src
GPU=$1; shift
for cfg in "$@"; do
  echo "[$(date +%H:%M)] GPU$GPU eval $cfg (mimic full)"
  CUDA_VISIBLE_DEVICES=$GPU $PY $SRC/eval_metrics.py \
    --adapter results/$cfg/adapter_ep1 \
    --dataset mimic_mlf --split test --bs 16 --min_new 45 --tag $cfg \
    > results/logs/eval_$cfg.log 2>&1 \
    && echo "[$(date +%H:%M)] done $cfg" || echo "[$(date +%H:%M)] FAIL $cfg"
done
echo "EVAL_BATCH_DONE GPU$GPU: $*"
