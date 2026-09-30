#!/bin/bash
# Parallel main-table training: 9 configs across GPUs 1-7 (GPU0 reserved for evals).
# Each GPU runs its job queue sequentially; all logs under results/logs/.
PY=python3
SRC=src
R=results
mkdir -p $R/logs

run () {  # run <gpu> <name> <extra args...>
  local gpu=$1 name=$2; shift 2
  echo "[$(date +%H:%M:%S)] GPU$gpu START $name"
  CUDA_VISIBLE_DEVICES=$gpu $PY $SRC/train_lora_ot.py \
    --dataset mimic_mlf --bs 8 --accum 2 --epochs 2 \
    --out_dir $R/$name "$@" > $R/logs/$name.log 2>&1 \
    && echo "[$(date +%H:%M:%S)] GPU$gpu DONE  $name" \
    || echo "[$(date +%H:%M:%S)] GPU$gpu FAIL  $name (see $R/logs/$name.log)"
}

worker () {  # worker <gpu> <job...>
  local gpu=$1; shift
  for j in "$@"; do
    case $j in
      sft_s42)    run $gpu sft_s42    --seed 42 ;;
      sft_s1337)  run $gpu sft_s1337  --seed 1337 ;;
      sft_s9233)  run $gpu sft_s9233  --seed 9233 ;;
      otuni_s42)   run $gpu otuni_s42   --seed 42 --use_ot --token_marginal uniform ;;
      otuni_s1337) run $gpu otuni_s1337 --seed 1337 --use_ot --token_marginal uniform ;;
      otuni_s9233) run $gpu otuni_s9233 --seed 9233 --use_ot --token_marginal uniform ;;
      otsal_s42)   run $gpu otsal_s42   --seed 42 --use_ot --token_marginal salience ;;
      otsal_s1337) run $gpu otsal_s1337 --seed 1337 --use_ot --token_marginal salience ;;
      otsal_s9233) run $gpu otsal_s9233 --seed 9233 --use_ot --token_marginal salience ;;
    esac
  done
}

# GPU1-7 take the first 7 jobs; two GPUs take a second queued job.
worker 1 sft_s42 otuni_s9233 &
worker 2 sft_s1337 otsal_s9233 &
worker 3 sft_s9233 &
worker 4 otuni_s42 &
worker 5 otuni_s1337 &
worker 6 otsal_s42 &
worker 7 otsal_s1337 &
wait
echo "ALL_MAIN_TRAINING_DONE $(date)"
