#!/bin/bash
# Final experiment batch: full-test reranking headline + externals + seeds + 7B
PY=python3
SRC=src
R=results
D=data/processed

# Lane A (GPU0): FULL test set headline — sample k=8 on all 4596 + rerank + eval
laneA () {
  CUDA_VISIBLE_DEVICES=0 $PY $SRC/sample_k.py --adapter $R/sft_s42/adapter_ep1 \
    --dataset mimic_mlf --max_items 0 --k 8 --tag sft42_full > $R/logs/full_sample.log 2>&1
  CUDA_VISIBLE_DEVICES=0 $PY $SRC/rerank_ot.py --adapter $R/sft_s42/adapter_ep1 \
    --dataset mimic_mlf --cands_tag sft42_full > $R/logs/full_rerank.log 2>&1
  for s in greedy ot logprob hybrid; do
    CUDA_VISIBLE_DEVICES=0 $PY -c "
import sys; sys.path.insert(0,'$SRC')
from eval_metrics import compute_nlg
compute_nlg('$D/mimic_mlf/sft42_full_${s}_pred.jsonl', '$D/mimic_mlf/sft42_full_${s}_pred_nlg.json')" \
      >> $R/logs/full_eval.log 2>&1
  done
  echo "LANE_A_DONE"
}

# Lane B (GPU1): RRG-2461 external
laneB () {
  CUDA_VISIBLE_DEVICES=1 $PY $SRC/sample_k.py --adapter $R/sft_s42/adapter_ep1 \
    --dataset rrg_test --max_items 0 --k 8 --tag rrg_sft42 > $R/logs/rrg_sample.log 2>&1
  CUDA_VISIBLE_DEVICES=1 $PY $SRC/rerank_ot.py --adapter $R/sft_s42/adapter_ep1 \
    --dataset rrg_test --cands_tag rrg_sft42 > $R/logs/rrg_rerank.log 2>&1
  for s in greedy ot logprob hybrid; do
    CUDA_VISIBLE_DEVICES=1 $PY -c "
import sys; sys.path.insert(0,'$SRC')
from eval_metrics import compute_nlg
compute_nlg('$D/rrg_test/rrg_sft42_${s}_pred.jsonl', '$D/rrg_test/rrg_sft42_${s}_pred_nlg.json')" \
      >> $R/logs/rrg_eval.log 2>&1
  done
  echo "LANE_B_DONE"
}

# Lane C (GPU2): IU external + 2 extra seeds on 1500-item MIMIC subset
laneC () {
  CUDA_VISIBLE_DEVICES=2 $PY $SRC/sample_k.py --adapter $R/sft_s42/adapter_ep1 \
    --dataset iu --max_items 0 --k 8 --tag iu_sft42 > $R/logs/iu_sample.log 2>&1
  CUDA_VISIBLE_DEVICES=2 $PY $SRC/rerank_ot.py --adapter $R/sft_s42/adapter_ep1 \
    --dataset iu --cands_tag iu_sft42 > $R/logs/iu_rerank.log 2>&1
  echo "LANE_C_IU_DONE"
  for seed in s1337 s9233; do
    CUDA_VISIBLE_DEVICES=2 $PY $SRC/sample_k.py --adapter $R/sft_$seed/adapter_ep1 \
      --dataset mimic_mlf --max_items 1500 --k 8 --tag sft${seed}_k8 \
      > $R/logs/sample_$seed.log 2>&1
    CUDA_VISIBLE_DEVICES=2 $PY $SRC/rerank_ot.py --adapter $R/sft_$seed/adapter_ep1 \
      --dataset mimic_mlf --cands_tag sft${seed}_k8 > $R/logs/rerank_$seed.log 2>&1
    echo "LANE_C_SEED_${seed}_DONE"
  done
  echo "LANE_C_DONE"
}

# Lane D (GPU3): k sensitivity (k=4, 16) on the 800-item subset
laneD () {
  for k in 4 16; do
    CUDA_VISIBLE_DEVICES=3 $PY $SRC/sample_k.py --adapter $R/sft_s42/adapter_ep1 \
      --dataset mimic_mlf --max_items 800 --k $k --tag sft42_k$k > $R/logs/sample_k$k.log 2>&1
    CUDA_VISIBLE_DEVICES=3 $PY $SRC/rerank_ot.py --adapter $R/sft_s42/adapter_ep1 \
      --dataset mimic_mlf --cands_tag sft42_k$k > $R/logs/rerank_k$k.log 2>&1
    for s in greedy ot logprob hybrid; do
      CUDA_VISIBLE_DEVICES=3 $PY -c "
import sys; sys.path.insert(0,'$SRC')
from eval_metrics import compute_nlg
compute_nlg('$D/mimic_mlf/sft42_k${k}_${s}_pred.jsonl', '$D/mimic_mlf/sft42_k${k}_${s}_pred_nlg.json')" \
        >> $R/logs/k${k}_eval.log 2>&1
    done
    echo "LANE_D_K${k}_DONE"
  done
  echo "LANE_D_DONE"
}

# Lane E (GPU4): download 7B then train SFT baseline
laneE () {
  M=https://hf-mirror.com; DEST=models/Qwen2.5-VL-7B-Instruct
  mkdir -p $DEST
  curl -s --max-time 30 "$M/api/models/Qwen/Qwen2.5-VL-7B-Instruct/tree/main" | python3 -c "
import json,sys
for f in json.load(sys.stdin):
    if f['type']=='file' and not f['path'].startswith('.'): print(f['path'])
" | while read -r f; do
    [ -s "$DEST/$f" ] || curl -sL --retry 5 --retry-delay 3 -o "$DEST/$f" "$M/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/$f"
  done
  echo "LANE_E_DOWNLOAD_DONE $(du -sh $DEST | cut -f1)"
  CUDA_VISIBLE_DEVICES=4 $PY $SRC/train_lora_ot.py --dataset mimic_mlf --bs 4 --accum 4 \
    --epochs 2 --seed 42 --model_path $DEST \
    --out_dir $R/sft7b_s42 > $R/logs/sft7b.log 2>&1
  echo "LANE_E_TRAIN_DONE"
}

laneA & laneB & laneC & laneD & laneE &
wait
echo "FINALE_ALL_DONE $(date)"
