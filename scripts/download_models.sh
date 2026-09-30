#!/bin/bash
# Download Qwen2.5-VL-3B-Instruct + CheXbert via hf-mirror
set -e
M=https://hf-mirror.com
BASE=models

dl_repo () {  # dl_repo <repo_id> <dest_dir>
  local repo=$1 dest=$2
  mkdir -p "$dest"
  curl -s --max-time 30 "$M/api/models/$repo/tree/main" | python3 -c "
import json,sys
for f in json.load(sys.stdin):
    if f['type']=='file' and not f['path'].startswith('.'):
        print(f['path'])
" | while read -r f; do
    [ -s "$dest/$f" ] || curl -sL --retry 5 --retry-delay 3 -o "$dest/$f" "$M/$repo/resolve/main/$f"
    echo "  got $f"
  done
}

echo '=== Qwen2.5-VL-3B-Instruct (~7GB) ==='
dl_repo Qwen/Qwen2.5-VL-3B-Instruct $BASE/Qwen2.5-VL-3B-Instruct
echo '=== CheXbert (oscarloch/rrg-grpo_chexbert, 450MB) ==='
dl_repo oscarloch/rrg-grpo_chexbert $BASE/chexbert
echo "MODELS_DONE $(du -sh $BASE | cut -f1)"
