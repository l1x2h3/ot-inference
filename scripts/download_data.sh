#!/bin/bash
# OT-LoRA project data downloader (via hf-mirror.com)
set -x
M=https://hf-mirror.com
D=data

# 1) MLforHealthcare/mimic-cxr (~792MB)
cd $D/mimic_mlf
for f in test-00000-of-00001.parquet train-00000-of-00002.parquet train-00001-of-00002.parquet validation-00000-of-00001.parquet; do
  curl -sL --retry 5 --retry-delay 3 -o "$f" "$M/datasets/MLforHealthcare/mimic-cxr/resolve/main/data/$f" || echo "FAIL $f"
done
echo "MLF_DONE $(du -sh . | cut -f1)"

# 2) dz-osamu/IU-Xray (~1.1GB)
cd $D/iu_xray
for f in image.zip train.jsonl val.jsonl test.jsonl; do
  curl -sL --retry 5 --retry-delay 3 -o "$f" "$M/datasets/dz-osamu/IU-Xray/resolve/main/$f" || echo "FAIL $f"
done
echo "IU_DONE $(du -sh . | cut -f1)"

# 3) Yamini MIMIC-CXR-RRG test 2461 (~8GB, 16 shards) - external eval
cd $D/mimic_rrg_test
for i in $(seq -w 0 15); do
  f="test-000$i-of-00016.parquet"
  [ -s "$f" ] || curl -sL --retry 5 --retry-delay 3 -o "$f" "$M/datasets/Yamini-1628/MIMIC-CXR-RRG/resolve/main/findings_section/$f" || echo "FAIL $f"
done
echo "RRG_DONE $(du -sh . | cut -f1)"
echo "ALL_DOWNLOADS_COMPLETE"
