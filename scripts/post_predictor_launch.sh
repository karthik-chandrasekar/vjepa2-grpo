#!/usr/bin/env bash
# Run after predictor's final.pt lands: builds FAISS, launches critic.
set -euo pipefail
source /root/venv/bin/activate
cd /workspace/vjepa2_grpo

# 1. confirm predictor actually finished
if [ ! -f /workspace/checkpoints/predictor/final.pt ]; then
    echo "FATAL: /workspace/checkpoints/predictor/final.pt not found"; exit 1
fi
echo "[ok] predictor final.pt present ($(du -h /workspace/checkpoints/predictor/final.pt | cut -f1))"

# 2. FAISS anchor build (~15 min)
echo "=== building FAISS anchor buffer ==="
python scripts/build_faiss.py \
    --emb-root /workspace/data/embeddings \
    --out /workspace/data/anchor_buffer \
    --nlist 4096 --subsample 5

# 3. launch critic (under nohup, detached — survives shell exit)
echo "=== launching critic ==="
mkdir -p /workspace/logs
nohup python -u scripts/train_critic.py --config configs/critic.yaml \
    > /workspace/logs/critic_train.log 2>&1 &
CRITIC_PID=$!
echo "launched critic PID $CRITIC_PID"
sleep 8

N=$(ps aux | grep train_critic | grep -v grep | wc -l)
echo "$N critic process(es) running"
if [ "$N" -lt 1 ]; then echo "FATAL: critic died at launch"; tail -30 /workspace/logs/critic_train.log; exit 1; fi

echo "=== first 30 lines of critic log ==="
sleep 5
tail -30 /workspace/logs/critic_train.log
