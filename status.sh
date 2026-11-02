#!/bin/bash
# Show training progress:  bash status.sh
cd "$(dirname "$0")"

echo "=== Queue"
if pgrep -f "train_all.sh --worker" > /dev/null; then
    echo "train_all.sh: running"
else
    echo "train_all.sh: NOT running (start or resume with: bash train_all.sh)"
fi
cat runs/train_all.log 2>/dev/null

# current run = most recently written runs/*/train.log
log=$(ls -t runs/*/train.log 2>/dev/null | head -1)
if [ -n "$log" ]; then
    echo
    echo "=== Current run: $(basename "$(dirname "$log")")"
    grep -E "VAL Results|^AUC" "$log" | paste - - \
        | sed -E 's/VAL Results \(Epoch ([0-9]+)\):[[:space:]]*AUC[[:space:]]*:[[:space:]]*/epoch \1   val AUC /'
    grep -E "Step|Exp Loss" "$log" | tail -1
    grep -E "Early stopping|^Test AUC|Training completed|Traceback|Error" "$log" | tail -3
fi

echo
echo "=== Celeb-DF face extraction"
echo "$(ls -d /data/datasets/deepfake_datasets/Celeb-DF-v2/*/frames/* 2>/dev/null | wc -l) / 6529 videos"
