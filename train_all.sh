#!/bin/bash
# Train every model one after another on a single GPU.
#
#   bash train_all.sh
#
# - Starts itself in the background (survives closing the terminal) and logs to runs/train_all.log.
# - Each run writes to runs/<name>/ (train.log, checkpoints/).
# - Finished runs are skipped, so re-running after an interruption resumes the queue.
# - Waits for any training that is already running instead of starting a second one.
# - Only one copy can run at a time; running it again just says so.
set -u
SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SCRIPT")"
mkdir -p runs

if [ "${1:-}" != "--worker" ]; then
    if ! flock -n runs/.train_all.lock true; then
        echo "train_all.sh is already running. Progress: cat runs/train_all.log"
        exit 0
    fi
    nohup setsid bash "$SCRIPT" --worker >> runs/train_all.log 2>&1 < /dev/null &
    echo "Started in the background. Progress: cat runs/train_all.log"
    exit 0
fi

exec 9> runs/.train_all.lock
flock -n 9 || exit 0

# stage 1 env; USE_TF=0 stops transformers importing the broken TensorFlow in ~/.local
POP="env USE_TF=0 /data1/Umesh/miniconda3/envs/pop/bin/python"
# stage 2 env (TinyLlama needs newer transformers); ignore ~/.local packages entirely
NEW="env PYTHONNOUSERSITE=1 /data1/Umesh/miniconda3/envs/deepfake/bin/python"
CELEBDF=/data/datasets/deepfake_datasets/Celeb-DF-v2
CELEBDF_VIDEOS=6529

log() { echo "[$(date '+%F %T')] $*"; }

wait_for_running_training() {
    while pgrep -f "train.py .*hydra.run.dir" > /dev/null; do sleep 60; done
}

# $1 = run name, $2 = config name, $3 = python
train() {
    wait_for_running_training
    if grep -qs "Training completed" runs/$1/train.log; then
        log "SKIP $1 (already finished)"
        return 0
    fi
    log "START $1 ($2)"
    rm -rf runs/$1 && mkdir -p runs/$1
    PYTHONUNBUFFERED=1 $3 train.py --config-name $2 \
        hydra.run.dir=$PWD/runs/$1 hydra.job.chdir=True > runs/$1/train.log 2>&1
    if grep -q "Training completed" runs/$1/train.log; then
        log "DONE $1 | $(grep -E '^Test AUC' runs/$1/train.log | tail -1)"
    else
        log "FAILED $1, see runs/$1/train.log"
        return 1
    fi
}

celebdf_faces_ready() {
    [ "$(ls -d $CELEBDF/*/frames/* 2>/dev/null | wc -l)" -ge $CELEBDF_VIDEOS ]
}

# 1. FF++ detector, then stage 2 explanations on top of it
train ff_stage1_full train_stage1 "$POP" || exit 1
train ff_stage2 train_stage2 "$NEW"

# 2. Celeb-DF face crops (waits for extraction already running, otherwise resumes it)
while pgrep -f extract_celebdf_faces.py > /dev/null; do sleep 60; done
if ! celebdf_faces_ready; then
    log "START Celeb-DF face extraction"
    PYTHONUNBUFFERED=1 $POP scripts/extract_celebdf_faces.py > $CELEBDF/extract_faces.log 2>&1
fi
celebdf_faces_ready || { log "Celeb-DF face extraction incomplete, see $CELEBDF/extract_faces.log"; exit 1; }
log "Celeb-DF faces ready"

# 3. detectors that use Celeb-DF / DeepFakeFace
train celebdf_stage1 train_celebdf_stage1 "$POP"
train multisource_ffpp_celebdf train_multisource_ffpp_celebdf_stage1 "$POP"
train multisource_ffpp_celebdf_dff train_multisource_ffpp_celebdf_dff_stage1 "$POP"

log "ALL DONE"
