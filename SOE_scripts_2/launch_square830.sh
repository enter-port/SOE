#!/bin/bash
# launch_square830.sh -- SOE square 3-seed rerun with DDIM eta=1 (user order 2026-08-30)
# wandb project SQUARE-8-30-SOE-s{seed}; DATA_ROOT soe_data/2026_8_30_soe; GPUs 2/3/5.
# Env below = SQUARE-8-29-SOE protocol, reconstructed from its round.log artifacts
# (first relaunch attempt 08-30 20:48 died on the DATASETS default=can; 8-29 log
# lines show save=200 -> SAVE0/SAVE=200, horizon=500, VISGATE=0 per user ruling).
set -eu
BASE=/root/workspace/baojiachun
LOGROOT=$BASE/soe_data/2026_8_30_soe/logs
mkdir -p "$LOGROOT"

launch() {
  local SEED=$1 GPU=$2
  local ROOT=$BASE/soe_data/2026_8_30_soe/SOE-s$SEED
  mkdir -p "$ROOT"
  tmux new-session -d -s "soe_sq${SEED}_chain" \
    "GPU=$GPU TSEED=$SEED DATA_ROOT=$ROOT WPROJ=SQUARE-8-30-SOE-s$SEED \
     DATASETS=$BASE/soe_data/datasets/square HORIZON=500 SAVE0=200 SAVE=200 VISGATE=0 \
     bash $BASE/SOE_scripts_2/chain_soe.sh square 2>&1 | tee -a $LOGROOT/soe_sq${SEED}_chain.console.log"
  echo "launched soe_sq${SEED}_chain GPU$GPU -> $ROOT"
}

launch 233 2
launch 2333 3
launch 23333 5
