#!/bin/bash
# chain_soe.sh -- full SOE-internal round chain (0..5) for ONE seed on ONE GPU.
#
# Usage:  GPU=7 TSEED=233 DATA_ROOT=/root/workspace/baojiachun/soe_data/2026_8_26_soe/SOE-s233 \
#         WPROJ=CAN-8-24-SOE-s233 bash chain_soe.sh can
# Env:    CHAIN_START=0 CHAIN_END=5  (resume window)
#         + all round_soe.sh tunables pass through.
set -u

TASK=${1:-can}
S=${CHAIN_START:-0}
E=${CHAIN_END:-5}

: "${GPU:?GPU required}" "${TSEED:?TSEED required}" "${DATA_ROOT:?DATA_ROOT required}" "${WPROJ:?WPROJ required}"
SOE_SCRIPTS_2=${SOE_SCRIPTS_2:-/root/workspace/baojiachun/SOE_scripts_2}

mkdir -p "$DATA_ROOT/$TASK" "$DATA_ROOT/logs"

for k in $(seq "$S" "$E"); do
  echo "[chain] === SOE round $k (SCOUT round $((k+1))) start $(date +'%F %T') ==="
  if ! GPU=$GPU TSEED=$TSEED DATA_ROOT=$DATA_ROOT WPROJ=$WPROJ \
       bash "$SOE_SCRIPTS_2/round_soe.sh" "$TASK" "$k"; then
    echo "[chain] FAILED at round $k -- see $DATA_ROOT/$TASK/round.log" >&2
    exit 1
  fi
done
echo "[chain] ALL DONE (rounds $S..$E) $(date +'%F %T')"
