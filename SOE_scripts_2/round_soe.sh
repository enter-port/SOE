#!/bin/bash
# round_soe.sh -- SCOUT-aligned SOE campaign: ONE SOE-internal round (k=0..5)
# of ONE seed on ONE GPU.
#
# SOE-internal round k:
#   [1/2] train DPExt (train_single_gpu.py)
#         k=0   : on the seed's 20-demo SOE core (mask core_20), epochs=$EP0,
#                 wandb run SOE-s$TSEED-BASE-round0 (created fresh)
#         k>=1  : on round k-1's demo_plus_core.hdf5 (mask train_success),
#                 epochs=$EPOCHS, wandb resumes the run created by round k-1's
#                 eval (= SCOUT round k's run)
#   [2/2] eval+rescue (simulation/run_scout_align.py):
#         100 scenes seeded 42..141 (SCOUT-identical), horizon=$HORIZON,
#         retries=$ETRIES per failed scene from the same init state with the
#         DPExt exploration extension ($NOISE). Render-integrity gate
#         (vis_validate_soe) with one reduced-shard retry.
#         then extract_useful_data_v2 + dataset_combine -> demo_plus_core.hdf5
#
# SCOUT round mapping: SOE round k eval == SCOUT round k+1
# (6 SOE evals == SCOUT r1..r6; 6 trainings: BASE + r1..r5's).
#
# Usage:  GPU=7 TSEED=233 DATA_ROOT=<abs> WPROJ=CAN-8-24-SOE-s233 \
#         bash round_soe.sh can <k>
set -u

TASK=${1:-can}
K=${2:-}
if [ -z "$K" ]; then echo "usage: $0 <task> <soe-round 0..5>" >&2; exit 2; fi

# ---- required env ----------------------------------------------------- #
: "${GPU:?GPU required}" "${TSEED:?TSEED required}" "${DATA_ROOT:?DATA_ROOT required}" "${WPROJ:?WPROJ required}"

# ---- tunables (defaults) ---------------------------------------------- #
SOE_REPO=${SOE_REPO:-/root/workspace/baojiachun/SOE}
SOE_SCRIPTS_2=${SOE_SCRIPTS_2:-/root/workspace/baojiachun/SOE_scripts_2}
VENV=${VENV:-/root/workspace/baojiachun/.venv_soe/bin/python}
DATASETS=${DATASETS:-/root/workspace/baojiachun/soe_data/datasets/can}
SCENES=${SCENES:-100}
ETRIES=${ETRIES:-10}
BASE_SEED=${BASE_SEED:-42}
HORIZON=${HORIZON:-300}
NOISE=${NOISE:-2.0}
NSHARDS=${NSHARDS:-8}
NSHARDS_RETRY=${NSHARDS_RETRY:-4}
EP0=${EP0:-1000};  SAVE0=${SAVE0:-50}
EPOCHS=${EPOCHS:-1000}; SAVE=${SAVE:-50}
WORKERS=${WORKERS:-8}
# VISGATE: 1 = run the vis_validate_soe render gate (thresholds calibrated on
# CAN: agentview tstd>20 = noise, <1 = frozen). 0 = skip the gate entirely.
# SQUARE must run with VISGATE=0: healthy square tstd is 17.6-27.4 (measured
# 08-26 on square core re-renders), overlapping can's >20 noise line -- the
# gate false-killed both square SCOUT arms on 08-26 and the user ruled it off
# for square (same ruling applies here).
VISGATE=${VISGATE:-1}
DRY_RUN=${DRY_RUN:-0}
SKIP_ROLLOUT=${SKIP_ROLLOUT:-0}
FORCE_TRAIN=${FORCE_TRAIN:-0}
WANDB_ENV=${WANDB_ENV:-/root/workspace/baojiachun/.secrets/wandb.env}

# ---- paths ------------------------------------------------------------ #
D="$DATA_ROOT/$TASK"
RLOG="$D/round.log"
TRAIN="$D/train/round_$K"
LOGROOT="$TRAIN/logs/round_${K}_seed_${TSEED}"
RDIR="$D/rollout/round_$K"
PREV_RDIR="$D/rollout/round_$((K-1))"
CORE="$DATASETS/${TASK}_core_soe_s${TSEED}.hdf5"
TEMPLATE=${TEMPLATE:-$SOE_REPO/simulation/config_template/${TASK}_soe.json}
mkdir -p "$D" "$TRAIN"

log() {
  local line="[$(date +'%Y-%m-%d %H:%M:%S')] $*"
  echo "$line"
  [ "$DRY_RUN" = "1" ] || echo "$line" >> "$RLOG"
}
run() { # run <cmd...>  (DRY_RUN=1 -> print only)
  if [ "$DRY_RUN" = "1" ]; then echo "[dry-run] $*"; else "$@"; fi
}

# wandb credentials (isolated key file, never /root/.netrc)
if [ "$DRY_RUN" != "1" ] && [ -f "$WANDB_ENV" ]; then
  set -a; . "$WANDB_ENV"; set +a
fi
export TMPDIR=/tmp
export CUBLAS_WORKSPACE_CONFIG=:4096:8

T0=$(date +%s)
log "=== ROUND $TASK soe-seed=$TSEED soe-round=$K START (GPU$GPU; SCOUT-round=$((K+1))) ==="

if [ "$K" = "0" ]; then
  DS="$CORE"; FKEY="core_20"; EP=$EP0; SV=$SAVE0; RID=""; WBASE="SOE-s${TSEED}-BASE-round0"
else
  DS="$PREV_RDIR/demo_plus_core.hdf5"; FKEY="train_success"; EP=$EPOCHS; SV=$SAVE; RID=""
  MJSON="$PREV_RDIR/metrics.json"
  if [ -f "$MJSON" ]; then
    RID=$("$VENV" -c "import json;print(json.load(open('$MJSON')).get('wandb_run_id') or '')" 2>/dev/null || true)
  fi
  if [ ! -f "$DS" ]; then
    log "FATAL: round $((K-1)) dataset missing: $DS"
    exit 1
  fi
  [ -n "$RID" ] || log "[warn] no wandb_run_id in $MJSON -- train will create a fresh run"
fi

# ---- [1/2] train ------------------------------------------------------ #
TRY=""
if [ "$FORCE_TRAIN" != "1" ]; then
  TRY=$("$VENV" "$SOE_SCRIPTS_2/soe_round_step.py" latest_try --log-root "$LOGROOT" 2>/dev/null | tail -1 || true)
fi
if [ -n "$TRY" ] && [ -f "$LOGROOT/$TRY/ckpt/policy_last.ckpt" ]; then
  log "[1/2] train SKIP (existing ckpt in try $TRY)"
else
  CFG="$TRAIN/config_round_${K}_seed_${TSEED}.json"
  log "[1/2] train: ep=$EP save=$SV seed=$TSEED ds=$DS filter=$FKEY -> $TRAIN"
  run "$VENV" "$SOE_SCRIPTS_2/gen_config_soe.py" --template "$TEMPLATE" \
      --seed "$TSEED" --dataset "$DS" --filter-key "$FKEY" \
      --log-dir "$LOGROOT" --out "$CFG" \
      --num-epochs "$EP" --save-epochs "$SV" --num-workers "$WORKERS"
  if [ "$DRY_RUN" != "1" ]; then
    # live-stream the loss curve so the wandb project/run exists from the
    # first minutes of training (waits for the try dir via --auto; the
    # post-hoc upload below stays as a net -- state file dedupes)
    if [ -n "$RID" ]; then
      nohup "$VENV" "$SOE_SCRIPTS_2/wandb_train_log.py" --tail --auto \
          --log-txt "$LOGROOT" --project "$WPROJ" --run-id "$RID" \
          --tail-max-min 600 >> "$TRAIN/wandb_tail.log" 2>&1 &
    else
      nohup "$VENV" "$SOE_SCRIPTS_2/wandb_train_log.py" --tail --auto \
          --log-txt "$LOGROOT" --project "$WPROJ" --run-name "$WBASE" \
          --config-json "$CFG" --tail-max-min 600 >> "$TRAIN/wandb_tail.log" 2>&1 &
    fi
    ( cd "$SOE_REPO/src" && CUDA_VISIBLE_DEVICES=$GPU "$VENV" train_single_gpu.py \
        --config "$CFG" ) >> "$TRAIN/train.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then log "FATAL: train rc=$rc (see $TRAIN/train.log)"; exit 1; fi
  fi
  TRY=$("$VENV" "$SOE_SCRIPTS_2/soe_round_step.py" latest_try --log-root "$LOGROOT" | tail -1)
  # upload loss curve into the round's shared run (BASE run for k=0)
  if [ -n "$RID" ]; then
    run "$VENV" "$SOE_SCRIPTS_2/wandb_train_log.py" --log-txt "$LOGROOT/$TRY/log.txt" \
        --project "$WPROJ" --run-id "$RID" || true
  else
    run "$VENV" "$SOE_SCRIPTS_2/wandb_train_log.py" --log-txt "$LOGROOT/$TRY/log.txt" \
        --project "$WPROJ" --run-name "$WBASE" --config-json "$LOGROOT/$TRY/config.json" || true
  fi
  log "[1/2] train done (try $TRY)"
fi
CKPT="$LOGROOT/$TRY/ckpt/policy_last.ckpt"
TCFG="$LOGROOT/$TRY/config.json"
[ -f "$CKPT" ] || { log "FATAL: ckpt missing $CKPT"; exit 1; }

# ---- [2/2] eval + rescue + combine ------------------------------------ #
WNAME="SOE-s${TSEED}-round$((K+1))"
if [ -f "$RDIR/demo_plus_core.hdf5" ]; then
  log "[2/2] eval SKIP (demo_plus_core.hdf5 exists)"
  log "=== ROUND $TASK soe-seed=$TSEED soe-round=$K TOTAL: $(( $(date +%s) - T0 ))s ==="
  exit 0
fi
if [ "$SKIP_ROLLOUT" = "1" ]; then
  [ -f "$RDIR/demo.hdf5" ] || { log "FATAL: SKIP_ROLLOUT=1 but $RDIR/demo.hdf5 missing"; exit 1; }
  log "[2/2] rollout SKIP (SKIP_ROLLOUT=1, reusing $RDIR/demo.hdf5)"
else
  NS_NOW=$NSHARDS
  for ATTEMPT in 1 2; do
    log "[2/2] rollout attempt $ATTEMPT: scenes=$SCENES seeds=$BASE_SEED.. tries=$ETRIES horizon=$HORIZON noise=$NOISE shards=$NS_NOW ckpt=$CKPT -> $RDIR"
    if [ "$DRY_RUN" != "1" ]; then
      mkdir -p "$RDIR"
      ( cd "$SOE_REPO/simulation" && \
        CUDA_VISIBLE_DEVICES=$GPU SOE_RENDER_GPU=$GPU MUJOCO_GL=egl "$VENV" run_scout_align.py orchestrate \
          --agent "$CKPT" --config "$TCFG" --out-dir "$RDIR" \
          --task "$TASK" --exp-num $((K+1)) \
          --n-scenes "$SCENES" --seed-base "$BASE_SEED" --try-times "$ETRIES" \
          --n-shards "$NS_NOW" --horizon "$HORIZON" --noise-scale "$NOISE" \
          --metrics-json "$RDIR/metrics.json" \
          --wandb-project "$WPROJ" --wandb-run-name "$WNAME" ) >> "$RDIR/rollout.stdout" 2>&1
      rc=$?
    else
      run echo "(rollout as above)"; rc=0
    fi
    if [ $rc -ne 0 ]; then
      log "FATAL: rollout rc=$rc (see $RDIR/rollout.stdout)"; exit 1
    fi
    if [ "$DRY_RUN" != "1" ]; then
      if [ "$VISGATE" != "1" ]; then
        log "[2/2] render gate SKIPPED (VISGATE=0; square mode)"
        break
      fi
      if "$VENV" "$SOE_SCRIPTS_2/vis_validate_soe.py" "$RDIR/demo.hdf5" > "$RDIR/validate.log" 2>&1; then
        log "[2/2] rollout images HEALTHY (attempt $ATTEMPT, shards=$NS_NOW)"
        break
      else
        cat "$RDIR/validate.log"
        if [ "$ATTEMPT" = "2" ]; then
          log "FATAL: render corruption persists after retry (see $RDIR/validate.log)"
          exit 1
        fi
        log "[2/2] render CORRUPT -- cleaning and retrying with shards=$NSHARDS_RETRY"
        rm -f "$RDIR"/demo.hdf5 "$RDIR"/shard_*_phase*.hdf5 "$RDIR"/shard_*.log "$RDIR"/failed_inits.h5 "$RDIR"/env_meta_used.json
        NS_NOW=$NSHARDS_RETRY
      fi
    else
      break
    fi
  done
fi

# extract rescued successes + combine into the accumulated train dataset
if [ "$K" = "0" ]; then
  COMB_CORE="$CORE"; COMB_KEY="core_20"
else
  COMB_CORE="$PREV_RDIR/demo_plus_core.hdf5"; COMB_KEY="train_success"
fi
log "[2/2] extract+combine: demo=$RDIR/demo.hdf5 core=$COMB_CORE key=$COMB_KEY"
run "$VENV" "$SOE_SCRIPTS_2/soe_round_step.py" extract_and_combine \
    --demo "$RDIR/demo.hdf5" --core "$COMB_CORE" --used-demo "$COMB_KEY" --n-rollouts "$SCENES"

if [ "$DRY_RUN" != "1" ]; then
  SR=$("$VENV" -c "import json;m=json.load(open('$RDIR/metrics.json'));print('%.3f'%m['success_rate'])" 2>/dev/null || echo NA)
  P10=$("$VENV" -c "import json;m=json.load(open('$RDIR/metrics.json'));print('%.3f'%m['pass_at_10'])" 2>/dev/null || echo NA)
  log "[2/2] eval result: SR=$SR pass@10=$P10 (run $WNAME)"
fi

# disk hygiene (2026-08-29, PRUNE_ACCUM=1 for the square campaign): round K's
# demo_plus_core was just built FROM round K-1's (see extract+combine above);
# the chain never reads $PREV_RDIR/demo_plus_core.hdf5 again. Raw demo.hdf5 +
# metrics are kept forever; the accum is rebuildable via extract_and_combine.
if [ "${PRUNE_ACCUM:-0}" = "1" ] && [ "$K" -ge 1 ] && [ "$DRY_RUN" != "1" ] \
   && [ -f "$RDIR/demo_plus_core.hdf5" ]; then
  rm -f "$PREV_RDIR/demo_plus_core.hdf5"
  log "[prune] removed superseded $PREV_RDIR/demo_plus_core.hdf5 (raw demo.hdf5 kept)"
fi
log "=== ROUND $TASK soe-seed=$TSEED soe-round=$K TOTAL: $(( $(date +%s) - T0 ))s ==="
