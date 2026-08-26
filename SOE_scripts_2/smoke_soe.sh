#!/bin/bash
# smoke_soe.sh -- end-to-end mini validation of the SCOUT-aligned SOE pipeline
# on ONE GPU (~15 min). Exercises every stage the real chain uses:
#   1. env imports (torch/robomimic/robosuite/pytorch3d/diffusers/wandb)
#   2. config generation + 2-epoch DPExt training on the seed core
#   3. run_scout_align: 4 scenes x 2 retries, 2 shards, with wandb DISABLED
#   4. vis_validate_soe on the demo.hdf5
#   5. extract_and_combine -> demo_plus_core.hdf5
#   6. 2-epoch retrain on demo_plus_core (train_success path)
# Results land in $SMOKE_ROOT; nothing writes to wandb or the real DATA_ROOT.
#
# Usage: GPU=7 TSEED=233 bash smoke_soe.sh
set -u

: "${GPU:?GPU required}" "${TSEED:?TSEED required}"
SOE_REPO=${SOE_REPO:-/root/workspace/baojiachun/SOE}
SOE_SCRIPTS_2=${SOE_SCRIPTS_2:-/root/workspace/baojiachun/SOE_scripts_2}
VENV=${VENV:-/root/workspace/baojiachun/.venv_soe/bin/python}
DATASETS=${DATASETS:-/root/workspace/baojiachun/soe_data/datasets/can}
SMOKE_ROOT=${SMOKE_ROOT:-/root/workspace/baojiachun/soe_data/smoke_s${TSEED}}
export TMPDIR=/tmp

mkdir -p "$SMOKE_ROOT"
step() { echo; echo "===== [smoke] $* ====="; }

step "1/6 imports"
( cd "$SOE_REPO/src" && "$VENV" - <<'PY'
import torch, robomimic, robosuite, diffusers, h5py, numpy
import pytorch3d, easydict, einops, scipy, click, wandb
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0))
print("robomimic", robomimic.__version__, "robosuite", robosuite.__version__)
print("imports OK")
PY
) || { echo "IMPORTS FAILED"; exit 1; }

CORE="$DATASETS/can_core_soe_s${TSEED}.hdf5"
[ -f "$CORE" ] || { echo "core missing: $CORE (run make_core_soe.py first)"; exit 1; }

step "2/6 train 2 epochs"
TR="$SMOKE_ROOT/train"; rm -rf "$TR"; mkdir -p "$TR"
"$VENV" "$SOE_SCRIPTS_2/gen_config_soe.py" --template "$SOE_REPO/simulation/config_template/can_soe.json" \
  --seed "$TSEED" --dataset "$CORE" --filter-key core_20 \
  --log-dir "$TR/logs/smoke_seed_${TSEED}" --out "$TR/cfg.json" \
  --num-epochs 2 --save-epochs 1 --num-workers 4 || exit 1
( cd "$SOE_REPO/src" && CUDA_VISIBLE_DEVICES=$GPU "$VENV" train_single_gpu.py --config "$TR/cfg.json" ) || exit 1
TRY=$(ls "$TR/logs/smoke_seed_${TSEED}" | tail -1)
CKPT="$TR/logs/smoke_seed_${TSEED}/$TRY/ckpt/policy_last.ckpt"
TCFG="$TR/logs/smoke_seed_${TSEED}/$TRY/config.json"
[ -f "$CKPT" ] || { echo "no ckpt"; exit 1; }
echo "ckpt: $CKPT"

step "3/6 eval 4 scenes x 2 retries (2 shards)"
RD="$SMOKE_ROOT/rollout"; rm -rf "$RD"; mkdir -p "$RD"
( cd "$SOE_REPO/simulation" && \
  CUDA_VISIBLE_DEVICES=$GPU SOE_RENDER_GPU=$GPU MUJOCO_GL=egl "$VENV" run_scout_align.py orchestrate \
    --agent "$CKPT" --config "$TCFG" --out-dir "$RD" \
    --n-scenes 4 --seed-base 42 --try-times 2 --n-shards 2 \
    --horizon 300 --noise-scale 2.0 ) || exit 1

step "4/6 vis validate"
"$VENV" "$SOE_SCRIPTS_2/vis_validate_soe.py" "$RD/demo.hdf5" || exit 1

step "5/6 extract + combine"
"$VENV" "$SOE_SCRIPTS_2/soe_round_step.py" extract_and_combine \
  --demo "$RD/demo.hdf5" --core "$CORE" --used-demo core_20 --n-rollouts 4 || exit 1

step "6/6 retrain 2 epochs on demo_plus_core"
TR2="$SMOKE_ROOT/train2"; rm -rf "$TR2"; mkdir -p "$TR2"
"$VENV" "$SOE_SCRIPTS_2/gen_config_soe.py" --template "$SOE_REPO/simulation/config_template/can_soe.json" \
  --seed "$TSEED" --dataset "$RD/demo_plus_core.hdf5" --filter-key train_success \
  --log-dir "$TR2/logs/smoke2_seed_${TSEED}" --out "$TR2/cfg.json" \
  --num-epochs 2 --save-epochs 1 --num-workers 4 || exit 1
( cd "$SOE_REPO/src" && CUDA_VISIBLE_DEVICES=$GPU "$VENV" train_single_gpu.py --config "$TR2/cfg.json" ) || exit 1

echo
echo "===== SMOKE PASSED ====="
"$VENV" -c "import json;print(json.load(open('$RD/metrics.json')))"
