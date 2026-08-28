# SOE_scripts_2 — SCOUT-aligned SOE baseline campaign

Runs 学长's SOE (DPExt + latent-noise exploration) under the **exact experimental
setting of the SCOUT entropy campaign** so the two methods are directly comparable.
All changes live on the SOE repo branch `soe-scout-align` (SOE/ is its own git
repository; SCOUT's code is untouched). Server layout mirrors SCOUT:
`/root/workspace/baojiachun/SOE` (repo), `/root/workspace/baojiachun/SOE_scripts_2`
(this dir), `/root/workspace/baojiachun/.venv_soe` (uv env, torch 2.4.1+cu121 —
H20 sm_90 cannot run SOE's pinned torch 1.13).

## Protocol alignment (SOE side ← SCOUT entropy campaign)

| item | SCOUT (entropy, XMODE=soe) | SOE here | note |
|---|---|---|---|
| data split | 20/200 demos, `rng=default_rng(TSEED).choice(200,20)` | identical indices, extracted into SOE's own abs-6drot core file | cross-verified per-demo lengths vs SCOUT's `can_core.hdf5` |
| scenes | 100, init i seeded `np.random.seed(42+i)` before reset | same mechanism in `run_scout_align.py` worker | SCOUT `collect_initial_states` seeds the numpy global RNG; robosuite draws init randomness from it → bit-identical scenes |
| SR | success on the 100 scenes, 1 try | same (phase 1, exploration OFF) | |
| rescue | failed scenes × 10 retries from the same init state, no early stop | same (phase 2, DPExt exploration extension ON, noise_scale 2.0) | SOE run.py's own retry semantics |
| feedback | core + all rounds' rescued successes (accumulated `success_accum`) | SOE's own `demo_plus_core` chaining (`train_success` mask accumulates) | isomorphic; SOE utilities reused unchanged |
| rounds | 6 evaluated rounds (r1..r6) | SOE-internal rounds 0..5; SOE round k eval ≙ SCOUT round k+1 | 6 trainings: BASE (round0) + rounds 1..5 |
| horizon | 300 (can) | 300 (NOT robomimic's default 400) | |
| seeds | TSEED ∈ {233, 2333, 23333}, eval base seed 42 | same | |
| wandb | `CAN-8-24-entropy-s{seed}`, runs `SCOUT-s{seed}-round{N}` / `DP-...` | `CAN-8-24-SOE-s{seed}`, runs `SOE-s{seed}-round{N}` (+ `SOE-s{seed}-BASE-round0`), keys `eval/success_rate`, `explore/pass@10`, `DP/epoch`, `DP/loss` (+ additive `SOE/{pred,ext,kl,recon}_loss`) | one project per seed, one shared run per round (eval + next train resume the same run) |

### Known, deliberate differences (flagged for review)

1. **Training budget is SOE-native, not SCOUT-matched.** SOE template = 1000
   epochs × 100 iters/epoch (SIME `can_image.json`, 学长's public values) for
   every round; SCOUT retrains 600/300 *full-dataset* epochs. "Epoch" is not the
   same unit across the two codebases (fixed iters vs full pass), so exact
   step-matching would require guessing; we keep each method's own protocol.
   Override with `EP0/EPOCHS` env vars if budget matching is preferred.
2. **Model differs by design** (SOE: num_obs=1, chunk 20, DDIM-20 inference,
   batch 64, lr 3e-4, readout 64→style 16, kl_weight 1e-3; SCOUT: n_obs=2,
   chunk 8, DDPM-100). The comparison is method-level, not architecture-level.
3. **DP-arm baseline not duplicated**: SCOUT's own DP arm (plain DP + same
   rescue protocol) is method-agnostic and already covers that baseline.

## Files

- `make_core_soe.py` — one-time 200-demo rel→abs(6drot) conversion + per-seed
  20-demo cores (`can_core_soe_s{seed}.hdf5`), with cross-verification against
  SCOUT's `can_core.hdf5` (`--scout-core '...CAN-entropy-s{seed}/can/rollout/can_core.hdf5'`).
- `simulation/run_scout_align.py` (in the SOE repo, additive) — sharded, seeded
  eval + rescue runner; merges `demo.hdf5` in SOE run.py layout; wandb.
- `gen_config_soe.py` — training config from `config_template/can_soe.json`
  (reconstructed: SIME public `can_image.json` + DPExt fields from
  `SOE_LIB_training_notes.md` appendix).
- `soe_round_step.py` — extract rescued successes + accumulate `demo_plus_core`
  (reuses SOE's `extract_useful_data` / `dataset_combine`); `latest_try` helper.
- `wandb_train_log.py` — uploads train log.txt loss curve to the round's run.
- `vis_validate_soe.py` — render-integrity gate (SCOUT thresholds).
- `round_soe.sh` — one SOE-internal round (train → eval/rescue → combine),
  idempotent, DRY_RUN, render gate with reduced-shard retry.
- `chain_soe.sh` — rounds 0..5 chain for one seed/GPU.
- `smoke_soe.sh` — ~15 min end-to-end mini validation.

## Launch (per seed, per GPU)

```bash
# one-time data build (per seed set; also verifies vs SCOUT cores)
DATASETS=/root/workspace/baojiachun/soe_data/datasets/can \
  /root/workspace/baojiachun/.venv_soe/bin/python SOE_scripts_2/make_core_soe.py \
  --src /root/workspace/baojiachun/scout/data/robomimic/can/ph/image_v141.hdf5 \
  --out-dir $DATASETS --seeds 233 2333 23333 \
  --scout-core '/root/workspace/baojiachun/scout-entropy/data/2026_8_21_entropy/CAN-entropy-s{seed}/can/rollout/can_core.hdf5'

# smoke (~15 min)
GPU=7 TSEED=233 bash SOE_scripts_2/smoke_soe.sh

# full chain (s=233 on GPU7)
tmux new -s soe233_chain
GPU=7 TSEED=233 DATA_ROOT=/root/workspace/baojiachun/soe_data/2026_8_26_soe/SOE-s233 \
  WPROJ=CAN-8-24-SOE-s233 bash SOE_scripts_2/chain_soe.sh can \
  2>&1 | tee -a /root/workspace/baojiachun/soe_data/2026_8_26_soe/logs/soe233_chain.console.log
```

## Directory layout after round k

```
$DATA_ROOT/can/
  round.log                          # chronicle (START/rc/TOTAL lines)
  train/round_k/
    config_round_k_seed_T.json       # generated training config
    train.log                        # train_single_gpu console
    logs/round_k_seed_T/<ts>/        # ckpt/policy_last.ckpt, log.txt, config.json
  rollout/round_k/
    rollout.stdout  metrics.json  validate.log
    shard_*_phase*.hdf5              # per-worker raw episodes
    demo.hdf5  failed_inits.h5  env_meta_used.json
    demo_plus_core.hdf5              # accumulated train dataset (mask train_success)
```

## Square (2026-08-29)

- Template `config_template/square_soe.json` = `can_soe.json` with SIME public
  `square_image.json` normalize bounds (min `[-0.226,-0.049,0.779]` / max
  `[0.296,0.313,1.101]`); every other field byte-identical (DPExt recipe is
  part of the method). `config_template/` is gitignored -- deploy the file
  manually like `can_soe.json`.
- Square chains need three env overrides (can defaults stay unchanged):
  `DATASETS=/root/workspace/baojiachun/soe_data/datasets/square HORIZON=500
  VISGATE=0`. HORIZON=500 matches SCOUT's square eval horizon; VISGATE=0
  because vis_validate_soe's can-calibrated tstd>20 noise line false-kills
  healthy square frames (healthy square tstd 17.6-27.4, proven 08-26).
- wandb `SQUARE-29-SOE-s{seed}`, chain data `soe_data/2026_8_29_soe/SOE-s{seed}`.
- One-time data build:
  `make_core_soe.py --task square --src .../scout/data/robomimic/square/ph/image_v141.hdf5 --out-dir .../soe_data/datasets/square --seeds 233 2333 23333`
