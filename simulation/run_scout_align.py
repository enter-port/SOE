"""SCOUT-aligned sharded eval + rescue runner for SOE (DPExt).

Protocol mirrors the SCOUT entropy campaign (XMODE=soe / rescue mode) so that
SOE numbers are directly comparable to the SCOUT DP / SCOUT arms:

- Phase 1 (eval): N fresh scenes. Scene i is seeded with ``base_seed + i``
  (``np.random.seed(base_seed + i)`` immediately before the env reset inside
  :func:`rollout_utils.rollout`) -- exactly the mechanism SCOUT's
  ``collect_initial_states`` uses (``RobomimicImageWrapper`` seeds the numpy
  global RNG, robosuite draws its init-state randomness from it), so both
  pipelines evaluate on the *same* 100 initial states. Exploration OFF.
- Phase 2 (rescue): every scene failed by phase 1 is retried ``try_times``
  times from its saved initial state (no early stop -- SOE's own run.py
  semantics), with the DPExt exploration extension enabled.

Sharding: episodes are distributed over ``--n-shards`` worker subprocesses
(one env each, all on the GPU given by CUDA_VISIBLE_DEVICES). Per-episode
seeding makes the scenes independent of the shard count.

Outputs (layout identical to SOE ``run.py`` so the downstream
``extract_useful_data_v2`` / ``dataset_combine`` steps work unchanged):
  <out>/demo.hdf5                demo_i = scene i; demo_{N + i*T + j} = retry j
  <out>/metrics.json             SCOUT-compatible metric fields
  <out>/shard_*_phase*.hdf5      per-worker raw episodes (intermediates)

Env creation replicates ``rollout_utils.dp_load`` with two overrides:
``horizon`` is taken from the CLI (SCOUT can = 300, NOT robomimic's 400
default) and ``env_kwargs.render_gpu_device_id`` is pinned to
$SOE_RENDER_GPU when set (EGL renders on the assigned GPU; the dataset
env_args hardcode device 0).
"""

import os
import sys
import json
import time
import argparse
import subprocess
import traceback

import numpy as np
import h5py

SIM_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SIM_DIR)
sys.path.insert(0, os.path.join(SIM_DIR, "..", "src"))


def _ns(**kw):
    """argparse.Namespace with run.py's defaults, patched by kw."""
    import argparse as _ap
    defaults = dict(
        critic_agent=None, eta=None, num_inference_steps=None,
        inference_horizon=None, high_noise_eval=False,
        enable_exploration=False, enable_exploration_debug=False,
        tau1=None, tau2=None, noise_scale=None,
        disable_styles=False, enable_action_noise=False,
        action_noise_scale=None, return_intermediate=False,
    )
    defaults.update(kw)
    return _ap.Namespace(**defaults)


def _load_policy_and_env(cfg, agent_path, explore, noise_scale, horizon):
    """dp_load replica with render-device + horizon overrides."""
    import torch
    import robomimic.utils.obs_utils as ObsUtils
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils
    from robomimic.environments.env_base import EnvBase  # noqa: F401 (rollout asserts)
    from rollout_utils import RolloutDP
    from policy.dp_ext import DPExt

    obs_spec = dict()
    use_image_obs = False
    for k, v in cfg.policy.params.obs_shape_meta.items():
        if v.type not in obs_spec:
            if v.type == "rgb":
                use_image_obs = True
            obs_spec[v.type] = []
        obs_spec[v.type].append(k)
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": obs_spec, "goal": dict()})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = DPExt(**cfg.policy.params).to(device)
    policy.load_state_dict(torch.load(agent_path, map_location=device), strict=False)

    args = _ns(enable_exploration=explore, noise_scale=noise_scale)
    policy = RolloutDP(policy, args, cfg, enable_exploration_as_args=True)

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=cfg.dataset.params.path)
    # abs actions -> absolute controller (same as dp_load)
    env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False
    rg = os.environ.get("SOE_RENDER_GPU")
    if rg is not None:
        env_meta["env_kwargs"]["render_gpu_device_id"] = int(rg)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        env_name=env_meta["env_name"],
        render=False,
        render_offscreen=True,  # image obs -> offscreen EGL context
        use_image_obs=use_image_obs,
        use_depth_obs=False,
    )
    return policy, env


def _write_episode(fout, idx, traj, with_init):
    g = fout.create_group("ep_%d" % idx)
    eg = g.create_group("traj")
    eg.create_dataset("actions", data=np.array(traj["actions"]))
    eg.create_dataset("states", data=np.array(traj["states"]))
    eg.create_dataset("rewards", data=np.array(traj["rewards"]))
    eg.create_dataset("dones", data=np.array(traj["dones"]))
    for k in traj["obs"]:
        eg.create_dataset("obs/%s" % k, data=np.array(traj["obs"][k]))
        eg.create_dataset("next_obs/%s" % k, data=np.array(traj["next_obs"][k]))
    eg.attrs["num_samples"] = traj["actions"].shape[0]
    isd = traj["initial_state_dict"]
    if "model" in isd:
        eg.attrs["model_file"] = isd["model"]
    if with_init:
        ig = g.create_group("init")
        ig.create_dataset("states", data=np.asarray(isd["states"]))
        if "model" in isd:
            ig.attrs["model_file"] = isd["model"]


def worker_main(a):
    import torch
    from easydict import EasyDict
    from rollout_utils import rollout
    from rotation_transformer import RotationTransformer

    with open(a.config, "r") as f:
        cfg = EasyDict(json.load(f))

    # worker-level torch seeding (policy stochasticity); reproducible, shard-dependent.
    torch.manual_seed(a.torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.torch_seed)

    episodes = [int(x) for x in a.episodes.split(",") if x != ""]
    explore = a.phase == 2
    policy, env = _load_policy_and_env(
        cfg, a.agent, explore=explore,
        noise_scale=a.noise_scale, horizon=a.horizon)

    rot = RotationTransformer("axis_angle", "rotation_6d") if a.abs_action else None
    if a.init_file:
        finit = h5py.File(a.init_file, "r")
    else:
        finit = None

    out_path = a.out_file
    done_marker = out_path + ".done"
    if os.path.exists(done_marker):
        print("[worker] %s already done, skipping" % out_path)
        return 0

    with h5py.File(out_path, "w") as fout:
        for idx in episodes:
            t0 = time.time()
            if a.phase == 1:
                scene = idx
                # SCOUT scene identity: np.random.seed(base_seed + i) right
                # before the fresh env.reset() inside rollout().
                np.random.seed(a.seed_base + scene)
                init_state = None
            else:
                scene = (idx - a.n_scenes) // a.try_times
                assert finit is not None
                ig = finit["scene_%d" % scene]
                init_state = {"states": ig["states"][()]}
                if "model_file" in ig.attrs:
                    init_state["model"] = ig.attrs["model_file"]
            stats, traj = rollout(
                policy=policy, env=env, horizon=a.horizon,
                return_obs=True, camera_names=None,
                initial_state_dict=init_state,
                traj_renderer=None,
                abs_action=a.abs_action, rotation_transformer=rot,
            )
            _write_episode(fout, idx, traj, with_init=(a.phase == 1))
            fout.flush()
            print("[worker p%d] ep %d scene %d success=%s len=%d %.1fs" % (
                a.phase, idx, scene, float(stats["Success_Rate"]),
                traj["actions"].shape[0], time.time() - t0), flush=True)
            if a.phase == 1 and a.dump_env_meta:
                emp = os.path.join(os.path.dirname(out_path), "env_meta_used.json")
                if not os.path.exists(emp):
                    try:
                        with open(emp, "w") as f:
                            json.dump(env.serialize(), f, indent=4)
                    except Exception as e:
                        print("[worker] env.serialize failed: %s" % e)
    with open(done_marker, "w") as f:
        f.write("ok\n")
    return 0


def _episodes_in(path):
    if not os.path.exists(path):
        return set()
    with h5py.File(path, "r") as f:
        return {int(k.split("_")[1]) for k in f.keys() if k.startswith("ep_")}


def _run_phase(a, phase, episode_lists, extra_env, init_file=None, tag=""):
    """Spawn one worker per episode list; return dict of rc per shard.

    ``tag`` separates retry passes ("" or "_retry") so a recovery worker
    never truncates a previous shard file (h5py "w" mode)."""
    n_shards = len(episode_lists)
    procs = []
    for k, eps in enumerate(episode_lists):
        if not eps:
            continue
        out_file = os.path.join(a.out_dir, "shard_%d_phase%d%s.hdf5" % (k, phase, tag))
        cmd = [sys.executable, os.path.abspath(__file__), "worker",
               "--phase", str(phase), "--config", a.config, "--agent", a.agent,
               "--out-file", out_file,
               "--episodes", ",".join(str(e) for e in eps),
               "--horizon", str(a.horizon), "--noise-scale", str(a.noise_scale),
               "--seed-base", str(a.seed_base), "--n-scenes", str(a.n_scenes),
               "--try-times", str(a.try_times),
               "--torch-seed", str(a.seed_base * 1000 + phase * 100 + k),
               "--abs-action" if a.abs_action else "--no-abs-action"]
        if init_file:
            cmd += ["--init-file", init_file]
        if phase == 1:
            cmd += ["--dump-env-meta"]
        env = dict(os.environ)
        env.update(extra_env)
        logf = open(os.path.join(a.out_dir, "shard_%d_phase%d%s.log" % (k, phase, tag)), "w")
        p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
        procs.append((k, p, logf))
    rcs = {}
    for k, p, logf in procs:
        rcs[k] = p.wait()
        logf.close()
    return rcs


def _shard_files(a, phase):
    import glob
    return sorted(glob.glob(os.path.join(
        a.out_dir, "shard_*_phase%d*.hdf5" % phase)))


def _retry_missing(a, phase, expected, extra_env, init_file=None):
    """One recovery pass at reduced concurrency for missing episodes.

    Worker rc's are irrelevant here -- episode presence is re-scanned from
    the shard files afterwards."""
    have = set()
    for p in _shard_files(a, phase):
        have |= _episodes_in(p)
    missing = sorted(set(expected) - have)
    if not missing:
        return True
    print("[orchestrate] phase %d missing %d episodes, recovery pass: %s" % (
        phase, len(missing), missing[:20]), flush=True)
    half = max(1, len(missing) // 2 + 1)
    lists = [missing[:half], missing[half:]]
    _run_phase(a, phase, lists, extra_env, init_file=init_file, tag="_retry")
    have = set()
    for p in _shard_files(a, phase):
        have |= _episodes_in(p)
    return set(missing) <= have


def merge_demo(a, expected, demo_path):
    """Assemble demo.hdf5 in SOE run.py layout from the shard files."""
    shard_files = {}
    for phase in (1, 2):
        for p in _shard_files(a, phase):
            for e in _episodes_in(p):
                shard_files[e] = p
    missing = [e for e in expected if e not in shard_files]
    assert not missing, "cannot merge, missing episodes: %s" % missing[:10]

    env_args = None
    emp = os.path.join(a.out_dir, "env_meta_used.json")
    if os.path.exists(emp):
        with open(emp) as f:
            env_args = f.read()

    total = 0
    with h5py.File(demo_path, "w") as fout:
        data = fout.create_group("data")
        for e in sorted(expected):
            with h5py.File(shard_files[e], "r") as fin:
                src = fin["ep_%d/traj" % e]
                data.copy(src, data, name="demo_%d" % e)
                total += src["actions"].shape[0]
        data.attrs["total"] = total
        if env_args is not None:
            data.attrs["env_args"] = env_args
    return total


def orchestrate(a):
    t_start = time.time()
    os.makedirs(a.out_dir, exist_ok=True)

    wandb_run, rid = None, None
    if a.wandb_project:
        import wandb
        wandb_run = wandb.init(
            project=a.wandb_project, name=a.wandb_run_name,
            config={"task": a.task, "seed_base": a.seed_base,
                    "n_scenes": a.n_scenes, "try_times": a.try_times,
                    "noise_scale": a.noise_scale, "horizon": a.horizon,
                    "agent": a.agent, "mode": "soe_rescue"},
        )
        rid = wandb_run.id
        # pre-register x-axes: the train-log uploader resumes this run and
        # logs DP/* series (resumed runs cannot define_metric themselves)
        wandb.define_metric("DP/epoch", hidden=True)
        for m in ("DP/loss", "SOE/pred_loss", "SOE/ext_loss",
                  "SOE/kl_loss", "SOE/recon_loss"):
            wandb.define_metric(m, step_metric="DP/epoch")

    extra_env = {"MUJOCO_GL": "egl"}
    # ---- phase 1: eval on the N seeded scenes ------------------------- #
    scenes = list(range(a.n_scenes))
    shard_lists = [scenes[i::a.n_shards] for i in range(a.n_shards)]
    print("[orchestrate] phase 1: %d scenes over %d shards" % (
        a.n_scenes, a.n_shards), flush=True)
    t0 = time.time()
    _run_phase(a, 1, shard_lists, extra_env)
    if not _retry_missing(a, 1, scenes, extra_env):
        raise RuntimeError("phase 1 incomplete after recovery pass")
    print("[orchestrate] phase 1 done in %.1fs" % (time.time() - t0), flush=True)

    success = {}
    init_src = {}
    for p in _shard_files(a, 1):
        with h5py.File(p, "r") as f:
            for e in _episodes_in(p):
                g = f["ep_%d/traj" % e]
                success[e] = bool(np.any(g["dones"][()]))
                if e not in init_src:
                    init_src[e] = p
    baseline_solved = sum(success.values())
    sr = baseline_solved / float(a.n_scenes)
    failed = [i for i in scenes if not success[i]]
    print("[orchestrate] SR=%.3f (%d/%d), failed scenes=%d" % (
        sr, baseline_solved, a.n_scenes, len(failed)), flush=True)
    if wandb_run:
        wandb_run.log({"eval/success_rate": sr})

    # ---- phase 2: rescue x try_times from saved initial states -------- #
    init_file = os.path.join(a.out_dir, "failed_inits.h5")
    with h5py.File(init_file, "w") as f:
        for i in failed:
            with h5py.File(init_src[i], "r") as fs:
                f.copy(fs["ep_%d/init" % i], f, name="scene_%d" % i)

    retry_idx = [a.n_scenes + i * a.try_times + j
                 for i in failed for j in range(a.try_times)]
    t0 = time.time()
    if retry_idx:
        shard_lists = [retry_idx[i::a.n_shards] for i in range(a.n_shards)]
        _run_phase(a, 2, shard_lists, extra_env, init_file=init_file)
        if not _retry_missing(a, 2, retry_idx, extra_env, init_file=init_file):
            raise RuntimeError("phase 2 incomplete after recovery pass")
    print("[orchestrate] phase 2 done in %.1fs (%d retries)" % (
        time.time() - t0, len(retry_idx)), flush=True)

    # ---- merge + metrics ---------------------------------------------- #
    expected = scenes + retry_idx
    demo_path = os.path.join(a.out_dir, "demo.hdf5")
    total_steps = merge_demo(a, expected, demo_path)
    print("[orchestrate] merged demo.hdf5: %d episodes, %d steps" % (
        len(expected), total_steps), flush=True)

    rescued_scenes = 0
    rescued_trajs = 0
    retry_success = {}
    for p in _shard_files(a, 2):
        with h5py.File(p, "r") as f:
            for e in _episodes_in(p):
                g = f["ep_%d/traj" % e]
                retry_success[e] = bool(np.any(g["dones"][()]))
    for i in failed:
        s = any(retry_success.get(a.n_scenes + i * a.try_times + j, False)
                for j in range(a.try_times))
        rescued_scenes += int(s)
        rescued_trajs += sum(
            retry_success.get(a.n_scenes + i * a.try_times + j, False)
            for j in range(a.try_times))
    pass_at_k = (baseline_solved + rescued_scenes) / float(a.n_scenes)

    if wandb_run:
        wandb_run.log({
            "explore/pass@10": pass_at_k,
            "explore/rescued_scenes": rescued_scenes,
            "explore/rescued_trajs": rescued_trajs,
            "explore/n_failed_scenes": len(failed),
        })
        wandb_run.finish()

    metrics = {
        "task": a.task, "mode": "soe_rescue", "exp_num": a.exp_num,
        "dp_ckpt": a.agent, "config": a.config,
        "n_init_states": a.n_scenes, "try_times": a.try_times,
        "explore_try_times": a.try_times, "seed": a.seed_base,
        "noise_scale": a.noise_scale, "horizon": a.horizon,
        "success_rate": sr, "baseline_solved": baseline_solved,
        "n_failed": len(failed), "failed_init_indices": failed,
        "exploration_rescued": rescued_scenes,
        "rescued_trajs": rescued_trajs,
        "pass_at_5": pass_at_k,   # SCOUT json-compat alias (value is pass@try_times)
        "pass_at_10": pass_at_k,
        "n_all_trajs": len(expected),
        "n_success_trajs": baseline_solved + rescued_trajs,
        "total_steps": total_steps,
        "wandb_run_id": rid,
        "wandb_run_name": a.wandb_run_name,
        "outputs": {"demo": demo_path},
        "elapsed_s": time.time() - t_start,
    }
    with open(a.metrics_json or os.path.join(a.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    print("[orchestrate] FINAL SR=%.3f pass@%d=%.3f rescued=%d/%d scenes "
          "(%d trajs) elapsed=%.1fs" % (
              sr, a.try_times, pass_at_k, rescued_scenes, len(failed),
              rescued_trajs, time.time() - t_start), flush=True)
    return 0


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    o = sub.add_parser("orchestrate")
    o.add_argument("--agent", required=True)
    o.add_argument("--config", required=True)
    o.add_argument("--out-dir", required=True)
    o.add_argument("--task", default="can")
    o.add_argument("--exp-num", default=0)
    o.add_argument("--n-scenes", type=int, default=100)
    o.add_argument("--seed-base", type=int, default=42)
    o.add_argument("--try-times", type=int, default=10)
    o.add_argument("--n-shards", type=int, default=8)
    o.add_argument("--horizon", type=int, default=300)
    o.add_argument("--noise-scale", type=float, default=2.0)
    o.add_argument("--metrics-json", default=None)
    o.add_argument("--wandb-project", default=None)
    o.add_argument("--wandb-run-name", default=None)
    o.set_defaults(fn=orchestrate)

    w = sub.add_parser("worker")
    w.add_argument("--phase", type=int, required=True, choices=[1, 2])
    w.add_argument("--config", required=True)
    w.add_argument("--agent", required=True)
    w.add_argument("--out-file", required=True)
    w.add_argument("--episodes", required=True)
    w.add_argument("--horizon", type=int, default=300)
    w.add_argument("--noise-scale", type=float, default=2.0)
    w.add_argument("--seed-base", type=int, default=42)
    w.add_argument("--n-scenes", type=int, default=100)
    w.add_argument("--try-times", type=int, default=10)
    w.add_argument("--torch-seed", type=int, default=0)
    w.add_argument("--init-file", default=None)
    w.add_argument("--dump-env-meta", action="store_true")
    w.add_argument("--abs-action", dest="abs_action", action="store_true",
                   default=True)
    w.add_argument("--no-abs-action", dest="abs_action", action="store_false")
    w.set_defaults(fn=worker_main)

    args = p.parse_args()
    try:
        rc = args.fn(args)
    except Exception:
        traceback.print_exc()
        rc = 1
    sys.exit(rc)


if __name__ == "__main__":
    main()
