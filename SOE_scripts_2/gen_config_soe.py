"""Generate a SOE training config from the can_soe.json template.

Replicates run_full_multi_round.generate_training_cfg (seed, dataset path,
train_filter_key, log_dir injection) plus optional overrides used by
round_soe.sh (num_epochs / save_epochs / num_workers).
"""
import argparse
import json
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--template", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--dataset", required=True, help="hdf5 path (abs)")
    p.add_argument("--filter-key", required=True, help="core_20 | train_success")
    p.add_argument("--log-dir", required=True, help="training log root (abs)")
    p.add_argument("--out", required=True)
    p.add_argument("--num-epochs", type=int, default=None)
    p.add_argument("--save-epochs", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    a = p.parse_args()

    with open(a.template, "r") as f:
        cfg = json.load(f)

    cfg["seed"] = a.seed
    cfg["dataset"]["params"]["path"] = os.path.abspath(a.dataset)
    cfg["dataset"]["params"]["train_filter_key"] = a.filter_key
    cfg["log_dir"] = os.path.abspath(a.log_dir)
    cfg.setdefault("resume_ckpt", None)
    cfg.setdefault("resume_epoch", -1)
    if a.num_epochs is not None:
        cfg["num_epochs"] = a.num_epochs
    if a.save_epochs is not None:
        cfg["save_epochs"] = a.save_epochs
    if a.num_workers is not None:
        cfg["num_workers"] = a.num_workers

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(cfg, f, indent=4)

    # report mask size like the upstream driver does
    import h5py
    with h5py.File(cfg["dataset"]["params"]["path"], "r") as f:
        key = "mask/{}".format(a.filter_key)
        n = len(f[key]) if key in f else -1
    print("wrote {} (dataset {}, filter {} -> {} demos)".format(
        a.out, os.path.basename(a.dataset), a.filter_key, n))


if __name__ == "__main__":
    main()
