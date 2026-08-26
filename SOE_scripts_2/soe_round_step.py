"""Post-eval dataset steps for the SCOUT-aligned SOE campaign.

Sub-commands (both reuse SOE's own simulation utilities unchanged):

  extract_and_combine --demo <dir>/demo.hdf5 --core <core_or_prev_plus.hdf5>
      --used-demo core_20|train_success [--n-rollouts 100]
        -> writes mask/success on demo.hdf5 (exploration-segment successes),
           then demo_plus_core.hdf5 with mask/train_success =
           this round's rescued successes U previous train_success
           (accumulated, SOE's own multi-round semantics = SCOUT's
           success_accum).

  latest_try --log-root <dir>
        -> prints the newest timestamp subdir that holds ckpt/policy_last.ckpt
           (or newest subdir at all with --any).
"""
import argparse
import os
import sys

from easydict import EasyDict

SIM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "simulation")
if not os.path.isdir(SIM_DIR):  # deployed flat (baojiachun/SOE_scripts_2)
    SIM_DIR = os.path.join(os.environ.get("SOE_REPO",
                                          "/root/workspace/baojiachun/SOE"),
                           "simulation")
sys.path.insert(0, SIM_DIR)

from extract_useful_data import extract_useful_data_v2  # noqa: E402
from dataset_combine import combine_dataset             # noqa: E402


def extract_and_combine(a):
    demo = a.demo
    plus = os.path.join(os.path.dirname(demo), "demo_plus_core.hdf5")
    extract_useful_data_v2(demo, "success", start_index=a.n_rollouts)
    cfg = EasyDict({
        "output": plus,
        "input": [
            {"path": demo,
             "map": [[None, "all"], ["success", "train_success"],
                     [None, "from_eval"]]},
            {"path": a.core,
             "map": [[a.used_demo, "all"], [a.used_demo, "train_success"],
                     [a.used_demo, "from_prev"]]},
        ],
    })
    combine_dataset(cfg)
    import h5py
    with h5py.File(plus, "r") as f:
        n_ts = len(f["mask/train_success"])
        n_all = len(f["mask/all"])
    print("demo_plus_core: %s train_success=%d all=%d" % (plus, n_ts, n_all))


def latest_try(a):
    root = a.log_root
    tries = sorted(t for t in os.listdir(root)
                   if os.path.isdir(os.path.join(root, t)))
    if not tries:
        sys.exit("no try dirs under %s" % root)
    if a.any:
        print(tries[-1])
        return
    for t in reversed(tries):
        if os.path.exists(os.path.join(root, t, "ckpt", "policy_last.ckpt")):
            print(t)
            return
    sys.exit("no try with ckpt/policy_last.ckpt under %s" % root)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("extract_and_combine")
    c.add_argument("--demo", required=True)
    c.add_argument("--core", required=True,
                   help="round0: seed core hdf5 (mask core_20); rounds>=1: "
                        "previous demo_plus_core.hdf5 (mask train_success)")
    c.add_argument("--used-demo", default="core_20")
    c.add_argument("--n-rollouts", type=int, default=100)
    c.set_defaults(fn=extract_and_combine)
    t = sub.add_parser("latest_try")
    t.add_argument("--log-root", required=True)
    t.add_argument("--any", action="store_true")
    t.set_defaults(fn=latest_try)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
