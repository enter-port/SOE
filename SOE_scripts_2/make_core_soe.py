"""Build the SOE-format datasets for the SCOUT-aligned campaign.

1. Convert the official 200-demo can image dataset (relative 7-dim actions)
   to SOE's absolute 10-dim (pos3|rot6d|grip1) action format via SOE's own
   ``convert_rel_actions_to_abs`` -> ``<out>/image_v141_abs_6drot.hdf5``
   (one-time, shared across seeds).
2. Per training seed: extract the SAME 20 demos SCOUT's split_core.py chose
   (``sorted(np.random.default_rng(seed).choice(200, 20, replace=False))``,
   verified to reproduce SCOUT's recorded indices, e.g. seed 233 ->
   [2,14,17,20,28,34,41,42,73,82,98,106,111,132,140,145,173,181,186,190])
   -> ``<out>/can_core_soe_s<seed>.hdf5`` with ``mask/core_20`` = all 20.

Optionally cross-verify against SCOUT's own per-seed core files
(--scout-core pattern like .../CAN-entropy-s{seed}/can/rollout/can_core.hdf5):
demo count, per-demo lengths and total steps must match exactly.
"""
import argparse
import os
import sys

import numpy as np
import h5py

SIM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "simulation")


def build_abs(src, out, workers):
    sys.path.insert(0, SIM_DIR)
    from robomimic_dataset_conversion import convert_rel_actions_to_abs
    convert_rel_actions_to_abs(src, out, None, num_workers=workers)


def extract_core(abs_path, out, indices):
    with h5py.File(abs_path, "r") as fin, h5py.File(out, "w") as fout:
        for k in fin:
            # skip data (rebuilt below) and any existing mask groups -- the
            # source's masks index 200 demos and are invalid in the 20-demo
            # core; mask/core_20 is written fresh instead.
            if k in ("data", "mask"):
                continue
            fin.copy(k, fout)
        din, dout = fin["data"], fout.create_group("data")
        for k in din.attrs:
            dout.attrs[k] = din.attrs[k]
        total = 0
        for new_id, src_id in enumerate(indices):
            din.copy("demo_%d" % src_id, dout, name="demo_%d" % new_id)
            total += din["demo_%d/actions" % src_id].shape[0]
        dout.attrs["total"] = total
    # write mask/core_20 with robomimic's own writer -- robomimic_v2 reads
    # mask/<key> directly as an iterable of b"demo_i" keys, so the format
    # must match robomimic's create_hdf5_filter_key byte-for-byte
    from robomimic.utils.file_utils import create_hdf5_filter_key
    lengths = create_hdf5_filter_key(
        hdf5_path=out,
        demo_keys=["demo_%d" % i for i in range(len(indices))],
        key_name="core_20")
    print("wrote %s: %d demos, %d steps, indices=%s" % (out, len(indices), total, indices))
    return total


def verify(out_path, scout_core_path):
    with h5py.File(out_path, "r") as f1, h5py.File(scout_core_path, "r") as f2:
        d1, d2 = f1["data"], f2["data"]
        n1 = sorted(int(k.split("_")[1]) for k in d1.keys())
        n2 = sorted(int(k.split("_")[1]) for k in d2.keys())
        ok = n1 == n2
        lens1 = [d1["demo_%d/actions" % i].shape[0] for i in n1]
        lens2 = []
        for i in n2:
            key = "demo_%d/actions" % i
            if key not in d2:
                key = "demo_%d/abs_actions" % i  # SCOUT core keeps both keys
            lens2.append(d2[key].shape[0])
        ok = ok and lens1 == lens2 and int(d1.attrs["total"]) == int(d2.attrs["total"])
        print("verify vs %s: %s (demos %d/%d, lens %s vs %s, total %s/%s)" % (
            scout_core_path, "PASS" if ok else "FAIL", len(n1), len(n2),
            lens1, lens2, d1.attrs["total"], d2.attrs["total"]))
        return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="official image_v141.hdf5 (200 demos)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--task", default="can")
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--convert-workers", type=int, default=8)
    p.add_argument("--skip-convert", action="store_true")
    p.add_argument("--scout-core", default=None,
                   help="pattern with {seed} pointing at SCOUT's can_core.hdf5 "
                        "for cross-verification, e.g. '.../CAN-entropy-s{seed}/can/rollout/can_core.hdf5'")
    a = p.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    abs_path = os.path.join(a.out_dir, "image_v141_abs_6drot.hdf5")
    if os.path.exists(abs_path):
        print("abs dataset exists:", abs_path)
    elif a.skip_convert:
        print("skip convert, but missing:", abs_path)
        sys.exit(1)
    else:
        print("converting", a.src, "->", abs_path, flush=True)
        build_abs(a.src, abs_path, a.convert_workers)

    all_ok = True
    for seed in a.seeds:
        indices = sorted(int(i) for i in
                         np.random.default_rng(seed).choice(200, a.n, replace=False))
        out = os.path.join(a.out_dir, "%s_core_soe_s%d.hdf5" % (a.task, seed))
        if os.path.exists(out):
            print("core exists:", out)
        else:
            extract_core(abs_path, out, indices)
        if a.scout_core:
            all_ok &= verify(out, a.scout_core.format(seed=seed))
    if a.scout_core and not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
