"""Image-integrity validation for SOE demo.hdf5 (run_scout_align output).

Adapted from SCOUT soe_scripts/vis_validate.py v3 thresholds (calibrated
2026-08-20/21): per-camera temporal-std; agentview OK in [1, 20]
(noise mode ~27-32, frozen ~0, healthy 3.4-14.4); both cameras get the
frozen (<1) gate. SOE demo.hdf5 stores episodes as data/demo_<i> with
obs/<cam> -- same robomimic layout, so only the min-demo-id default differs
(0: validate the eval segment too; pass 100 to check retries only).

Usage:  python vis_validate_soe.py <demo.hdf5> [min-demo-id]
Exit 0 = healthy, 1 = corrupted.
"""
import sys

import h5py
import numpy as np

CAMS = ("agentview_image", "robot0_eye_in_hand_image")


def demo_flags(g) -> list:
    msgs = []
    for cam in CAMS:
        if "obs/" + cam not in g:
            continue
        img = np.asarray(g["obs/" + cam][()], dtype=np.float32)
        tstd = float(img.std(axis=0).mean())
        if tstd < 1.0:
            msgs.append(f"{cam} frozen (tstd={tstd:.2f})")
        if cam == "agentview_image" and tstd > 20.0:
            msgs.append(f"{cam} noise (tstd={tstd:.2f})")
    return msgs


def main() -> int:
    path = sys.argv[1]
    min_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    bad = []
    with h5py.File(path, "r") as f:
        demos = sorted((k for k in f["data"].keys()
                        if str(k).startswith("demo")
                        and int(str(k).split("_")[1]) >= min_id),
                       key=lambda s: int(str(s).split("_")[1]))
        for name in demos:
            msgs = demo_flags(f["data"][name])
            if msgs:
                bad.append(name)
                print(f"  {name:<10s} {'; '.join(msgs)}")
    print(f"{path}: {'CORRUPT' if bad else 'HEALTHY'} "
          f"({len(bad)} bad / {len(demos)} demos)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
