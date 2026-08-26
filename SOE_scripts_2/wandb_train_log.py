"""Upload SOE train_single_gpu log.txt loss curve to the round's wandb run.

Parses "<log_root>/<try>/log.txt" lines:
    Epoch 12
    Train loss: 0.123456 loss: 0.123456 pred_loss: ... ext_loss: ... kl_loss: ... recon_loss: ...
and logs, per epoch: {"DP/epoch": e, "DP/loss": l, "SOE/pred_loss": ...,
"SOE/ext_loss": ..., "SOE/kl_loss": ..., "SOE/recon_loss": ...}.

Metric names match SCOUT's DP/* convention (DP/epoch + DP/loss are the two
SCOUT logs for its DP retrain); the extra SOE/* breakdown is additive.

Run-id protocol: --run-id + resume=must resumes the run created by
run_scout_align's eval (same round). Without --run-id a fresh run is
created with --run-name (used for the round0 BASE run). The x-axis metric
DP/epoch is pre-registered by run_scout_align; when this script creates
the run itself it registers it here.

wandb failures are non-fatal (exit 0, warning only) -- the authoritative
loss record stays in log.txt.
"""
import argparse
import os
import re
import sys

EPOCH_RE = re.compile(r"^Epoch (\d+)\s*$")
LOSS_RE = re.compile(r"Train loss:\s*([0-9.eE+-]+)")
KV_RE = re.compile(r"(\w+):\s*([0-9.eE+-]+)")


def parse_log(path):
    rows = []
    epoch = None
    with open(path, "r") as f:
        for line in f:
            m = EPOCH_RE.match(line)
            if m:
                epoch = int(m.group(1))
                continue
            if "Train loss:" in line:
                kv = dict((k, float(v)) for k, v in KV_RE.findall(line))
                if epoch is None:
                    continue
                rows.append({
                    "DP/epoch": epoch + 1,
                    "DP/loss": kv.get("loss"),
                    "SOE/pred_loss": kv.get("pred_loss"),
                    "SOE/ext_loss": kv.get("ext_loss"),
                    "SOE/kl_loss": kv.get("kl_loss"),
                    "SOE/recon_loss": kv.get("recon_loss"),
                })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log-txt", required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--run-id", default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--config-json", default=None,
                   help="training config json to attach when creating a run")
    a = p.parse_args()

    rows = parse_log(a.log_txt)
    print("parsed %d epochs from %s" % (len(rows), a.log_txt))
    if not rows:
        return 0

    try:
        import wandb
        run = None
        if a.run_id:
            try:
                run = wandb.init(project=a.project, id=a.run_id,
                                 resume="must")
            except Exception as e:
                print("resume %s failed (%s); creating fresh run" % (a.run_id, e))
                run = None
        if run is None:
            cfg = {}
            if a.config_json and os.path.exists(a.config_json):
                import json
                with open(a.config_json) as f:
                    cfg = json.load(f)
            run = wandb.init(project=a.project, name=a.run_name, config=cfg)
            wandb.define_metric("DP/epoch", hidden=True)
            for m in ("DP/loss", "SOE/pred_loss", "SOE/ext_loss",
                      "SOE/kl_loss", "SOE/recon_loss"):
                wandb.define_metric(m, step_metric="DP/epoch")
        for r in rows:
            run.log({k: v for k, v in r.items() if v is not None})
        run.finish()
        print("uploaded to %s (id=%s)" % (a.project, run.id))
    except Exception as e:
        print("WARNING: wandb upload failed (non-fatal): %s" % e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
