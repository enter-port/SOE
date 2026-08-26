"""Upload SOE train_single_gpu log.txt loss curve to the round's wandb run.

Parses "<log_root>/<try>/log.txt" lines:
    Epoch 12
    Train loss: 0.123456 loss: 0.123456 pred_loss: ... ext_loss: ... kl_loss: ... recon_loss: ...
and logs, per epoch: {"DP/epoch": e, "DP/loss": l, "SOE/pred_loss": ...,
"SOE/ext_loss": ..., "SOE/kl_loss": ..., "SOE/recon_loss": ...}.

Metric names match SCOUT's DP/* convention (DP/epoch + DP/loss are the two
SCOUT logs for its DP retrain); the extra SOE/* breakdown is additive.

Modes:
  default  one-shot post-hoc upload of the whole curve (round_soe.sh keeps
           calling this after training as a safety net)
  --tail   live: poll the log every --poll seconds and stream new epochs;
           wait for the log to appear (--auto), exit once the training is
           done (ckpt/policy_last.ckpt exists AND no new rows).

Idempotency: a state file <log.txt>.wandb_state.json records
{run_id, last_epoch}; both modes upload only epochs > last_epoch and the
tail mode reuses the recorded run_id, so tail + post-hoc on the same log
never duplicate history and never fork a second run for the same curve.

Run-id protocol: --run-id + resume=must resumes the run created by
run_scout_align's eval (same round). Without --run-id a fresh run is
created with --run-name (used for the round0 BASE run). wandb failures are
non-fatal (warning only) -- the authoritative loss record stays in log.txt.
"""
import argparse
import glob
import json
import os
import re
import sys
import time

EPOCH_RE = re.compile(r"^Epoch (\d+)\s*$")
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


def state_path(log_txt):
    return log_txt + ".wandb_state.json"


def load_state(log_txt):
    try:
        with open(state_path(log_txt)) as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(log_txt, run_id, last_epoch):
    with open(state_path(log_txt), "w") as f:
        json.dump({"run_id": run_id, "last_epoch": last_epoch}, f)


def resolve_log_path(a):
    """--auto: --log-txt is the log ROOT; use the newest */log.txt beneath."""
    if not a.auto:
        return a.log_txt
    deadline = time.time() + 60 * a.tail_max_min
    while time.time() < deadline:
        cands = sorted(glob.glob(os.path.join(a.log_txt, "*", "log.txt")))
        if cands:
            return cands[-1]
        time.sleep(15)
    raise FileNotFoundError("no */log.txt under %s" % a.log_txt)


def upload_new(run, rows, last_epoch):
    new = [r for r in rows if r["DP/epoch"] > last_epoch]
    for r in new:
        run.log({k: v for k, v in r.items() if v is not None})
    new_last = new[-1]["DP/epoch"] if new else last_epoch
    return new_last, len(new)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log-txt", required=True,
                   help="path to log.txt (with --auto: the training log ROOT)")
    p.add_argument("--project", required=True)
    p.add_argument("--run-id", default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--config-json", default=None)
    p.add_argument("--auto", action="store_true",
                   help="treat --log-txt as log root, wait for newest */log.txt")
    p.add_argument("--tail", action="store_true", help="live streaming mode")
    p.add_argument("--poll", type=float, default=60.0)
    p.add_argument("--tail-max-min", type=float, default=600.0,
                   help="give up waiting for the log / stop tailing after N min")
    a = p.parse_args()

    log_txt = resolve_log_path(a)
    print("log:", log_txt, flush=True)

    try:
        import wandb
        st = load_state(log_txt)
        run = None
        resume_id = st.get("run_id") or a.run_id
        if resume_id:
            try:
                run = wandb.init(project=a.project, id=resume_id, resume="must")
            except Exception as e:
                print("resume %s failed (%s); creating fresh run" % (resume_id, e))
                run = None
        if run is None:
            cfg = {}
            if a.config_json and os.path.exists(a.config_json):
                with open(a.config_json) as f:
                    cfg = json.load(f)
            run = wandb.init(project=a.project, name=a.run_name, config=cfg)
            wandb.define_metric("DP/epoch", hidden=True)
            for m in ("DP/loss", "SOE/pred_loss", "SOE/ext_loss",
                      "SOE/kl_loss", "SOE/recon_loss"):
                wandb.define_metric(m, step_metric="DP/epoch")
        save_state(log_txt, run.id, st.get("last_epoch", 0))

        if not a.tail:
            rows = parse_log(log_txt)
            last, n = upload_new(run, rows, st.get("last_epoch", 0))
            save_state(log_txt, run.id, last)
            print("uploaded %d new epochs (total %d parsed)" % (n, len(rows)))
            run.finish()
            print("uploaded to %s (id=%s)" % (a.project, run.id))
            return 0

        # tail mode: stream until training is done
        ckpt = os.path.join(os.path.dirname(log_txt), "ckpt", "policy_last.ckpt")
        deadline = time.time() + 60 * a.tail_max_min
        last = st.get("last_epoch", 0)
        idle = 0
        while time.time() < deadline:
            try:
                rows = parse_log(log_txt)
            except FileNotFoundError:
                rows = []
            last, n = upload_new(run, rows, last)
            save_state(log_txt, run.id, last)
            if n == 0 and os.path.exists(ckpt):
                idle += 1
                if idle >= 2:
                    break
            else:
                idle = 0
            time.sleep(a.poll)
        run.finish()
        print("tail done at epoch %s (id=%s)" % (last, run.id))
    except Exception as e:
        print("WARNING: wandb upload failed (non-fatal): %s" % e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
