"""Verify config 'eta' reaches DDIMScheduler.step end-to-end (SQUARE-8-30 prep).

Run on the server: .venv_soe/bin/python SOE_scripts_2/check_eta_flow.py
Proves the passthrough chain policy.params -> DPExt **kwargs ->
DiffusionUNetPolicy.self.kwargs -> scheduler.step(eta=...) by spying on step.
"""
import sys, inspect
import torch
sys.path.insert(0, "/root/workspace/baojiachun/SOE/src")
import diffusers
from diffusers import DDIMScheduler
from policy.diffusion import DiffusionUNetPolicy

print("diffusers", diffusers.__version__)
assert "eta" in inspect.signature(DDIMScheduler.step).parameters, "step() has no eta param"

torch.manual_seed(0)
p = DiffusionUNetPolicy(action_dim=10, horizon=4, n_obs_steps=1, obs_feature_dim=8,
                        num_inference_steps=5, eta=1.0)
assert p.kwargs.get("eta") == 1.0, p.kwargs

captured = []
orig = p.noise_scheduler.step
def spy(model_output, t, sample, **kw):
    captured.append(kw.get("eta", "MISSING"))
    return orig(model_output, t, sample, **kw)
p.noise_scheduler.step = spy

out = p.predict_action(torch.randn(1, 1, 8))
assert out.shape == (1, 4, 10), out.shape
assert captured and all(e == 1.0 for e in captured), captured
print("ETA FLOW OK: step called %dx, eta=%s" % (len(captured), captured[0]))
