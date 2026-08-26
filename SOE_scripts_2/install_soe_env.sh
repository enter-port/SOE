#!/bin/bash
# install_soe_env.sh -- build /root/workspace/baojiachun/.venv_soe with uv.
# Mirrors the SCOUT .venv recipe (torch 2.4.1+cu121 for H20 sm_90 -- SOE's
# pinned torch 1.13 has no sm_90 kernels) + SOE's own dep pins
# (numpy==1.24.4, easydict, click, ...). open3d skipped (realworld-only).
set -e
cd /root/workspace/baojiachun
export UV_CACHE_DIR=/root/workspace/baojiachun/.uv-cache
MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple
VENV_DIR=/root/workspace/baojiachun/.venv_soe
PY=$VENV_DIR/bin/python
P3D_WHL=https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt240/pytorch3d-0.7.8-cp310-cp310-linux_x86_64.whl

echo "[venv $(date)]"
[ -x "$PY" ] || uv venv "$VENV_DIR" --python /usr/bin/python3.10

echo "[torch $(date)]"
uv pip install --python "$PY" --index-url $MIRROR \
  torch==2.4.1 torchvision==0.19.1

echo "[coredeps $(date)]"
uv pip install --python "$PY" --index-url $MIRROR \
  numpy==1.24.4 easydict==1.13 einops==0.7.0 tqdm matplotlib==3.7.5 \
  opencv-python==4.9.0.80 h5py imageio imageio-ffmpeg pillow scipy click \
  diffusers==0.27.2 huggingface-hub==0.24.6 wandb "zarr<3" "mujoco<3"

echo "[pytorch3d wheel $(date)]"
curl -sfI "$P3D_WHL" >/dev/null && \
  uv pip install --python "$PY" "$P3D_WHL" || echo "P3D_WHL_UNREACHABLE"

echo "[robosuite --no-deps $(date)]"
uv pip install --python "$PY" --index-url $MIRROR --no-deps robosuite==1.4.1

echo "[robomimic --no-deps -e $(date)]"
uv pip install --python "$PY" --no-deps -e /root/workspace/baojiachun/dependencies/robomimic

echo "[egl_probe stub $(date)]"
SITE=$($PY -c "import site; print(site.getsitepackages()[0])")
if [ ! -f "$SITE/egl_probe.py" ]; then
  printf 'def get_available_devices():\n    return [0]\n' > "$SITE/egl_probe.py"
fi
ls -la "$SITE/egl_probe.py"

echo "[verify $(date)]"
"$PY" - <<'PYEOF'
import torch, numpy, h5py, easydict, einops, scipy, click, diffusers, wandb
import robomimic, robosuite, pytorch3d
print("torch", torch.__version__, "cuda_ok", torch.cuda.is_available())
print("numpy", numpy.__version__, "| robomimic", robomimic.__version__,
      "| robosuite", robosuite.__version__, "| pytorch3d", pytorch3d.__version__)
x = torch.randn(8, 8, device="cuda")
print("gpu matmul ok:", float((x @ x).sum()))
print("VERIFY_OK")
PYEOF
echo "[done $(date)]"
