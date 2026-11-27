"""
Monocular depth models behind one common interface, without ROS.

    model = load_model("moge", num_tokens=600)
    depth = model(rgb, fov_x)     # rgb (H, W, 3) uint8, fov_x [deg] or None
                                  # -> torch tensor (H, W) [m] on model.dev, invalid = inf

Used by the ROS node (depth_completion_node.py), the viewer (viewer.py) and the
offline benchmark (bench/), so all of them call a model the same way.

Models (runtime and accuracy: bench/README.md):
  moge        MoGe-2 vitl            package moge (default of the ROS node, 600 tokens, 29 Hz on an A6000)
  moge2-s     MoGe-2 vits-normal     package moge
  moge2-b     MoGe-2 vitb-normal     package moge
  moge2-l-normal  MoGe-2 vitl-normal package moge
  moge3-l     MoGe-3 vitl            package moge; refinement only with CUDA (flex_gemm/Triton)
  depthpro2-s|b|l|h  Apple Depth Pro 2   package depth_pro_2, weights for research use only
"""
import math
import sys
import types

import numpy as np
import torch

MOGE = {"moge": "Ruicheng/moge-2-vitl",
        "moge2-s": "Ruicheng/moge-2-vits-normal",
        "moge2-b": "Ruicheng/moge-2-vitb-normal",
        "moge2-l-normal": "Ruicheng/moge-2-vitl-normal",
        "moge3-l": "Ruicheng/moge-3-vitl",
        "moge3-g": "Ruicheng/moge-3-vitg"}
DEPTHPRO = ("depthpro2-s", "depthpro2-b", "depthpro2-l", "depthpro2-h")
MODELS = tuple(MOGE) + DEPTHPRO
MAX_DEPTH = 50.0   # [m] anything beyond (sky) counts as invalid


def default_device():
    return "cuda" if torch.cuda.is_available() else \
        "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(name, num_tokens=1200, device=None, stream=True, window=4, refine=None):
    """num_tokens applies to MoGe only, refine to MoGe-3 only (refinement steps, None = 3 with
    CUDA, else 0), stream/window to Depth Pro 2 only (default: streaming mode, window 4)."""
    if name in MOGE:
        return MoGe(MOGE[name], num_tokens, device, refine)
    if name in DEPTHPRO:
        return DepthPro2(name, device, stream, window)
    raise ValueError(f"unknown model '{name}', available: {', '.join(MODELS)}")


def _invalid_to_inf(d):
    bad = ~torch.isfinite(d) | (d <= 0) | (d > MAX_DEPTH)
    return torch.where(bad, torch.full_like(d, float("inf")), d)


class MoGe:
    """MoGe-2 or MoGe-3. The field of view (fov_x) is passed in, so the depth is metric."""

    def __init__(self, repo, num_tokens=1200, device=None, refine=None):
        self.dev = torch.device(device or default_device())
        self.num_tokens = num_tokens
        self.kw = {}
        load_kw = {}
        if "moge-3" in repo:
            if refine is None:
                refine = 3 if self.dev.type == "cuda" else 0
            self.kw["refine_steps"] = refine
            if refine == 0:            # the refinement needs flex_gemm (CUDA/Triton)
                try:
                    import flex_gemm  # noqa: F401
                except ImportError:
                    _stub_flex_gemm()
                load_kw = {"model_kwargs": {"refiner": None}}
            from moge.model.v3 import MoGeModel
        else:
            from moge.model.v2 import MoGeModel
        self.model = MoGeModel.from_pretrained(repo, **load_kw).to(self.dev).eval()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    @torch.inference_mode()
    def __call__(self, rgb, fov_x):
        img = torch.from_numpy(rgb).to(self.dev).permute(2, 0, 1).float() / 255.0
        # use_fp16 uses CUDA autocast internally and fails on MPS
        out = self.model.infer(img, fov_x=fov_x, num_tokens=self.num_tokens,
                               use_fp16=self.dev.type == "cuda", **self.kw)
        d = out["depth"].float()
        if "mask" in out:
            d = torch.where(out["mask"].bool(), d, torch.full_like(d, float("inf")))
        return _invalid_to_inf(d)


def _stub_flex_gemm():
    """Placeholder so that moge.model.v3 can be imported without flex_gemm (refinement off)."""
    if "flex_gemm" in sys.modules:
        return
    root, nn_, ops = (types.ModuleType(n) for n in ("flex_gemm", "flex_gemm.nn", "flex_gemm.ops"))
    for name in ("SubmanifoldConv3d", "SparsePool3d", "SparseUpsample3d"):
        setattr(nn_, name, torch.nn.Module)
    ops.NeighborCache = object
    root.nn, root.ops = nn_, ops
    sys.modules.update({"flex_gemm": root, "flex_gemm.nn": nn_, "flex_gemm.ops": ops})


class DepthPro2:
    """Apple Depth Pro 2. stream=True (default): live stream (unbuffered), attends to the
    previous window - 1 frames (default window 4) and is temporally more stable, at a higher cost
    per frame that
    grows with the window; the focal length is fixed at the first frame,
    so one instance serves one camera with a fixed image size. stream=False: every image on its
    own (deterministic, used by the offline benchmark)."""

    def __init__(self, name="depthpro2-s", device=None, stream=True, window=4):
        from depth_pro_2 import create_model_and_transforms
        self.model, self.transform = create_model_and_transforms(
            name, device=torch.device(device or default_device()))
        self.model.eval()
        self.dev = next(self.model.parameters()).device
        self.use_stream, self.window, self.stream = stream, window, None

    @torch.inference_mode()
    def __call__(self, rgb, fov_x):
        f_px = None if fov_x is None else 0.5 * rgb.shape[1] / math.tan(math.radians(fov_x) / 2)
        x = self.transform(np.ascontiguousarray(rgb))
        if self.use_stream:
            if self.stream is None:
                self.stream = self.model.stream(window=self.window, f_px=f_px, mode="unbuffered")
            d = self.stream.step(x)[-1]["depth"]
        else:
            d = self.model.infer(x, f_px=f_px)["depth"]
        return _invalid_to_inf(d.float())
