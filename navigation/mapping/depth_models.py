"""
Monokulare Tiefenmodelle hinter einer gemeinsamen Schnittstelle, ohne ROS.

    model = load_model("moge2-b", num_tokens=600)
    depth = model(rgb, fov_x)     # rgb (H, W, 3) uint8, fov_x [Grad] oder None
                                  # -> torch-Tensor (H, W) [m] auf model.dev, ungültig = inf

Genutzt vom ROS-Knoten (moge_depth_completion.py), vom Viewer (dev/depth_viewer.py) und vom
Offline-Benchmark (bench/), damit alle dasselbe Modell gleich aufrufen.

Modelle (Laufzeit M4 Pro/MPS, 1280x720, siehe bench/README.md):
  moge        MoGe-2 vitl            Paket moge
  moge2-s     MoGe-2 vits-normal     Paket moge
  moge2-b     MoGe-2 vitb-normal     Paket moge (Standard im ROS-Knoten, 600 Tokens ~9 Hz)
  moge3-l     MoGe-3 vitl            Paket moge; Verfeinerung nur mit CUDA (flex_gemm/Triton)
  depthpro2-s|b|l|h  Apple Depth Pro 2   Paket depth_pro_2, Gewichte nur für Forschung
"""
import math
import sys
import types

import numpy as np
import torch

MOGE = {"moge": "Ruicheng/moge-2-vitl",
        "moge2-s": "Ruicheng/moge-2-vits-normal",
        "moge2-b": "Ruicheng/moge-2-vitb-normal",
        "moge3-l": "Ruicheng/moge-3-vitl"}
DEPTHPRO = ("depthpro2-s", "depthpro2-b", "depthpro2-l", "depthpro2-h")
MODELS = tuple(MOGE) + DEPTHPRO
MAX_DEPTH = 50.0   # [m] darüber (Himmel) gilt als ungültig


def default_device():
    return "cuda" if torch.cuda.is_available() else \
        "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(name, num_tokens=1200, device=None, stream=False):
    """num_tokens gilt nur für MoGe, stream nur für Depth Pro 2."""
    if name in MOGE:
        return MoGe(MOGE[name], num_tokens, device)
    if name in DEPTHPRO:
        return DepthPro2(name, device, stream)
    raise ValueError(f"unbekanntes Modell '{name}', verfügbar: {', '.join(MODELS)}")


def _invalid_to_inf(d):
    bad = ~torch.isfinite(d) | (d <= 0) | (d > MAX_DEPTH)
    return torch.where(bad, torch.full_like(d, float("inf")), d)


class MoGe:
    """MoGe-2 oder MoGe-3. Die Brennweite (fov_x) wird übergeben, die Tiefe ist dann metrisch."""

    def __init__(self, repo, num_tokens=1200, device=None):
        self.dev = torch.device(device or default_device())
        self.num_tokens = num_tokens
        self.kw = {}
        load_kw = {}
        if "moge-3" in repo:
            from_v3 = self.dev.type == "cuda"
            self.kw["refine_steps"] = 3 if from_v3 else 0
            if not from_v3:            # Verfeinerung braucht flex_gemm (CUDA/Triton)
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
        # use_fp16 nutzt intern CUDA-Autocast und bricht auf MPS ab
        out = self.model.infer(img, fov_x=fov_x, num_tokens=self.num_tokens,
                               use_fp16=self.dev.type == "cuda", **self.kw)
        d = out["depth"].float()
        if "mask" in out:
            d = torch.where(out["mask"].bool(), d, torch.full_like(d, float("inf")))
        return _invalid_to_inf(d)


def _stub_flex_gemm():
    """Platzhalter, damit moge.model.v3 ohne flex_gemm importierbar ist (Verfeinerung aus)."""
    if "flex_gemm" in sys.modules:
        return
    root, nn_, ops = (types.ModuleType(n) for n in ("flex_gemm", "flex_gemm.nn", "flex_gemm.ops"))
    for name in ("SubmanifoldConv3d", "SparsePool3d", "SparseUpsample3d"):
        setattr(nn_, name, torch.nn.Module)
    ops.NeighborCache = object
    root.nn, root.ops = nn_, ops
    sys.modules.update({"flex_gemm": root, "flex_gemm.nn": nn_, "flex_gemm.ops": ops})


class DepthPro2:
    """Apple Depth Pro 2. stream=False: jedes Bild einzeln (deterministisch). stream=True: Live-
    Stream (unbuffered), sieht die 15 Bilder davor und ist zeitlich stabiler; die Brennweite wird
    beim ersten Bild festgelegt. Nur für eine Kamera mit fester Bildgröße."""

    def __init__(self, name="depthpro2-s", device=None, stream=False):
        from depth_pro_2 import create_model_and_transforms
        self.model, self.transform = create_model_and_transforms(
            name, device=torch.device(device or default_device()))
        self.model.eval()
        self.dev = next(self.model.parameters()).device
        self.use_stream, self.stream = stream, None

    @torch.inference_mode()
    def __call__(self, rgb, fov_x):
        f_px = None if fov_x is None else 0.5 * rgb.shape[1] / math.tan(math.radians(fov_x) / 2)
        x = self.transform(np.ascontiguousarray(rgb))
        if self.use_stream:
            if self.stream is None:
                self.stream = self.model.stream(f_px=f_px, mode="unbuffered")
            d = self.stream.step(x)[-1]["depth"]
        else:
            d = self.model.infer(x, f_px=f_px)["depth"]
        return _invalid_to_inf(d.float())
