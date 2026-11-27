"""
Depth completion without ROS: sensor depth with holes + prediction of a monocular model
(depth_models.py) -> completed depth. Entry point is fill(). See README.md.

  1. Global scale: median(measurement / prediction), smoothed over time (ema).
  2. Gradient fusion: correction field c with log D = log M + c that matches the measurements,
     is smooth along surfaces (the model provides the shape), does not act across edges and
     decays to 0 far behind the measured range. Preconditioned CG on the GPU.
  3. Edge filter against floating points. The model draws depth discontinuities as a soft
     ramp; exactly there the sensor often measures nothing, so the ramp would end up as points
     between foreground and background in the map. Depth edges are found in the completed
     model depth (a jump in 1/z), dilated, and filled pixels there are discarded.
     Measured pixels are never changed. A hole is harmless, a point in mid-air is not.

Used by the ROS node (depth_completion_node.py), the viewer (viewer.py) and the
offline benchmark (bench/), so all of them always run the same code.
"""
import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------- gradient fusion (GPU)

def _block_stats(z, m, f):
    """Masked mean, variance and coverage per f×f block (avg_pool = cv2 INTER_AREA)."""
    H, W = z.shape
    h, w = H // f, W // f
    m = m[: h * f, : w * f].float()
    zm = z[: h * f, : w * f] * m
    pool = lambda a: F.avg_pool2d(a[None, None], f, f)[0, 0]
    frac = pool(m)
    fi = 1.0 / torch.clamp(frac, min=1e-6)
    mean = pool(zm) * fi
    var = torch.clamp(pool(zm * zm) * fi - mean ** 2, min=0)
    mean = torch.where(frac == 0, torch.zeros_like(mean), mean)
    return mean, var, frac


class GradientFusion:
    """Local correction of the model depth M towards the measurement Z, in log depth:
    log D = log M + c.

    The correction field c has to satisfy three things at once (least squares):
      1. Data:       c ≈ r = log Z - log M where a measurement exists (weight wm).
      2. Smoothness: c barely changes between neighbours (weight lam), except across edges of
                     the model depth, where the coupling exp(-|Δ log M| / sigma_edge) becomes
                     weak. The shape comes from the model, only scale and offset from the sensor.
      3. Decay:      c is pulled slightly towards 0 (mu_near), more strongly behind the measured
                     range (mu_far, transition over far_band in log depth). Without nearby
                     measurements the model depth with the global scale remains.
    Minimised is  Σ wm·(c - r)² + Σ w_neighbour·(c_i - c_j)² + Σ mu·c²,  a linear system with a
    sparse symmetric matrix. It is solved on a grid downsampled by `down` with preconditioned CG
    (iters iterations, warm start from the previous frame), then a second time with Huber weights
    against outliers (outlier, in log units). Finally c is brought back to full resolution in an
    edge-preserving way (_guided_upsample).

    Measurement weight wm per grid cell: coverage with measurements, sensor accuracy ~ 1/z²
    (falling off beyond z_ref), less weight for spread inside the cell (edges), outliers w.r.t.
    the median. Inputs and output are torch tensors (float32, same device).
    """

    def __init__(self, down=2, lam=1.0, mu_near=1e-4, mu_far=0.005, far_band=0.15,
                 sigma_edge=0.05, z_ref=4.0, outlier=1.0, iters=40, tol=1e-4):
        self.down, self.lam = down, lam
        self.mu_near, self.mu_far, self.far_band = mu_near, mu_far, far_band
        self.sigma_edge, self.z_ref, self.outlier = sigma_edge, z_ref, outlier
        self.iters, self.tol = iters, tol
        self._c_prev = None      # warm start with the correction field of the previous frame
        self._up_cache = None

    @staticmethod
    def _apply(c, diag, wx, wy):
        """Matrix-vector product of the system: (wm + mu + degree)·c - Σ neighbour weight·c_neighbour."""
        out = diag * c
        out[:, :-1] -= wx * c[:, 1:]
        out[:, 1:] -= wx * c[:, :-1]
        out[:-1, :] -= wy * c[1:, :]
        out[1:, :] -= wy * c[:-1, :]
        return out

    def _cg(self, b, diag, wx, wy, x):
        """Conjugate gradients with Jacobi (diagonal) preconditioning, starting at x."""
        r = b - self._apply(x, diag, wx, wy)
        z = r / diag
        p = z.clone()
        rz = (r * z).sum()
        bn = torch.sqrt((b * b).sum()) + 1e-12
        for i in range(self.iters):
            Ap = self._apply(p, diag, wx, wy)
            alpha = rz / torch.clamp((p * Ap).sum(), min=1e-30)
            x = x + alpha * p
            r = r - alpha * Ap
            # convergence test only every 10 iterations: each test forces a GPU synchronisation
            if i % 10 == 9 and bool(torch.sqrt((r * r).sum()) < self.tol * bn):
                break
            z = r / diag
            rz_new = (r * z).sum()
            p = z + (rz_new / torch.clamp(rz, min=1e-30)) * p
            rz = rz_new
        return x

    def __call__(self, depth_m, est):
        f = self.down
        H, W = depth_m.shape
        pv = est > 0
        both = (depth_m > 0) & pv
        zero = torch.zeros_like(est)
        logM = torch.where(pv, torch.log(torch.where(pv, est, torch.ones_like(est))), zero)
        logZ = torch.where(both, torch.log(torch.where(both, depth_m, torch.ones_like(est))), zero)
        r_full = torch.where(both, logZ - logM, zero)

        gM, gMvar, gMfrac = _block_stats(logM, pv, f)
        r, rvar, frac = _block_stats(r_full, both, f)
        zlog, _, _ = _block_stats(logZ, both, f)

        # measurement weights: coverage, sensor accuracy ~1/z², edges/spread, outliers
        wm = frac * torch.clamp(torch.exp(2 * (math.log(self.z_ref) - zlog)), max=1.0)
        wm = wm * torch.exp(-(torch.sqrt(gMvar) + torch.sqrt(rvar)) / self.sigma_edge)
        wm = torch.where(frac < 0.2, torch.zeros_like(wm), wm)
        has = wm > 0
        if bool(has.any()):
            rh, zh = r[has], zlog[has]
            med = torch.quantile(rh, 0.5) if rh.numel() <= 16_000_000 else rh.median()
            wm = wm * (1.0 / (1.0 + ((r - med) / self.outlier) ** 2))
            t = torch.clamp((gM - torch.quantile(zh, 0.98)) / self.far_band, 0, 1)
        else:
            t = torch.ones_like(gM)
        mu = self.mu_near + (self.mu_far - self.mu_near) * t   # decay behind the measured range

        # edge-aware smoothness
        ok = gMfrac > 0.5
        wx = self.lam * torch.exp(-torch.abs(torch.diff(gM, dim=1)) / self.sigma_edge) \
            * (ok[:, 1:] & ok[:, :-1])
        wy = self.lam * torch.exp(-torch.abs(torch.diff(gM, dim=0)) / self.sigma_edge) \
            * (ok[1:, :] & ok[:-1, :])
        deg = torch.zeros_like(gM)
        deg[:, :-1] += wx
        deg[:, 1:] += wx
        deg[:-1, :] += wy
        deg[1:, :] += wy

        x = self._c_prev if (self._c_prev is not None and self._c_prev.shape == gM.shape) \
            else torch.zeros_like(gM)
        x = self._cg(wm * r, wm + mu + deg, wx, wy, x)
        wm2 = wm * torch.clamp(self.outlier / torch.clamp(torch.abs(r - x), min=1e-9), max=1.0)  # Huber
        x = self._cg(wm2 * r, wm2 + mu + deg, wx, wy, x)
        self._c_prev = x

        # edge-preserving, directly to full resolution: an extra bilinear step
        # (half -> full resolution) would smear c across object edges.
        c_full = self._guided_upsample(x, gM, logM, H, W)
        return torch.where(pv, est * torch.exp(c_full), zero)

    def _guided_upsample(self, c, gM, logM, H, W):
        """Bilinear upsampling where grid points with similar model depth count more (edge-preserving)."""
        h, w = c.shape
        key = (h, w, H, W, str(c.device))
        if self._up_cache is None or self._up_cache[0] != key:
            ys = (torch.arange(H, dtype=torch.float32, device=c.device) + 0.5) * h / H - 0.5
            xs = (torch.arange(W, dtype=torch.float32, device=c.device) + 0.5) * w / W - 0.5
            y0 = torch.clamp(torch.floor(ys).long(), 0, h - 1)
            x0 = torch.clamp(torch.floor(xs).long(), 0, w - 1)
            y1, x1 = torch.clamp(y0 + 1, 0, h - 1), torch.clamp(x0 + 1, 0, w - 1)
            fy = torch.clamp(ys - torch.floor(ys), 0, 1)[:, None]
            fx = torch.clamp(xs - torch.floor(xs), 0, 1)[None, :]
            tabs = [(yy[:, None] * w + xx[None, :], wy * wx)
                    for yy, wy in ((y0, 1 - fy), (y1, fy)) for xx, wx in ((x0, 1 - fx), (x1, fx))]
            self._up_cache = (key, tabs)
        cf, gf = c.reshape(-1), gM.reshape(-1)
        acc = torch.zeros((H, W), dtype=torch.float32, device=c.device)
        wsum = torch.full((H, W), 1e-12, dtype=torch.float32, device=c.device)
        for idx, bw in self._up_cache[1]:
            d = torch.abs(logM - gf[idx]) / self.sigma_edge
            wgt = bw / (1.0 + d * d)
            acc += wgt * cf[idx]
            wsum += wgt
        return acc / wsum


def global_scale(depth_m, pred, prev_log_s=None, ema=0.3):
    """Model depth times the median of measurement / model (every 4th pixel), smoothed over time.
    Returns (scaled prediction with 0 = invalid, log of the scale)."""
    pvalid = torch.isfinite(pred) & (pred > 0)
    sub = ((depth_m > 0) & pvalid)[::4, ::4]
    log_s = prev_log_s if prev_log_s is not None else 0.0
    if int(sub.sum()) > 200:
        ratio = depth_m[::4, ::4][sub] / pred[::4, ::4][sub]
        cur = float(torch.log(torch.quantile(ratio, 0.5) if ratio.numel() <= 16_000_000
                              else ratio.median()))
        log_s = cur if prev_log_s is None else (1 - ema) * prev_log_s + ema * cur
    return torch.where(pvalid, pred * math.exp(log_s), torch.zeros_like(pred)), log_s


def jump_edges(est, edge_jump, edge_grow, w=5):
    """Depth edges of est, dilated by edge_grow px (find the edge -> zone around it).

    An edge is a jump of more than edge_jump [1/m] in inverse depth q = 1/z inside a w×w window.
    Measured in 1/z, a flat surface such as the floor changes evenly at any distance (a
    plane is linear in 1/z) and is not mistaken for an edge, while the soft ramp the model draws
    between an object at 1.5 m and a wall at 3 m is a jump of ~0.33 1/m.
    """
    v = est > 0
    q = torch.where(v, 1.0 / torch.where(v, est, torch.ones_like(est)), torch.zeros_like(est))
    qmax = F.max_pool2d(q[None, None], w, 1, w // 2)[0, 0]
    qmin = -F.max_pool2d(-torch.where(v, q, torch.full_like(q, 1e9))[None, None], w, 1, w // 2)[0, 0]
    edge = v & (qmax - qmin > edge_jump)
    if edge_grow > 0:
        g = 2 * edge_grow + 1
        edge = F.max_pool2d(edge[None, None].float(), g, 1, edge_grow)[0, 0] > 0
    return edge


@torch.inference_mode()
def fill(depth_m, pred, fusion, prev_log_s=None, ema=0.3, edge_jump=0.015, edge_grow=4):
    """Fills the holes of depth_m. Returns (completed [m], log_s); 0 = empty.

    depth_m: sensor depth [m], 0 = hole. pred: model depth [m], inf = invalid. Both torch
    tensors (H, W) on the same device. fusion: a GradientFusion kept over frames (warm start).
    prev_log_s: global scale of the previous frame (log), ema: its smoothing over time.

    Edge filter (jump_edges): edge_jump [1/m] = jump in 1/z that counts as an edge,
    edge_grow [px] = width of the zone around an edge in which nothing is filled. 0 = off.
    """
    valid = depth_m > 0
    est, log_s = global_scale(depth_m, pred, prev_log_s, ema)            # 1. global scale
    est = fusion(depth_m, est)                                           # 2. local correction
    keep = est > 0                                                       # 3. edge filter
    if edge_jump > 0:
        keep &= ~jump_edges(est, edge_jump, edge_grow)
    filled = torch.where(valid, depth_m, torch.where(keep, est, torch.zeros_like(est)))
    return filled, log_s
