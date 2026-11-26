"""
Depth completion without ROS: sensor depth with holes + prediction of a monocular model
(depth_models.py) -> completed depth. Entry point is fill(). See README.md.

  1. Global scale: median(measurement / prediction), smoothed over time (ema).
  2. Gradient fusion: correction field c with log D = log M + c that matches the measurements,
     is smooth along surfaces (the model provides the shape), does not act across edges and
     decays to 0 far behind the measured range. Preconditioned CG on the GPU.
  3. Edge filters against floating points. The model draws depth discontinuities as a soft
     ramp; exactly there the sensor often measures nothing, so the ramp would end up as points
     between foreground and background in the map. Filled pixels are discarded if they
       a) lie on the ramp itself: too steep for a real surface (ramp_mask), or
       b) lie in the zone around a measured depth edge and match neither the near nor the far
          side (supported).
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


@torch.inference_mode()
def fill(depth_m, pred, fusion, prev_log_s=None, ema=0.3, f_px=None,
         ramp_dmin=0.5, support_px=12, support_rtol=0.02, support_edge=0.1):
    """Fills the holes of depth_m. Returns (completed [m], log_s); 0 = empty.

    depth_m: sensor depth [m], 0 = hole. pred: model depth [m], inf = invalid. Both torch
    tensors (H, W) on the same device. prev_log_s: global scale of the previous frame (log).
    f_px: focal length [px] for the ramp filter (without f_px it is off).

    Edge filters (find the edge -> zone around the edge -> check there):
      ramp_dmin   [m]  step 1, ramp_mask: discard the model's smeared edge. 0 = off.
                       Must stay below the camera height (floor: distance = camera height
                       ≈ 0.58 m).
      support_px  [px] step 2, supported: radius in which real measurements are looked up
                       (= width of the zone around depth edges). 0 = off.
      support_edge     a zone exists where the measurements in the radius spread by > 10 %.
      support_rtol     allowed deviation from the near or far side (2 %).
    """
    valid = depth_m > 0
    pvalid = torch.isfinite(pred) & (pred > 0)

    # 1. global scale (robust median, every 4th pixel is enough)
    sub = (valid & pvalid)[::4, ::4]
    log_s = prev_log_s if prev_log_s is not None else 0.0
    if int(sub.sum()) > 200:
        ratio = depth_m[::4, ::4][sub] / pred[::4, ::4][sub]
        cur = float(torch.log(torch.quantile(ratio, 0.5) if ratio.numel() <= 16_000_000
                              else ratio.median()))
        log_s = cur if prev_log_s is None else (1 - ema) * prev_log_s + ema * cur
    est = torch.where(pvalid, pred * math.exp(log_s), torch.zeros_like(pred))

    # 2. local correction
    est = fusion(depth_m, est)

    # 3. edge filters: only filled pixels are discarded, measurements stay
    keep = est > 0
    if ramp_dmin > 0 and f_px:
        keep &= ~ramp_mask(est, f_px, ramp_dmin)
    if support_px > 0:
        keep &= supported(est, depth_m, support_px, support_rtol, support_edge)
    filled = torch.where(valid, depth_m, torch.where(keep, est, torch.zeros_like(est)))
    return filled, log_s


def ramp_mask(est, f_px, dmin, s=2):
    """Step 1: pixels on the model's smeared edge ("too steep to be real").

    The model draws a jump from 1.5 m to 3 m as a ramp 1.6 -> 1.9 -> 2.4 -> 2.8 m. A real surface
    can only look that steep in the image if one looks almost exactly along it. Formally: for a
    plane at distance d from the camera centre, f·|∇(1/z)| ≤ 1/d. Extended, a ramp passes almost
    through the camera (d ≈ 0), real surfaces do not: the floor has d = camera height at any
    range. Pixels with f·|∇(1/z)| > 1/dmin are discarded, so the mask automatically adapts to
    the width of the ramp. Gradient as central difference over ±s pixels; pairs with an invalid
    neighbour do not count.
    """
    v = est > 0
    q = torch.where(v, 1.0 / torch.where(v, est, torch.ones_like(est)), torch.zeros_like(est))
    g2 = torch.zeros_like(q)
    gu = (q[:, 2 * s:] - q[:, : -2 * s]) / (2 * s)
    gv = (q[2 * s:, :] - q[: -2 * s, :]) / (2 * s)
    g2[:, s:-s] += torch.where(v[:, 2 * s:] & v[:, : -2 * s], gu * gu, torch.zeros_like(gu))
    g2[s:-s, :] += torch.where(v[2 * s:, :] & v[: -2 * s, :], gv * gv, torch.zeros_like(gv))
    return (f_px * f_px) * g2 > 1.0 / (dmin * dmin)


def supported(est, depth_m, r, rtol, edge):
    """Step 2: does the filled value match the real measurements within radius r?

    The smallest (dmin) and largest (dmax) measurement in the (2r+1)² window are determined.
      - no measurement in the window (inside large holes): kept.
      - similar measurements (dmax ≤ dmin·(1+edge)), smooth surface: est must lie in
        [dmin·(1-rtol), dmax·(1+rtol)].
      - strongly spread measurements: the pixel is in the zone around a depth edge (the edge
        "dilated" by r px). est must match the near side (dmin) OR the far side (dmax) within
        ±rtol; values in between would be points in mid-air.
    Returns the mask of pixels that may stay.
    """
    v = depth_m > 0
    k = 2 * r + 1

    def mx(a):                                   # separable k×k max pooling
        a = F.max_pool2d(a[None, None], (1, k), 1, (0, r))
        return F.max_pool2d(a, (k, 1), 1, (r, 0))[0, 0]

    dmax = mx(torch.where(v, depth_m, torch.zeros_like(depth_m)))
    dmin = -mx(-torch.where(v, depth_m, torch.full_like(depth_m, 1e9)))
    inside = (est >= dmin * (1 - rtol)) & (est <= dmax * (1 + rtol))
    near_side = (torch.abs(est - dmin) <= rtol * dmin) | (torch.abs(est - dmax) <= rtol * dmax)
    is_edge = dmax > dmin * (1 + edge)
    return (dmax <= 0) | (inside & (~is_edge | near_side))
