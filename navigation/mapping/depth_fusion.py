"""
Depth completion algorithm, independent of ROS: MoGe-2 wrapper + global scale +
variational log-depth correction (gradient fusion) + edge cleanup.

Used by the ROS node (moge_depth_completion.py) and by the offline benchmark
(bench/depth_bench.py), so both always run the identical code.
"""
import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------- Gradienten-Fusion (GPU)

def _block_stats(z, m, f):
    """Maskierter Mittelwert, Varianz und Abdeckung je f×f-Block (avg_pool = cv2 INTER_AREA)."""
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
    """Eingaben und Ausgabe sind torch-Tensoren (float32, gleiches Gerät)."""

    def __init__(self, down=4, lam=1.0, mu_near=1e-4, mu_far=0.005, far_band=0.15,
                 sigma_edge=0.02, z_ref=4.0, outlier=1.0, iters=80, tol=1e-4):
        self.down, self.lam = down, lam
        self.mu_near, self.mu_far, self.far_band = mu_near, mu_far, far_band
        self.sigma_edge, self.z_ref, self.outlier = sigma_edge, z_ref, outlier
        self.iters, self.tol = iters, tol
        self._c_prev = None      # Warmstart mit dem Korrekturfeld des letzten Frames
        self._up_cache = None

    @staticmethod
    def _apply(c, diag, wx, wy):
        out = diag * c
        out[:, :-1] -= wx * c[:, 1:]
        out[:, 1:] -= wx * c[:, :-1]
        out[:-1, :] -= wy * c[1:, :]
        out[1:, :] -= wy * c[:-1, :]
        return out

    def _cg(self, b, diag, wx, wy, x):
        r = b - self._apply(x, diag, wx, wy)
        z = r / diag
        p = z.clone()
        rz = (r * z).sum()
        bn = torch.sqrt((b * b).sum()) + 1e-12
        for _ in range(self.iters):
            Ap = self._apply(p, diag, wx, wy)
            alpha = rz / torch.clamp((p * Ap).sum(), min=1e-30)
            x = x + alpha * p
            r = r - alpha * Ap
            if bool(torch.sqrt((r * r).sum()) < self.tol * bn):
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

        # Messgewichte: Abdeckung, Sensorgenauigkeit ~1/z², Kanten/Streuung, Ausreißer
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
        mu = self.mu_near + (self.mu_far - self.mu_near) * t   # Abklingen hinter Messbereich

        # kantenbewusste Glattheit
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

        # Kantentreu direkt auf volle Auflösung: ein zusätzlicher bilinearer Schritt
        # (halbe -> volle Auflösung) würde c über Objektkanten verschmieren.
        c_full = self._guided_upsample(x, gM, logM, H, W)
        return torch.where(pv, est * torch.exp(c_full), zero)

    def _guided_upsample(self, c, gM, logM, H, W):
        """Bilinear, Gitterpunkte mit ähnlicher MoGe-Tiefe zählen mehr (kantentreu)."""
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
def fill(depth_m, pred, fusion, prev_log_s=None, ema=0.3, edge_rtol=0.04):
    """Gibt (gefüllt [m], log_s) zurück. depth_m: 0 = Loch, pred: inf = ungültig (Tensoren).

    edge_rtol: gefüllte Pixel auf Tiefenkanten (Spanne im 3x3 > edge_rtol) verwerfen, 0 = aus.
    """
    valid = depth_m > 0
    pvalid = torch.isfinite(pred) & (pred > 0)
    sub = (valid & pvalid)[::4, ::4]
    log_s = prev_log_s if prev_log_s is not None else 0.0
    if int(sub.sum()) > 200:
        ratio = depth_m[::4, ::4][sub] / pred[::4, ::4][sub]
        cur = float(torch.log(torch.quantile(ratio, 0.5) if ratio.numel() <= 16_000_000
                              else ratio.median()))
        log_s = cur if prev_log_s is None else (1 - ema) * prev_log_s + ema * cur
    est = torch.where(pvalid, pred * math.exp(log_s), torch.zeros_like(pred))
    est = fusion(depth_m, est)
    filled = torch.where(valid, depth_m, est)
    if edge_rtol > 0:
        # MoGe zeichnet Objektkanten als weiche Rampe; genau dort misst der ToF oft nichts,
        # sodass die Rampe als "Kometenschweif" zwischen Vorder- und Hintergrund landet.
        # Gefüllte Pixel auf solchen Kanten werden verworfen, Messungen bleiben unverändert.
        on_edge = depth_edge(filled, edge_rtol)
        filled = torch.where(on_edge & ~valid, torch.zeros_like(filled), filled)
    return filled, log_s


def depth_edge(d, rtol, k=3):
    """Pixel, deren Tiefenspanne (max - min) im k×k-Fenster > rtol·Tiefe ist (Löcher ignoriert).

    Wie utils3d.depth_edge aus dem MoGe-Repo. Bei rtol=0.04 wird eine Rampe von 1 m auf 2 m bis
    ~16 px Breite fast ganz, bis ~30 px zu zwei Dritteln entfernt; flach gesehener Boden (Kamera
    0.58 m hoch) bleibt bis ~7 m Entfernung erhalten.
    """
    v = d > 0
    inf = torch.full_like(d, float("inf"))
    dmax = F.max_pool2d(torch.where(v, d, torch.zeros_like(d))[None, None], k, 1, k // 2)[0, 0]
    dmin = -F.max_pool2d(-torch.where(v, d, inf)[None, None], k, 1, k // 2)[0, 0]
    return v & ((dmax - dmin) > rtol * d)


# ---------------------------------------------------------------- MoGe

class MoGe:
    def __init__(self, model="Ruicheng/moge-2-vitl", num_tokens=1200, device=None):
        from moge.model.v2 import MoGeModel
        if device is None:
            device = "cuda" if torch.cuda.is_available() else \
                "mps" if torch.backends.mps.is_available() else "cpu"
        self.dev = torch.device(device)
        self.num_tokens = num_tokens
        self.model = MoGeModel.from_pretrained(model).to(self.dev).eval()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    @torch.inference_mode()
    def __call__(self, rgb, fov_x):
        img = torch.from_numpy(rgb).to(self.dev).permute(2, 0, 1).float() / 255.0
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.dev.type == "cuda"):
            out = self.model.infer(img, fov_x=fov_x, num_tokens=self.num_tokens)
        d = out["depth"].float()
        if "mask" in out:
            d = torch.where(out["mask"].bool(), d, torch.full_like(d, float("inf")))
        return d          # torch-Tensor auf dem MoGe-Gerät
