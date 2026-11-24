#!/usr/bin/env python3
"""
Eingefrorene Füll-Methoden für die Paper-Auswertung (Stand: Branch main, Commit 5146946).

Warum eine Kopie: Der Refactor (Branch refactor) entfernt moge_depth.py (Variante A) und die
CPU-Fusion. Für den Vergleich im Paper müssen aber alle Varianten mit genau den getunten
Parametern verfügbar bleiben, unabhängig vom Stand der Pipeline. Die Funktionen unten sind 1:1
aus main übernommen (nur die Importe sind zusammengefasst):

  global / A : local_affine, align_and_fill      <- navigation/mapping/moge_depth.py
  B          : GradientFusion (CPU), GradientFusionTorch + fill_torch (GPU), fill
  MoGe       : MoGe-2 ViT-L, FP16, 1200 Tokens    <- navigation/mapping/moge_depth_optim.py

Die eingesetzte Variante B auf refactor (moge_depth_completion.py) rechnet dieselbe Mathematik;
bag_fill.py --b-impl deployed nutzt stattdessen direkt diese.
Keine ROS-Abhängigkeit.
"""
import math

import cv2
import numpy as np
import torch

MODEL = "Ruicheng/moge-2-vitl"
NUM_TOKENS = 1200


# ---------------------------------------------------------------- Angleichung

def local_affine(depth_m, est, radius_px=16, eps=0.001, min_frac=0.05, down=2):
    """Lokale Angleichung: in jedem Fenster echt ≈ a·est + b (Least Squares nur über
    gemessene Pixel), danach a/b über die Fenster gemittelt (Guided-Filter-Prinzip).
    a = lokale Skalierung, b = lokale Verschiebung [m]. Ohne Messungen in der Nähe: a=1, b=0.
    eps zieht a Richtung 1 (größer = eher reine Verschiebung, robuster bei wenig Messungen)."""
    H, W = depth_m.shape
    h, w = max(H // down, 1), max(W // down, 1)
    y = cv2.resize(depth_m, (w, h), interpolation=cv2.INTER_NEAREST)
    x = cv2.resize(est, (w, h), interpolation=cv2.INTER_NEAREST)
    M = ((y > 0) & (x > 0)).astype(np.float32)
    x = x * M
    y = y * M
    k = (2 * max(radius_px // down, 1) + 1,) * 2

    def bf(z):
        return cv2.boxFilter(z, -1, k, normalize=True, borderType=cv2.BORDER_REFLECT)

    n = bf(M)                                   # Anteil gemessener Pixel im Fenster
    ni = 1.0 / np.maximum(n, 1e-6)
    mx, my = bf(x) * ni, bf(y) * ni
    var = bf(x * x) * ni - mx * mx
    cov = bf(x * y) * ni - mx * my
    a = np.clip((cov + eps) / (var + eps), 0.25, 4.0)
    b = my - a * mx
    conf = np.clip(n / min_frac, 0, 1)          # zu wenige Messungen -> neutral
    a = 1 + conf * (a - 1)
    b = conf * b
    cw = bf(conf)                               # a/b über überlappende Fenster mitteln
    cwi = 1.0 / np.maximum(cw, 1e-6)
    fade = np.clip(cw / min_frac, 0, 1)
    a_m = 1 + fade * (bf(a * conf) * cwi - 1)
    b_m = fade * (bf(b * conf) * cwi)
    a_f = cv2.resize(a_m, (W, H), interpolation=cv2.INTER_LINEAR)
    b_f = cv2.resize(b_m, (W, H), interpolation=cv2.INTER_LINEAR)
    return a_f * est + b_f


def align_and_fill(depth_m, pred, prev_log_s=None, ema=0.3, radius_px=32, eps=0.01,
                   max_range=0.0, local_max=0.0, local_band=1.0):
    """depth_m: Messung [m], 0 = Loch.  pred: MoGe-Tiefe [m] (inf/0 = ungültig, z. B. Himmel).
    Gibt (gefüllt, Maske geschätzt, log_s) zurück."""
    valid = depth_m > 0
    pvalid = np.isfinite(pred) & (pred > 0)
    both = valid & pvalid

    # 1) globaler Skalierungsfaktor (robust: Median im log-Raum, auf jedem 4. Pixel),
    #    zeitlich geglättet
    sub = both[::4, ::4]
    if sub.sum() > 200:
        ratio = depth_m[::4, ::4][sub] / pred[::4, ::4][sub]
        log_s = float(np.log(np.median(ratio)))
        if prev_log_s is not None:
            log_s = (1 - ema) * prev_log_s + ema * log_s
    else:
        log_s = prev_log_s if prev_log_s is not None else 0.0
    est = pred.astype(np.float32) * np.float32(math.exp(log_s))
    est[~pvalid] = 0

    # 2) lokale Skalierung + Verschiebung nahe an Messungen
    if radius_px > 0:
        loc = local_affine(depth_m, est, radius_px, eps)
        if local_max > 0:
            # nur bis local_max voll lokal, dann über `local_band` Meter Übergang zu global
            wgt = np.clip((local_max + local_band - est) / local_band, 0, 1)
            loc = wgt * loc + (1 - wgt) * est
        est = np.where(pvalid, loc, 0).astype(np.float32)

    filled = np.where(valid, depth_m, est)
    estimated = ~valid & (filled > 0)
    if max_range > 0:
        drop = estimated & (filled > max_range)
        filled[drop] = 0
        estimated &= ~drop
    return filled, estimated, log_s


# ---------------------------------------------------------------- Gradienten-Fusion

def _block_stats(z, mask, f):
    """Maskierter Mittelwert, Varianz und Abdeckung je f×f-Block."""
    H, W = z.shape
    h, w = H // f, W // f
    m = mask[: h * f, : w * f].astype(np.float32)
    zm = z[: h * f, : w * f].astype(np.float32) * m
    area = lambda a: cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
    frac = area(m)
    fi = 1.0 / np.maximum(frac, 1e-6)
    mean = area(zm) * fi
    var = np.maximum(area(zm * zm) * fi - mean ** 2, 0)
    mean[frac == 0] = 0
    return mean, var, frac


class GradientFusion:
    def __init__(self, down=4, lam=1.0, mu_near=1e-4, mu_far=0.005, far_band=0.15,
                 sigma_edge=0.02, z_ref=4.0, outlier=1.0, iters=80, tol=1e-4):
        self.down, self.lam = down, lam
        self.mu_near, self.mu_far, self.far_band = mu_near, mu_far, far_band
        self.sigma_edge, self.z_ref, self.outlier = sigma_edge, z_ref, outlier
        self.iters, self.tol = iters, tol
        self._c_prev = None
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
        p = z.copy()
        rz = float((r * z).sum())
        bn = float(np.sqrt((b * b).sum())) + 1e-12
        for _ in range(self.iters):
            Ap = self._apply(p, diag, wx, wy)
            alpha = rz / max(float((p * Ap).sum()), 1e-30)
            x = x + alpha * p
            r = r - alpha * Ap
            if np.sqrt((r * r).sum()) < self.tol * bn:
                break
            z = r / diag
            rz_new = float((r * z).sum())
            p = z + (rz_new / max(rz, 1e-30)) * p
            rz = rz_new
        return x

    def __call__(self, depth_m, est):
        f = self.down
        H, W = depth_m.shape
        pv = est > 0
        both = (depth_m > 0) & pv
        logM = np.zeros(est.shape, np.float32)
        logM[pv] = np.log(est[pv])
        logZ = np.zeros_like(logM)
        logZ[both] = np.log(depth_m[both])
        r_full = np.where(both, logZ - logM, 0).astype(np.float32)

        gM, gMvar, gMfrac = _block_stats(logM, pv, f)
        r, rvar, frac = _block_stats(r_full, both, f)
        zlog, _, _ = _block_stats(logZ, both, f)

        # Messgewichte: Abdeckung, Sensorgenauigkeit ~1/z², Kanten/Streuung, Ausreißer
        wm = frac * np.minimum(1.0, np.exp(2 * (np.log(self.z_ref) - zlog)))
        wm *= np.exp(-(np.sqrt(gMvar) + np.sqrt(rvar)) / self.sigma_edge)
        wm[frac < 0.2] = 0
        has = wm > 0
        if has.any():
            wm *= 1.0 / (1.0 + ((r - np.median(r[has])) / self.outlier) ** 2)
            t = np.clip((gM - np.percentile(zlog[has], 98)) / self.far_band, 0, 1)
        else:
            t = np.ones_like(gM)
        mu = self.mu_near + (self.mu_far - self.mu_near) * t   # Abklingen hinter Messbereich

        # kantenbewusste Glattheit
        ok = gMfrac > 0.5
        wx = self.lam * np.exp(-np.abs(np.diff(gM, axis=1)) / self.sigma_edge) * (ok[:, 1:] & ok[:, :-1])
        wy = self.lam * np.exp(-np.abs(np.diff(gM, axis=0)) / self.sigma_edge) * (ok[1:, :] & ok[:-1, :])
        deg = np.zeros_like(gM)
        deg[:, :-1] += wx
        deg[:, 1:] += wx
        deg[:-1, :] += wy
        deg[1:, :] += wy

        x = self._c_prev if (self._c_prev is not None and self._c_prev.shape == gM.shape) \
            else np.zeros_like(gM)
        x = self._cg(wm * r, wm + mu + deg, wx, wy, x)
        wm2 = wm * np.minimum(1.0, self.outlier / np.maximum(np.abs(r - x), 1e-9))  # Huber
        x = self._cg(wm2 * r, wm2 + mu + deg, wx, wy, x)
        self._c_prev = x

        H2, W2 = H // 2, W // 2
        logM2 = cv2.resize(logM, (W2, H2), interpolation=cv2.INTER_NEAREST)
        c_half = self._guided_upsample(x.astype(np.float32), gM, logM2, H2, W2)
        out = est * np.exp(cv2.resize(c_half, (W, H), interpolation=cv2.INTER_LINEAR))
        out[~pv] = 0
        return out.astype(np.float32)

    def _guided_upsample(self, c, gM, logM, H, W):
        """Bilinear, Gitterpunkte mit ähnlicher MoGe-Tiefe zählen mehr (kantentreu)."""
        h, w = c.shape
        if self._up_cache is None or self._up_cache[0] != (h, w, H, W):
            ys = (np.arange(H, dtype=np.float32) + 0.5) * h / H - 0.5
            xs = (np.arange(W, dtype=np.float32) + 0.5) * w / W - 0.5
            y0 = np.clip(np.floor(ys).astype(np.int32), 0, h - 1)
            x0 = np.clip(np.floor(xs).astype(np.int32), 0, w - 1)
            y1, x1 = np.clip(y0 + 1, 0, h - 1), np.clip(x0 + 1, 0, w - 1)
            fy = np.clip(ys - np.floor(ys), 0, 1)[:, None]
            fx = np.clip(xs - np.floor(xs), 0, 1)[None, :]
            tabs = [((yy[:, None] * w + xx[None, :]).astype(np.int32), (wy * wx).astype(np.float32))
                    for yy, wy in ((y0, 1 - fy), (y1, fy)) for xx, wx in ((x0, 1 - fx), (x1, fx))]
            self._up_cache = ((h, w, H, W), tabs)
        cf, gf = c.ravel(), gM.astype(np.float32).ravel()
        acc = np.zeros((H, W), np.float32)
        wsum = np.full((H, W), 1e-12, np.float32)
        for idx, bw in self._up_cache[1]:
            d = np.abs(logM - gf[idx]) / self.sigma_edge
            wgt = bw / (1.0 + d * d)
            acc += wgt * cf[idx]
            wsum += wgt
        return acc / wsum


def fill(depth_m, pred, fusion, prev_log_s=None, ema=0.3):
    """Gibt (gefüllt, Maske geschätzt, log_s) zurück. depth_m: 0 = Loch, pred: inf = ungültig."""
    valid = depth_m > 0
    pvalid = np.isfinite(pred) & (pred > 0)
    sub = (valid & pvalid)[::4, ::4]
    log_s = prev_log_s if prev_log_s is not None else 0.0
    if sub.sum() > 200:
        cur = float(np.log(np.median(depth_m[::4, ::4][sub] / pred[::4, ::4][sub])))
        log_s = cur if prev_log_s is None else (1 - ema) * prev_log_s + ema * cur
    est = np.where(pvalid, pred * math.exp(log_s), 0).astype(np.float32)
    est = fusion(depth_m, est)
    filled = np.where(valid, depth_m, est)
    return filled, ~valid & (filled > 0), log_s


# ---------------------------------------------------------------- Gradienten-Fusion auf der GPU
# Gleiche Mathematik wie GradientFusion/fill oben, nur mit torch-Tensoren auf der GPU.

import torch.nn.functional as F  # noqa: E402



def _block_stats_t(z, m, f):
    """Wie _block_stats: avg_pool mit ganzzahligem Faktor = cv2 INTER_AREA."""
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


class GradientFusionTorch(GradientFusion):
    """Gleiche Parameter wie GradientFusion; Eingaben und Ausgabe sind torch-Tensoren."""

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

        gM, gMvar, gMfrac = _block_stats_t(logM, pv, f)
        r, rvar, frac = _block_stats_t(r_full, both, f)
        zlog, _, _ = _block_stats_t(logZ, both, f)

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
        mu = self.mu_near + (self.mu_far - self.mu_near) * t

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
        wm2 = wm * torch.clamp(self.outlier / torch.clamp(torch.abs(r - x), min=1e-9), max=1.0)
        x = self._cg(wm2 * r, wm2 + mu + deg, wx, wy, x)
        self._c_prev = x

        H2, W2 = H // 2, W // 2
        logM2 = logM[: 2 * H2: 2, : 2 * W2: 2]          # = cv2 INTER_NEAREST bei Faktor 2
        c_half = self._guided_upsample(x, gM, logM2, H2, W2)
        c_full = F.interpolate(c_half[None, None], size=(H, W), mode="bilinear",
                               align_corners=False)[0, 0]
        return torch.where(pv, est * torch.exp(c_full), zero)

    def _guided_upsample(self, c, gM, logM, H, W):
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
def fill_torch(depth_m, pred, fusion, prev_log_s=None, ema=0.3):
    """Wie fill(), aber auf der GPU. depth_m/pred: torch-Tensoren (float32, gleiches Gerät).
    Gibt (gefüllt, Maske geschätzt, log_s) als numpy zurück."""
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
    filled_np = filled.cpu().numpy()
    return filled_np, (~valid).cpu().numpy() & (filled_np > 0), log_s


# ---------------------------------------------------------------- MoGe

class MoGe:
    def __init__(self):
        from moge.model.v2 import MoGeModel
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MoGeModel.from_pretrained(MODEL).to(self.dev).eval()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    @torch.inference_mode()
    def __call__(self, rgb, fov_x):
        img = torch.from_numpy(rgb).to(self.dev).permute(2, 0, 1).float() / 255.0
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.dev.type == "cuda"):
            out = self.model.infer(img, fov_x=fov_x, num_tokens=NUM_TOKENS)
        d = out["depth"].float()
        if "mask" in out:
            d = torch.where(out["mask"].bool(), d, torch.full_like(d, float("inf")))
        return d          # torch-Tensor auf dem MoGe-Gerät

