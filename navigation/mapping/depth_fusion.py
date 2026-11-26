"""
Tiefenvervollständigung ohne ROS: Sensortiefe mit Löchern + Vorhersage eines monokularen Modells
(depth_models.py) -> gefüllte Tiefe. Einstieg ist fill().

  1. Globaler Faktor: Median(Messung / Vorhersage), über die Zeit geglättet (ema).
  2. Gradienten-Fusion: Korrekturfeld c mit log D = log M + c, das die Messungen trifft, entlang
     von Oberflächen glatt ist (das Modell gibt die Form vor), nicht über Kanten wirkt und weit
     hinter dem Messbereich auf 0 abklingt. Vorkonditioniertes CG auf der GPU.
  3. Kantenfilter gegen schwebende Punkte. Das Modell zeichnet Tiefensprünge als weiche Rampe;
     genau dort misst der Sensor oft nichts, die Rampe würde als Punkte zwischen Vorder- und
     Hintergrund in der Karte landen. Verworfen werden gefüllte Pixel, die
       a) auf der Rampe selbst liegen: zu steil für eine echte Fläche (ramp_mask), oder
       b) in der Zone um eine gemessene Tiefenkante weder zur Vorder- noch zur Hinterseite
          passen (supported).
     Gemessene Pixel bleiben immer unverändert. Ein Loch ist harmlos, ein Punkt in der Luft nicht.

Genutzt vom ROS-Knoten (moge_depth_completion.py), vom Viewer (dev/depth_viewer.py) und vom
Offline-Benchmark (bench/), die damit immer denselben Code rechnen.
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

    def __init__(self, down=2, lam=1.0, mu_near=1e-4, mu_far=0.005, far_band=0.15,
                 sigma_edge=0.05, z_ref=4.0, outlier=1.0, iters=40, tol=1e-4):
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
        for i in range(self.iters):
            Ap = self._apply(p, diag, wx, wy)
            alpha = rz / torch.clamp((p * Ap).sum(), min=1e-30)
            x = x + alpha * p
            r = r - alpha * Ap
            # Abbruchtest nur alle 10 Iterationen: jeder Test erzwingt eine GPU-Synchronisation
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
def fill(depth_m, pred, fusion, prev_log_s=None, ema=0.3, f_px=None,
         ramp_dmin=0.5, support_px=12, support_rtol=0.02, support_edge=0.1):
    """Füllt die Löcher von depth_m. Gibt (gefüllt [m], log_s) zurück; 0 = leer.

    depth_m: Sensortiefe [m], 0 = Loch. pred: Modelltiefe [m], inf = ungültig. Beides
    torch-Tensoren (H, W) auf demselben Gerät. prev_log_s: Faktor des letzten Frames (log).
    f_px: Brennweite [px] für den Rampenfilter (ohne f_px ist er aus).

    Kantenfilter (Ablauf: Kante finden -> Zone um die Kante -> dort prüfen):
      ramp_dmin   [m]  Schritt 1, ramp_mask: verschmierte Kante des Modells verwerfen. 0 = aus.
                       Muss unter der Kamerahöhe liegen (Boden: Abstand = Kamerahöhe ≈ 0,58 m).
      support_px  [px] Schritt 2, supported: Umkreis, in dem nach echten Messungen geschaut
                       wird (= Breite der Zone um Tiefenkanten). 0 = aus.
      support_edge     Zone liegt vor, wenn die Messungen im Umkreis um mehr als 10 % streuen.
      support_rtol     erlaubte Abweichung zur Vorder- bzw. Hinterseite (2 %).
    """
    valid = depth_m > 0
    pvalid = torch.isfinite(pred) & (pred > 0)

    # 1. globaler Faktor (robust: Median, jedes 4. Pixel genügt)
    sub = (valid & pvalid)[::4, ::4]
    log_s = prev_log_s if prev_log_s is not None else 0.0
    if int(sub.sum()) > 200:
        ratio = depth_m[::4, ::4][sub] / pred[::4, ::4][sub]
        cur = float(torch.log(torch.quantile(ratio, 0.5) if ratio.numel() <= 16_000_000
                              else ratio.median()))
        log_s = cur if prev_log_s is None else (1 - ema) * prev_log_s + ema * cur
    est = torch.where(pvalid, pred * math.exp(log_s), torch.zeros_like(pred))

    # 2. lokale Korrektur
    est = fusion(depth_m, est)

    # 3. Kantenfilter: nur gefüllte Pixel werden verworfen, Messungen bleiben
    keep = est > 0
    if ramp_dmin > 0 and f_px:
        keep &= ~ramp_mask(est, f_px, ramp_dmin)
    if support_px > 0:
        keep &= supported(est, depth_m, support_px, support_rtol, support_edge)
    filled = torch.where(valid, depth_m, torch.where(keep, est, torch.zeros_like(est)))
    return filled, log_s


def ramp_mask(est, f_px, dmin, s=2):
    """Schritt 1: Pixel auf der verschmierten Kante des Modells ("zu steil, um echt zu sein").

    Das Modell zeichnet einen Sprung von 1,5 m auf 3 m als Rampe 1,6 -> 1,9 -> 2,4 -> 2,8 m.
    So steil kann eine echte Fläche im Bild nur sein, wenn man fast genau an ihr entlang
    schaut. Formal: Für eine Ebene im Abstand d vom Kamerazentrum gilt f·|∇(1/z)| ≤ 1/d. Eine
    Rampe läuft verlängert fast durch die Kamera (d ≈ 0), echte Flächen nicht: Der Boden hat in
    jeder Entfernung d = Kamerahöhe. Ist f·|∇(1/z)| > 1/dmin, wird das Pixel verworfen.
    Die Breite der Maske passt sich so von selbst an die Breite der Rampe an.
    Gradient als zentrale Differenz über ±s Pixel; Paare mit ungültigem Nachbarn zählen nicht.
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
    """Schritt 2: passt der Füllwert zu den echten Messungen im Umkreis r?

    Im (2r+1)²-Fenster werden kleinste (dmin) und größte (dmax) Messung bestimmt.
      - keine Messung im Fenster (Inneres großer Löcher): bleibt.
      - Messungen ähnlich (dmax ≤ dmin·(1+edge)), glatte Fläche: est muss in
        [dmin·(1-rtol), dmax·(1+rtol)] liegen.
      - Messungen streuen stark: Pixel liegt in der Zone um eine Tiefenkante (Kante um r px
        "aufgebläht"). est muss auf ±rtol zur Vorderseite (dmin) ODER Hinterseite (dmax) passen,
        Werte dazwischen wären Punkte in der Luft.
    Gibt die Maske der Pixel zurück, die bleiben dürfen.
    """
    v = depth_m > 0
    k = 2 * r + 1

    def mx(a):                                   # separierbares Max-Pooling, k×k
        a = F.max_pool2d(a[None, None], (1, k), 1, (0, r))
        return F.max_pool2d(a, (k, 1), 1, (r, 0))[0, 0]

    dmax = mx(torch.where(v, depth_m, torch.zeros_like(depth_m)))
    dmin = -mx(-torch.where(v, depth_m, torch.full_like(depth_m, 1e9)))
    inside = (est >= dmin * (1 - rtol)) & (est <= dmax * (1 + rtol))
    near_side = (torch.abs(est - dmin) <= rtol * dmin) | (torch.abs(est - dmax) <= rtol * dmax)
    is_edge = dmax > dmin * (1 + edge)
    return (dmax <= 0) | (inside & (~is_edge | near_side))
