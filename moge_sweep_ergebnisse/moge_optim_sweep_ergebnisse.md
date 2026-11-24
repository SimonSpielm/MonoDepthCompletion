# MoGe-2 Gradienten-Fusion – Parameter-Sweep Ergebnisse

**Datum:** 24.11.2026
**Skript:** `navigation/mapping/moge_depth_optim.py` (Fusion / Korrekturfeld c)
**Sweep-Tool:** `navigation/mapping/moge_sweep_optim.py` (neu, deterministisch)
**Daten:** gleicher Cache wie beim Affine-Sweep (60 Frames, rmc-lx0481, Domain 48,
1280×720, Tiefe 0.03–6.86 m, 28 % Löcher, MoGe-2 vitl, 1200 Tokens, fp16)

---

## Ergebnis

| Konfiguration | AbsRel | MAE | δ1 | Zeit/Frame |
|---|---|---|---|---|
| nur global, **ohne** Fusion | 12.9 % | 15.8 cm | 96.6 % | ~1 ms |
| alte Defaults (`down=8, lam=4, σ_edge=0.05, z_ref=2, outlier=0.1, μn=1e-3, μf=0.05`) | 5.48 % | 6.6 cm | 97.0 % | 38 ms |
| **neue Defaults (optimal)** | **3.27 %** | **4.1 cm** | **97.9 %** | 85 ms |

**→ 40 % weniger relativer Fehler** gegenüber den alten Defaults. Coverage 100 %.

### Fehler je Lochgröße (neues Optimum)

| 16 px | 32 px | 64 px |
|---|---|---|
| 2.80 % (3.5 cm) | 3.27 % (4.1 cm) | 3.75 % (4.7 cm) |

### Vergleich der beiden Ansätze (gleiche Daten, gleiche Metriken)

| Ansatz | AbsRel (gesamt) | MAE | Zeit/Frame |
|---|---|---|---|
| `moge_depth.py` – lokale affine, neue Defaults | **2.75 %** | **3.6 cm** | 9 ms |
| `moge_depth_optim.py` – Fusion, neues Optimum | 3.27 % | 4.1 cm | 85 ms |

Beide schlagen den reinen globalen Faktor um Faktor ~4. Die affine Angleichung bleibt
auf diesem Datensatz besser und ~10× schneller.

---

## Neue Parameter (ab sofort Defaults in `moge_depth_optim.py`)

```python
GradientFusion(down=4, lam=1.0, mu_near=1e-4, mu_far=0.005, far_band=0.15,
               sigma_edge=0.02, z_ref=4.0, outlier=1.0, iters=80, tol=1e-4)
```

| Parameter | alt | **neu** |
|---|---|---|
| `down` | 8 | **4** |
| `lam` | 4.0 | **1.0** |
| `mu_near` | 1e-3 | **1e-4** |
| `mu_far` | 0.05 | **0.005** |
| `far_band` | 0.15 | 0.15 (bleibt, ohne Effekt) |
| `sigma_edge` | 0.05 | **0.02** |
| `z_ref` | 2.0 | **4.0** |
| `outlier` | 0.1 | **1.0** |
| `iters` / `tol` | 80 / 1e-4 | 80 / 1e-4 (bleibt) |

Änderung: eine Zeile, die `__init__`-Signatur von `GradientFusion` (Zeile 62).

---

## Was der Sweep gezeigt hat (4 Stufen, ~270 Konfigurationen)

1. **`down=4` statt 8**: feineres Korrekturnetz, ~0.1 % besser, kostet 85 statt 38 ms.
2. **`lam=1`** (Plateau 0.3–2): schwache Glättung, MoGe gibt die Form vor.
3. **`sigma_edge=0.02`** (0.01–0.03 fast identisch): der alte Wert 0.05 hat das
   Korrekturfeld zu stark an Kanten/Streuung gedämpft — größter Einzel-Hebel.
4. **`outlier=1.0`** (praktisch keine Ausreißer-Ausscheidung): jede Penalisation
   (0.1/0.3/0.5) macht die Sache *schlechter* — Kinect-Messungen sind zu konsistent,
   robuste Gewichtung schadet mehr als sie nützt.
5. **`mu_near=1e-4`, `mu_far=0.005`**: je kleiner, desto besser; die Priors dürfen
   kaum wirken, sonst wird der Korrekturfreiraum hinter dem Messbereich eingeengt.
6. **`z_ref` (1/2/4) und `far_band` (0.05/0.15/0.5): praktisch egal.**
7. **`iters`/`tol`:** kein Hebel (CG bricht bei tol früh ab).

---

## Caveats

- Gleiche Datenbasis wie der Affine-Sweep: **eine Szene, ~4 s Stand-Aufnahme.**
  Plateaus sind flach (lam 0.3–2, σ 0.01–0.03), Overfit-Risiko gering.
- Synthetische Holdout-Blöcke ≠ echte Kinect-Löcher; der 64px-Wert (~3.8 %) ist der
  realistischere Richtwert.
- > 6.9 m ungetestet.
- Fusion wird pro Frame frisch gelöst (kein Warmstart der Produktion) — Ergebnis
  identisch (CG deterministisch), aber Timing ohne Warmstart-Vorteil.

---

## Wiederverwendung / Re-Tuning

```bash
# Predictions-Cache weiterverwenden (kein erneutes Inferieren):
ssh rmc-lx0481 "/home/spie_si/miniforge3/envs/moge/bin/python \
  /home/spie_si/Documents/main/navigation/mapping/moge_sweep_optim.py \
  --preds /home/spie_si/Documents/main/navigation/mapping/moge_sweep_out/preds.npz \
  --out-dir /home/spie_si/Documents/main/navigation/mapping/moge_sweep_optim_out_x \
  --down 4 --lam 1 --sigma-edge 0.02 --z-ref 4 \
  --outlier 0.5,1.0,2.0 --mu-far 0.003,0.005,0.01 \
  --blocks 16,32,64 --reps 2"
```

### Dateien

| Datei | Inhalt |
|---|---|
| `navigation/mapping/moge_sweep_optim.py` | Sweep-Pipeline für die Fusion |
| `navigation/mapping/moge_sweep_optim_out_A/` | Stufe A: down × lam × sigma_edge × z_ref (81 Konfigs) |
| `navigation/mapping/moge_sweep_optim_out_B/` | Stufe B: Rand-Expansion (18 Konfigs) |
| `navigation/mapping/moge_sweep_optim_out_C/` | Stufe C: outlier × μn × μf × far_band (81 Konfigs) |
| `navigation/mapping/moge_sweep_optim_out_final/` | Finale: outlier-Rand + 3 Lochgrößen × 2 reps |

Jedes Verzeichnis: `sweep_rows.csv`, `sweep_summary_per_block.csv`,
`sweep_summary_overall.csv`, `sweep_baseline.csv`.
