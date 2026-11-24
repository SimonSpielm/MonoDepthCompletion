# MoGe-2 Tiefenfüllung – Parameter-Sweep Ergebnisse

**Datum:** 24.11.2026
**System:** rmc-lx0481 (ROS domain 48), RTX A6000, conda-env `moge`
**Skript:** `navigation/mapping/moge_sweep.py` (neu, reproduzierbar)

---

## Datenbasis

- 60 synchronisierte Frames (~4 s, 15 Hz) live vom Roboter aufgenommen
- `/camera/color/image_raw` + `/camera/depth/image_raw` (Azure Kinect), 1280×720
- Tiefe: 0.03–6.86 m, **28 % Löcher** in der Roh-Tiefe
- MoGe-2 `Ruicheng/moge-2-vitl`, fp16, num_tokens=1200 (Vorhersage 1× pro Frame gecacht)

## Auswertungs-Setup (deterministisch)

- Pro Frame werden zufällige Blöcke **gemessener** Pixel ausgeblendet (15 % Fläche),
  mit fester Seed, 2 unabhängige Realisierungen → gefüllt und mit dem echten Wert verglichen.
- 3 Lochgrößen (16/32/64 px), damit die Fenstergröße zur Lochgröße passt.
- 3 Sweep-Runden, ~530 Konfigurationen, ~190.000 Einzel-Messungen.
- Metriken: AbsRel (mittlerer relativer Fehler), MAE, RMSE, δ1 (<25 %),
  Coverage (Anteil gefüllter Löcher-Pixel), AbsRel je Distanzklasse.

---

## Ergebnis

| Konfiguration | AbsRel | MAE | δ1 | RMSE | Zeit/Frame |
|---|---|---|---|---|---|
| nur globaler Faktor (`radius=0`) | 12.8 % | 15.6 cm | 96.7 % | 0.40 m | ~1 ms |
| alte Defaults (`r=32, eps=0.01, down=4, clip 0.5–2`) | 4.50 % | 5.5 cm | 95.9 % | 0.21 m | 4 ms |
| **neue Defaults (optimal)** | **2.75 %** | **3.6 cm** | **97.5 %** | **0.14 m** | 9 ms |

**→ 39 % weniger relativer Fehler** gegenüber dem alten Setup. Coverage 99.9 %.

### Fehler je Distanzklasse (neues Optimum)

| 0–1 m | 1–2 m | 2–3 m | 3–4 m | 4–6 m | > 6 m |
|---|---|---|---|---|---|
| 4.8 % | 1.5 % | 4.1 % | 2.4 % | – | 4.3 % |

### Fehler je Lochgröße (neues Optimum, radius=16)

| 16 px | 32 px | 64 px |
|---|---|---|
| ~1.9 % | ~2.5 % | ~3.5 % |

→ Optimaler radius wächst mit der Lochgröße (r=8 für 16px, r=12 für 32px, r=16 für 64px).
r=16 ist der beste Einzel-Kompromiss für alle Größen.

---

## Neue Parameter (ab sofort Defaults in `moge_depth.py`)

| Parameter | alt | **neu** |
|---|---|---|
| `radius` | 32 | **16** |
| `eps` | 0.01 | **0.001** |
| `down` (Fenster-Downsampling) | 4 | **2** |
| `clip` für Skalierung a | 0.5–2.0 | **0.25–4.0** |
| `local_max` | 0 (aus) | 0 (bleibt aus) |
| `min_frac` | 0.05 | 0.05 (bleibt) |
| `ema` (log_s) | 0.3 | 0.3 (bleibt, siehe unten) |

Änderungen in `navigation/mapping/moge_depth.py` (4 Zeilen):
- Zeile 75: `local_affine(..., radius_px=16, eps=0.001, ..., down=2)`
- Zeile 97: `np.clip(..., 0.25, 4.0)`
- Zeile 357: `--radius` default 16
- Zeile 359: `--eps` default 0.001

---

## Was der Sweep gezeigt hat

1. **Kleinere Fenster gewinnen**: radius 16 schlägt 32/48/64/96 in allen Lochgrößen.
2. **Kaum Regularisierung**: eps 0.001 optimal, flaches Plateau 0.0005–0.002 (robust).
3. **`local_max` ist nutzlos**: Distanz-Gating (3 m/4 m) verschlechtert die Ergebnisse
   (5.6–5.8 % statt 4.5 %) → Feature kann man sich sparen.
4. **Clip begrenzte die Korrektur**: 0.5–2.0 schnitt lokale Skalierungen >2×/0.5× ab;
   Aufweitung bringt ~0.15 % (Glas/Spiegel/feine Strukturen).
5. **down=2**: feinere a/b-Lokalisierung, +0.15 %, +5 ms (bei Latenzbedenklichkeit
   auch down=4 mit 2.90 % vertretbar).
6. **Verdacht auf Konfidenz-Doppelgewichtung (conf²) und nicht-skalen-invariantes eps:
   auf echten Daten praktisch ohne Effekt** (Δ ≤ 0.02 %) → kein Fix nötig.
7. **EMA (0 / 0.1 / 0.3): messbar identisch** → Zeitglättung des globalen Faktors ist
   für die Single-Frame-Genauigkeit irrelevant.

---

## Caveats

- **Eine Szene, ~4 s Stand-Aufnahme.** Plateaus sind aber flach (eps über Faktor 4,
  radius 12–16 fast identisch) → geringes Overfit-Risiko, aber nicht ausgeschlossen.
- Synthetische Holdout-Blöcke ≠ echte Kinect-Löcher (Glas, Schwarzes, Spiegelflächen
  sind gehäuft und größer). Der **64px-Wert (~3.5 %) ist der realistischere Richtwert**.
- > 6.9 m war in den Daten nicht enthalten (Ferne ungetestet).

---

## Wiederverwendung / Re-Tuning

```bash
# auf rmc-lx0481, conda-env `moge`:
# 1) neue Szene aufnehmen
ROS_DOMAIN_ID=48 LD_LIBRARY_PATH=/opt/ros/jazzy/lib \
PYTHONPATH=/opt/ros/jazzy/lib/python3.12/site-packages \
  ~/miniforge3/envs/moge/bin/python navigation/mapping/moge_sweep.py capture \
  --frames 120 --out navigation/mapping/moge_sweep_out/frames2.npz

# 2) Inferenz + Sweep (preset: quick / full oder freie Grids)
~/miniforge3/envs/moge/bin/python navigation/mapping/moge_sweep.py all \
  --frames navigation/mapping/moge_sweep_out/frames2.npz \
  --out navigation/mapping/moge_sweep_out/preds2.npz \
  --out-dir navigation/mapping/moge_sweep_out_r4 --preset quick
```

### Dateien

| Datei | Inhalt |
|---|---|
| `navigation/mapping/moge_sweep.py` | Sweep-Pipeline (capture / infer / sweep / all) |
| `navigation/mapping/moge_sweep_out/` | Runde 1: CSVs + `preds.npz` (332 MB Cache) |
| `navigation/mapping/moge_sweep_out_r2/` | Runde 2: Feinschliff + Global-Baseline |
| `navigation/mapping/moge_sweep_out_r3/` | Runde 3: finale Bestätigung |

Jedes Verzeichnis enthält: `sweep_rows.csv` (alle Einzel-Messungen),
`sweep_summary_per_block.csv`, `sweep_summary_overall.csv`.
