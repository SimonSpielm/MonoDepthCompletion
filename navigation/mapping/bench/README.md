# Offline-Benchmark für die Tiefenvervollständigung

Testet **genau den Produktionscode** (`navigation/mapping/depth_fusion.py`, den auch der
ROS-Knoten `moge_depth_completion.py` importiert) ohne ROS, Kamera oder Roboter, auf einem
Laptop. Damit lassen sich Parameter und Code-Änderungen deterministisch vergleichen, bevor
sie auf den Roboter gehen.

## Einrichten (einmalig)

```bash
conda create -y -n depthbench python=3.12
```

```bash
conda run -n depthbench pip install torch numpy opencv-python-headless
```

Nur wenn MoGe lokal laufen soll (für `--moge`; lädt beim ersten Lauf ~1,3 GB Gewichte):

```bash
conda run -n depthbench pip install git+https://github.com/microsoft/MoGe.git
```

## Daten

| Quelle | Woher | Wofür |
|---|---|---|
| `--synthetic N` | wird erzeugt | Boden, schräge Wand, Kisten, dünne Stangen. Sensor ToF-artig (Löcher an Kanten, dunkle Flecken, Reichweitenabfall, Mischpixel), Vorhersage MoGe-artig (weiche Kantenrampen, glatter Skalenfehler). Ground Truth überall bekannt. Schnell, gut für Kantenexperimente. |
| `--frames F.npz --preds P.npz` | `moge_sweep.py capture` und `infer` auf dem Roboter | echte Szenen. `preds.npz` enthält die MoGe-Vorhersagen, MoGe muss dann lokal nicht laufen. |
| `--frames F.npz --moge` | wie oben, ohne Vorhersage-Cache | rechnet MoGe lokal (CUDA, Apple MPS oder CPU) und legt `F_preds_<tokens>.npz` an. Nötig für Experimente mit `--num-tokens`. |

## Metriken

Jeder Frame wird mit einer frischen Fusion ausgewertet (ohne Warmstart und EMA), damit Läufe
vergleichbar sind.

- `hold`: allgemeine Genauigkeit. Bei Aufnahmen werden zufällige 32-px-Blöcke gemessener Pixel
  ausgeblendet (Protokoll der bisherigen Sweeps); synthetisch zählen alle echten Löcher abseits
  von Kanten.
- `edge`: Kantenverhalten. Bei Aufnahmen wird ein 6-px-Band um gemessene Tiefenkanten ausgeblendet
  (das, was der ToF an Kanten tatsächlich verliert); synthetisch zählen alle echten Löcher bis
  6 px neben einer Kante.
  - `bleed`: Anteil der Pixel, die mit > 10 % Fehler gefüllt wurden, also schwebende Punkte und
    Kometenschweife in der Karte. Kleiner ist besser.
  - `cov`: Anteil der Pixel, die überhaupt einen Wert bekommen. Immer zusammen mit `bleed`
    lesen: Wer nichts füllt, hat `bleed` 0, aber auch `cov` 0.
- `ms`: Median der Laufzeit der Füllung pro Frame auf `--device`. Ist nicht mit der A6000
  vergleichbar, nur als relatives Maß.

## Beispiele

```bash
python depth_bench.py --synthetic 8
```

```bash
python depth_bench.py --synthetic 8 --sweep edge_rtol=0,0.03,0.04,0.06 sigma_edge=0.02,0.05 --device mps
```

```bash
python depth_bench.py --frames frames.npz --preds preds.npz --set down=8 --csv out.csv
```

Mit `--set` und `--sweep` lassen sich alle Parameter von `GradientFusion`
(`down, lam, mu_near, mu_far, far_band, sigma_edge, z_ref, outlier, iters, tol`) sowie
`ema` und `edge_rtol` von `fill` setzen.

## Erster Befund (synthetisch, 4 Frames)

| `edge_rtol` | hold AbsRel | edge bleed | edge cov |
|---|---|---|---|
| 0 (aus) | 0,78 % | 58 % | 100 % |
| 0,04 | 0,78 % | 0 % | 9 % |

Ohne den Kantenfilter wird mehr als die Hälfte der Kantenlöcher falsch gefüllt. Mit Filter
verschwinden die Fehler, aber auch fast alle Füllungen an Kanten. Ziel weiterer Optimierung:
Kantenlöcher dem Vorder- *oder* Hintergrund zuordnen, statt sie zu verwerfen.
