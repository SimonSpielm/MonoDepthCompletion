# Offline-Benchmark für die Tiefenvervollständigung

Testet **genau den Produktionscode** (`../depth_fusion.py` und `../depth_models.py`, die auch der
ROS-Knoten `depth_completion_node.py` importiert) ohne ROS, Kamera oder Roboter, auf einem
Laptop. Damit lassen sich Parameter und Code-Änderungen deterministisch vergleichen, bevor
sie auf den Roboter gehen.

## Einrichten (einmalig)

```bash
conda create -y -n depthbench python=3.12
```

```bash
conda run -n depthbench pip install torch numpy opencv-python-headless
```

Für lokale Vorhersagen mit **Apple Depth Pro 2** (`--predict depthpro2-s|b|l|h`, `compare_models.py`):

```bash
conda run -n depthbench pip install git+https://github.com/apple-aiml-research/ml-depth-pro-2.git
```

Die Gewichte lädt das Paket beim ersten Start selbst nach `~/.cache/torch/hub/checkpoints`
(S 97 MB, B 386 MB, dazu 94 MB FoV-Schätzer). Sie sind nur für nicht-kommerzielle Forschung
freigegeben (`LICENSE_MODEL` im Repo).

Für MoGe (`--predict moge|moge2-s|moge2-b`): das Repo klonen und ohne Abhängigkeiten installieren,
weil `flex-gemm` (nur MoGe-3) Triton/CUDA braucht und auf macOS nicht baut:

```bash
conda run -n depthbench pip install --no-deps git+https://github.com/microsoft/MoGe.git
```

```bash
conda run -n depthbench pip install rosbags scipy trimesh huggingface-hub click "utils3d_moge @ git+https://github.com/EasternJournalist/utils3d-moge.git" "pipeline @ git+https://github.com/EasternJournalist/pipeline.git"
```

## Daten

| Quelle | Woher | Wofür |
|---|---|---|
| `--synthetic N` | wird erzeugt | Boden, schräge Wand, Kisten, dünne Stangen. Sensor ToF-artig (Löcher an Kanten, dunkle Flecken, Reichweitenabfall, Mischpixel), Vorhersage MoGe-artig (weiche Kantenrampen, glatter Skalenfehler). Ground Truth überall bekannt. Schnell, gut für Kantenexperimente. |
| `--frames F.npz --preds P.npz` | `moge_sweep.py capture` und `infer` auf dem Roboter | echte Szenen. `preds.npz` enthält die MoGe-Vorhersagen, MoGe muss dann lokal nicht laufen. |
| `--frames F.npz --predict depthpro2-s` | wie oben, ohne Vorhersage-Cache | rechnet das Modell lokal (CUDA, Apple MPS oder CPU) und legt `F_preds_<modell>.npz` an. |
| `bag_to_frames.py BAG --out F.npz` | ROS-1- oder ROS-2-Bag (`pip install rosbags`, kein ROS nötig) | RGB/Tiefe-Paare mit gleichem Zeitstempel, FoV aus `camera_info`. |

`--frames` nimmt mehrere Dateien und mittelt über alle. Die Aufnahmen liegen in `rosbags/frames/`
(nicht in git).

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
- `<3m`: dieselben Werte nur für Pixel mit Tiefe unter 3 m (`AbsRel`, mittlerer Fehler in `cm`).
- `--hold-px`: Kantenlänge der ausgeblendeten Blöcke (Standard 32). Mit 128 entstehen große
  Löcher wie bei schwarzen Objekten oder Glas, in denen die Stützprüfung nicht mehr greift.
- `ms`: Median der Laufzeit der Füllung pro Frame auf `--device`. Ist nicht mit der A6000
  vergleichbar, nur als relatives Maß. Gesamtlaufzeit Modell + Füllung: `speed.py`.

## Beispiele

```bash
python depth_bench.py --synthetic 8
```

```bash
python depth_bench.py --synthetic 8 --sweep ramp_dmin=0,0.5 support_px=0,12
```

```bash
python depth_bench.py --frames frames.npz --preds preds.npz --set down=8 --csv out.csv
```

Mit `--set` und `--sweep` lassen sich alle Parameter von `GradientFusion`
(`down, lam, mu_near, mu_far, far_band, sigma_edge, z_ref, outlier, iters, tol`) sowie die
Parameter von `fill` (`ema, ramp_dmin, support_px, support_rtol, support_edge`) setzen.
`ramp_dmin=0 support_px=0` schaltet die Kantenfilter ab. Ohne Angabe gelten die Standardwerte aus `depth_fusion.py` (= Produktion).

## Laufzeit

```bash
python speed.py --frames f.npz --models moge2-s:600 moge2-b:600 moge:1200 depthpro2-s
```

## Modelle direkt vergleichen

```bash
python compare_models.py bild.jpg --models depthpro2-s depthpro2-b
```

```bash
python compare_models.py --frames frames.npz --index 0
```

Schreibt `compare_models.png`: RGB, je Modell die Tiefe sowie die Kanten (weiß = harter Sprung,
orange = weiche Rampe). Dazu Laufzeit und Anteil der Rampen-Pixel. Bei Außenszenen zählt weit
entfernter, flach gesehener Boden ebenfalls als Rampe; aussagekräftig ist die Kennzahl vor allem
in Innenräumen.

Gemessen auf einem M4 Pro (MPS): Depth Pro 2 S 130 ms, B 326 ms pro Frame bei 1280×720.

## Ergebnisse (6 Aufnahmen, je 2 Frames, Fusion auf MPS)

Kantenfilter, Depth Pro 2 S, kleine Löcher (32 px):

| Kantenbehandlung | <3m AbsRel | edge bleed | edge cov |
|---|---|---|---|
| keine | 2,48 % | 20,9 % | 100 % |
| alter 3×3-Kantenfilter (inzwischen entfernt) | 0,86 % | 3,95 % | 68 % |
| neu: Rampenfilter + Stützprüfung (Standard) | 0,40 % | 0,00 % | 5 % |

Einrasten von Zwischenwerten auf die nähere Seite statt Verwerfen wurde getestet und verworfen
(21 % Bleeding, landet oft auf der falschen Seite). Die Fusionsparameter (`down`, `lam`,
`sigma_edge`) ändern den Fehler in großen Löchern kaum; er wird vom Modell bestimmt.

Modelle, Laufzeit auf M4 Pro (MPS, 1280×720, Modell + Füllung mit 40 CG-Iterationen), Fehler und
Bleeding unter 3 m:

| Modell | Tokens | ms | Hz | <3m kleine Löcher | <3m große Löcher (128 px) | edge bleed |
|---|---|---|---|---|---|---|
| MoGe-2 S (vits-normal) | 600 | 85 | 11,8 | 0,43 % / 0,6 cm | 3,80 % / 4,5 cm | 0,00 % |
| MoGe-2 S | 1200 | 134 | 7,5 | 0,41 % / 0,6 cm | 3,42 % / 4,1 cm | 0,00 % |
| **MoGe-2 B (vitb-normal)** | **600** | **107** | **9,4** | 0,40 % / 0,6 cm | 3,06 % / 3,7 cm | 0,00 % |
| MoGe-2 B | 700 | 117 | 8,5 | – | – | – |
| MoGe-2 B | 1200 | 182 | 5,5 | 0,39 % / 0,6 cm | 2,79 % / 3,2 cm | 0,00 % |
| MoGe-2 L (vitl, bisher) | 1200 | 306 | 3,3 | 0,37 % / 0,6 cm | 1,89 % / 2,6 cm | 0,00 % |
| Depth Pro 2 S | – | 153 | 6,5 | 0,43 % / 0,7 cm | 3,35 % / 4,4 cm | 0,00 % |
| Depth Pro 2 B | – | 348 | 2,9 | 0,41 % / 0,6 cm | 2,24 % / 3,3 cm | 0,00 % |
| Depth Pro 2 L | – | 1029 | 1,0 | 0,38 % / 0,6 cm | 1,59 % / 2,4 cm | 0,00 % |

Für 8 Hz ist MoGe-2 B mit 600 Tokens die Wahl (Standard im ROS-Knoten); im Knoten kommen noch
Konvertierung und Publish hinzu. MoGe-3 braucht für seine Verfeinerung CUDA und wurde hier nicht
verglichen.
