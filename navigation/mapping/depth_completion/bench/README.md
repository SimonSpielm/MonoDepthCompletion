# Offline-Benchmark für die Tiefenvervollständigung

Testet **genau den Produktionscode** (`../depth_fusion.py` und `../depth_models.py`, die auch der
ROS-Knoten `depth_completion_node.py` importiert) ohne ROS, Kamera oder Roboter. Damit lassen sich Parameter und Code-Änderungen deterministisch vergleichen, bevor
sie auf den Roboter gehen.

## Einrichten

Dasselbe conda-Env wie für den ROS-Knoten, siehe [../README.md](../README.md), Abschnitt
„Installation“. ROS wird für den Benchmark nicht gebraucht. Die Gewichte von Depth Pro 2 sind nur
für nicht-kommerzielle Forschung freigegeben.

## Testsuite auf dem Roboter-PC (`run_suite.py`)

Lässt alle Modellkonfigurationen auf aufgenommenen Szenen laufen und schreibt einen Bericht zur
Auswertung. Gemessen wird pro Konfiguration: Laufzeit (Modell, Füllung, gesamt, Hz), GPU-Speicher,
Genauigkeit (32- und 128-px-Löcher, grobe Fehler, Kantenband) und Flackern bei stillstehender
Kamera. Standard-Raster (`--list`, 27 Konfigurationen):

- MoGe-2 S, B, L, L-normal mit 600, 1200 und 2400 Tokens
- MoGe-3 L mit Verfeinerung (600, 1200, 2400 Tokens) und ohne (1200), MoGe-3 G (600, 1200)
- Depth Pro 2 S, B, L, jeweils Einzelbild, Stream Fenster 4 und 16 (H mit `--with-h`)

Erst ein kurzer Durchlauf (5 Konfigurationen, etwa 10 Minuten), dann alles (eher 1–2 Stunden).
Die ROS-Pipeline dabei nicht laufen lassen, sie teilt sich die GPU.

```bash
python run_suite.py --quick --bags ~/Desktop/rosbags/depth_estimation ~/Desktop/rosbags/octomap/octomap ~/Desktop/rosbags/local_map/azure_kinect
```

```bash
python run_suite.py --bags ~/Desktop/rosbags/depth_estimation ~/Desktop/rosbags/octomap/octomap ~/Desktop/rosbags/local_map/azure_kinect
```

- Bags werden einmal nach `~/depth_suite_data` umgewandelt (30 aufeinanderfolgende Frames, `--n`,
  `--step`). ROS-1-Bags (`.bag`) gehen auch.
- Der Bericht landet in `bench/reports/<rechner>_<zeit>/`: `report.md` (Tabelle), `summary.csv`,
  `results.jsonl` (inklusive Werten pro Aufnahme und Fehlermeldungen), `env.json` (GPU, Treiber,
  Versionen, Commit), `suite.log`.
- Bricht der Lauf ab, setzt `--resume bench/reports/<ordner>` an der Stelle fort.
- Einzelne Konfigurationen: `--only moge2-b:600 moge3-l:1200:refine3 depthpro2-s:stream4`.
- MoGe-3 mit Verfeinerung braucht `flex_gemm` (Installation Schritt 6); fehlt es, werden diese
  Konfigurationen als fehlgeschlagen protokolliert und der Rest läuft weiter.

Zur Auswertung den Berichtsordner committen und pushen, oder gezippt schicken.

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
  Löcher wie bei schwarzen Objekten oder Glas, weit weg von jeder Messung.
- `ms`: Median der Laufzeit der Füllung pro Frame auf `--device`. Ist nicht mit der A6000
  vergleichbar, nur als relatives Maß. Gesamtlaufzeit Modell + Füllung: `speed.py`.
- `--temporal`: Flackern statt Genauigkeit. Jede `--frames`-Datei läuft als Sequenz mit
  Gedächtnis wie im Live-Knoten (geglätteter Faktor, Warmstart). Gemessen wird die mittlere
  relative Änderung zwischen aufeinanderfolgenden Frames: auf gefüllten Pixeln (`filled`, auch
  unter 3 m), zum Vergleich für das Modell allein (`model`) und für den Sensor (`sensor`). Sinnvoll
  nur mit Aufnahmen einer stillstehenden Kamera.

## Beispiele

```bash
python depth_bench.py --synthetic 8
```

```bash
python depth_bench.py --synthetic 8 --sweep edge_jump=0,0.015,0.02 edge_grow=4,8
```

```bash
python depth_bench.py --frames frames.npz --preds preds.npz --set down=8 --csv out.csv
```

```bash
python depth_bench.py --frames a.npz b.npz --predict moge2-b --num-tokens 600 --limit 15 --temporal
```

Mit `--set` und `--sweep` lassen sich alle Parameter von `GradientFusion`
(`down, lam, mu_near, mu_far, far_band, sigma_edge, z_ref, outlier, iters, tol`) sowie die
Parameter von `fill` (`ema, edge_jump, edge_grow`) setzen. `edge_jump=0` schaltet den
Kantenfilter ab. Ohne Angabe gelten die Standardwerte aus `depth_fusion.py` (= Produktion).

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

## Ergebnisse (6 Aufnahmen, Fusion auf MPS)

### Verfahren im Vergleich (MoGe-2 B, 600 Tokens)

Genauigkeit mit je 2 Frames pro Aufnahme, Flackern mit je 15 aufeinanderfolgenden Frames (13
bei `rosbag`). Fehler unter 3 m.

| Verfahren | kleine Löcher | große Löcher (128 px) | Abdeckung klein / groß | Kanten-Bleeding | gefüllte Kantenlöcher | Flackern gefüllt | ms |
|---|---|---|---|---|---|---|---|
| ohne Kantenfilter (Depth Pro 2 S) | 2,48 % | – | – | 20,9 % | 100 % | – | – |
| alter 3×3-Kantenfilter (Depth Pro 2 S) | 0,86 % | – | – | 3,95 % | 68 % | – | – |
| Fusion + Ebenenabstand + Stützprüfung (bis Nov. 2026) | 0,40 % | 3,06 % | 87 / 89 % | 0,00 % | 2 % | 1,49 % | 24 |
| einfach: geglätteter Korrekturfaktor + Kantentest | 0,33–0,35 % | 5,7–7,7 % | 52–67 / 45–60 % | 0,15–0,64 % | 3–24 % | 1,31 % | 4 |
| Fusion + Kantentest 1/Tiefe, Zone 8 px | 0,30 % | 2,07 % | 81 / 82 % | 0,00 % | 19 % | 1,29 % | 22 |
| **Fusion + Kantentest 1/Tiefe, Zone 4 px (Standard)** | 0,36 % | 2,40 % | **85 / 86 %** | 0,04 % | **21 %** | 1,36 % | 22 |

- Zone 4 px statt 8 px: 66 % statt 59 % der echten Sensorlöcher werden gefüllt, dafür bleibt ein
  kleiner Rest Kanten-Bleeding (0,04 %, unter 3 m 0,01 %) und der Fehler steigt leicht.
- Das einfache Verfahren mittelt den Korrekturfaktor über Kanten hinweg und hat in großen
  Löchern keine Führung entlang der Flächen; deshalb bleibt die Fusion.
- Der Kantentest misst den Sprung in 1/Tiefe statt in der Tiefe: Im relativen Tiefensprung galt
  der flach gesehene Boden ab etwa 2 m als Kante, die Abdeckung sank auf 24–73 %.
- Flackern: Das Modell allein ändert sich von Frame zu Frame um 1,4–1,6 %, der Sensor um 0,2 %.
  Keine Variante der Füllung beruhigt das Modell nennenswert; das Flackern kommt vom Modell.
- Stream-Modus von Depth Pro 2 S (`--stream --window W`, 30 Frames pro Aufnahme): Flackern der
  gefüllten Tiefe 1,91 % (Einzelbild) → 0,89 % (Fenster 4) → 0,80 % (Fenster 16), Modell allein
  2,40 → 0,74 → 0,57 %. Laufzeit des Modells auf dem M4 Pro 134 → 184 → 312 ms.
- Einrasten von Zwischenwerten auf die nähere Seite statt Verwerfen wurde getestet und verworfen
  (21 % Bleeding, landet oft auf der falschen Seite).
- Die Fusionsparameter (`down`, `lam`, `sigma_edge`) ändern den Fehler in großen Löchern kaum;
  er wird vom Modell bestimmt.

### Modelle

Gemessen noch mit dem früheren Kantenfilter (Ebenenabstand + Stützprüfung) und nicht neu
gerechnet; mit dem aktuellen Filter für MoGe-2 B siehe oben. Die Laufzeiten enthalten die
damalige Füllung (24 ms), die aktuelle braucht mit echten Vorhersagen ohne Warmstart 22 ms.

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
