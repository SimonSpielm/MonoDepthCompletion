# Karten-Auswertung: Raw vs. Global vs. Variante A vs. Variante B

Ziel: zeigen, wie sich die nvblox-Karte mit den verschiedenen Tiefenfüllungen aufbaut, und zwar
über **mehrere Szenen**: Vollständigkeit, Phantome, übersehene Hindernisse, Stabilität über die Zeit
und Differenzkarten.

```
record_scene.sh  ->  bag_fill.py  ->  map_build.py  ->  map_eval.py
  (Aufnahme)         (MoGe + 4 Varianten)  (nvblox je Variante + Referenz)  (Metriken, Plots, Diffs)
```

Alles läuft **offline auf denselben Bags**. Jede Variante sieht also exakt dieselben Bilder und Posen.
Die Füll-Methoden liegen eingefroren in `fill_methods.py` (Stand main). Die Skripte laufen deshalb auf
**main und refactor**, obwohl refactor `moge_depth.py` (Variante A) entfernt hat.

---

## 1. Was geloggt werden muss

### 1.1 ROS-Topics im Bag (Pflicht)

| Topic | Typ | Rate | Wofür | Prüfen |
|---|---|---|---|---|
| `/camera/color/image_raw` | `sensor_msgs/Image` rgb8 1280×720 | 15 Hz | MoGe-Eingabe | `ros2 topic hz` ≈ 15 |
| `/camera/depth/image_raw` | `sensor_msgs/Image` 16UC1 [mm], **auf Farbe registriert** | 15 Hz | Messung | gleiche Auflösung wie Farbe |
| `/camera/color/camera_info` | `sensor_msgs/CameraInfo` | 15 Hz | K, FoV für MoGe und nvblox | `k[0]` plausibel (~600 bei 720p) |
| `/camera/depth/camera_info` | `sensor_msgs/CameraInfo` | 15 Hz | Kontrolle | identisch zu color |
| `/tf` | `tf2_msgs/TFMessage` | 15 Hz | Pose `map → Wt → camera` je Bild | `ros2 run tf2_ros tf2_echo map camera` |
| `/tf_static` | `tf2_msgs/TFMessage` | latched | statische Frames, falls vorhanden | – |

**Wichtig:** RGB, Tiefe und TF müssen **denselben `header.stamp`** tragen. `azure_interface.py` macht
das bereits (`odom_ros_time`). `bag_fill.py` paart RGB und Tiefe nur über **exakt gleiche** Stempel.

### 1.2 Optional: Maske der absichtlich entfernten Pixel

Die Rohtiefe im Bag muss **ungefiltert** sein. Gefüllt wird vorher, maskiert danach, wie im Betrieb.
Damit `bag_fill.py` die Masken (Mensch, Arm, Range) kennt, braucht es eines der beiden Topics:

| Topic | Branch | Wofür |
|---|---|---|
| `/camera/filtered/depth/image_raw` (16UC1) | **refactor** (`pointcloud_publisher`) | Maske = in der Rohtiefe gemessen, gefiltert 0. Wird automatisch erkannt. |
| `/camera/depth/ignore_mask` (mono8) | main (falls nachgerüstet) | Maske direkt |

Gibt es keins von beiden: Filter aus, Arm außerhalb des Bildes, keine Personen im Bild.
Auch mit der abgeleiteten Maske gilt: Mensch- oder Arm-Pixel, die schon roh ein Loch waren, sind
darin nicht enthalten. Am saubersten ist deshalb immer eine Aufnahme **ohne Personen und Arm im Bild**.

### 1.3 Was bei der Aufnahme laufen muss

**Branch refactor** (neue Pipeline, `navigation/mapping/README.md`):

```bash
python3 -m navigation.mapping.synchronization_buffer
python3 -m navigation.mapping.ln_ros_camera_bridge
python3 -m navigation.mapping.pointcloud_publisher --raw            # liefert /tf zum Frame-Stempel
# optional mit --mask_humans --filter_robot -> /camera/filtered/... enthält dann die Masken
```

`moge_depth_completion.py` und nvblox müssen für die Aufnahme **nicht** laufen. Gefüllt und
kartiert wird offline. Aufgezeichnet wird die Rohtiefe `/camera/depth/image_raw` aus der Bridge.

**Branch main:**

- `azure_interface.py` **ohne** `--mask_humans` und `--filter_robot`
- Pfad notieren: `--use_ln` oder k4a direkt. Beim k4a-Pfad ist die Tiefe sicher registriert.
- `synchronization_buffer` läuft, sonst fehlt `map → Wt`

Für beide gilt: `MAX_DIST = 8000` (Standard) beibehalten, `NOINFO_BEHAVIOUR = 0`.

### 1.4 Speicherbedarf

≈ 70 MB/s unkomprimiert, mit `zstd_fast` etwa 35–45 MB/s. Eine Szene mit 2 min ≈ 5 GB.
Für Zwischenergebnisse kommen pro Variante ≈ 1.8 MB/Frame dazu. **Freien Platz vorher prüfen.**

---

## 2. Ablauf einer Aufnahme (pro Szene)

```bash
export SCENE_ROOT=~/map_eval_scenes
./record_scene.sh s01_monitor_tisch
```

Das Skript prüft vorab die Raten, nimmt nach `$SCENE_ROOT/<name>/bag` auf (mcap + zstd) und legt
`scene_log.yaml` und `record_start_unix.txt` an.

Eine Aufnahme hat **zwei Segmente**, in dieser Reihenfolge und ohne Neustart (gleiche Odometrie!):

| Segment | Dauer | Bewegung | Zweck |
|---|---|---|---|
| **EVAL** | 30–60 s | (a) erst 15 s **statisch** in typischer Arbeitspose, dann (b) langsamer Kopf-Schwenk oder Fahrt (≤ 0.1 m/s, ≤ 15 °/s) | Karte, die verglichen wird. (a) für Flip-Rate, (b) für Kartenaufbau |
| Pause | 3 s | – | Trennung, im Log sichtbar |
| **REF** | 60–120 s | **Nah-Scan:** jede relevante Fläche aus 0.5–2 m, mehrere Blickwinkel | Referenzkarte aus roher Tiefe |

Hinweise:
- Im REF-Segment schwierige Flächen (schwarz, glänzend) **schräg und nah** anfahren. Dort misst die
  Kinect noch am ehesten.
- **Nichts in der Szene verändern** zwischen EVAL und REF (keine Personen, keine Objekte bewegen).
- Wenn MAYA nur den Kopf oder Arm bewegt und nicht fährt: unkritisch für die Odometrie. Bei Fahrt
  die Odometrie-Drift im Blick behalten (Kontrolle: Referenzkarte auf Doppelwände prüfen).

### 2.1 Szenen-Auswahl (Vorschlag, 4–6 Szenen)

| ID | Szene | Warum |
|---|---|---|
| s01 | Schreibtisch mit **schwarzem Monitor**, Tastatur, Kabeln | große, dunkle Löcher, Arbeitsraum |
| s02 | **Glastür** oder Fenster mit Rahmen | transparente Flächen |
| s03 | **Stühle mit dünnen / glänzenden Beinen**, Tischkanten | dünne Strukturen, Kantenpixel |
| s04 | **Flur / großer Raum** (Tiefe > 5 m) | Fernbereich, einseitige Löcher |
| s05 | **Küche / Ablage** mit Objekten zum Greifen | typisches Manipulations-Szenario |
| s06 | (optional) **bekannte Hindernisse**: schwarze Box, Plexiglasplatte, an notierter Position | Kollisionstest |

### 2.2 Protokoll `scene_log.yaml` (direkt nach der Aufnahme ausfüllen)

| Feld | Inhalt | Pflicht |
|---|---|---|
| `description` | Raum, Möbel, Materialien | ja |
| `hard_surfaces` | Liste schwieriger Flächen | ja |
| `camera_path` | `ln` oder `k4a` | ja |
| `filters_during_recording` | Mensch-/Arm-Filter an oder aus | ja |
| `arm_pose`, `people_in_view` | Arm im Bild? Personen? | ja |
| `lighting` | Tageslicht/Kunstlicht/Gegenlicht | ja |
| `segments.eval.start/end`, `segments.ref.start/end` | **Sekunden ab Aufnahmestart** (Uhr/Stoppuhr mitlaufen lassen) | **ja** |
| `segments.*.motion` | statisch / Schwenk / Fahrt | ja |
| `obstacles` | Name, Position im map-Frame, Größe | nur s06 |
| `notes` | Auffälligkeiten (Drift, Ruckler, Leute durchs Bild) | wenn nötig |

Ohne Segment-Zeiten lassen sich EVAL und REF nicht trennen. Notfalls `map_build.py --eval 0:45 --ref 50:160`.

### 2.3 Zusätzlich festhalten

- **Handyfoto** der Szene aus Kamerasicht (für die Paper-Abbildungen, `<scene>/photo.jpg`)
- **RViz-Screenshots** der nvblox-Karte (`/nvblox/occupied_points_rgb`) live während EVAL, mit
  roher Tiefe. Für die Füll-Varianten kommen die Screenshots aus Schritt 3 (`--rviz`), damit alle
  denselben Blickwinkel haben.
- Nach der Aufnahme: `bag_info.txt` prüfen (Anzahl Nachrichten RGB ≈ Tiefe ≈ Dauer × 15).

---

## 3. Auswertung

### Kurzweg mit `run_eval.sh`

Pfade und Envs oben in `run_eval.sh` prüfen (`MAIN`, `SCENE_ROOT`, `PY_MOGE`, `PY_NVBLOX`), dann:

```bash
./run_eval.sh stability                              # Pixel-Stabilität auf dem vorhandenen Sweep-Cache
./run_eval.sh all s01_monitor_tisch s02_glastuer ...  # Füllen -> Karten -> Metriken
./run_eval.sh rviz s01_monitor_tisch                  # Differenzwolken für RViz-Screenshots
```

`stability` erwartet `moge_eval_stability.py` in `navigation/mapping/`, also neben `moge_sweep.py`
und `moge_sweep_optim.py`. Im Repo liegt es unter `moge_sweep_ergebnisse/`.

### Einzelschritte

```bash
# Schritt 1, im conda-Env "moge" (ROS gesourct): MoGe + 4 Varianten
python bag_fill.py --scene $SCENE_ROOT/s01_monitor_tisch          # --save-pred für spätere Neu-Füllungen
#   --b-impl deployed: B direkt aus moge_depth_completion.py (refactor, GPU) statt aus fill_methods.py

# Schritt 2, im Env mit nvblox_torch (wie start_nvblox.py): Karten je Variante + Referenz
python map_build.py --scene $SCENE_ROOT/s01_monitor_tisch

# Schritt 3, beliebiges Env mit numpy + matplotlib: alle Szenen zusammen
python map_eval.py --scenes $SCENE_ROOT/s0* --out $SCENE_ROOT/summary --static-until 15
python map_eval.py --scenes $SCENE_ROOT/s01_monitor_tisch --rviz   # Differenzwolken in RViz
```

### Was rauskommt

| Datei | Inhalt |
|---|---|
| `<scene>/fill/fill_log.txt` | Laufzeiten je Variante, Skalierungsfaktoren, übersprungene Frames |
| `<scene>/maps/map_*.npz` | belegt/bekannt je Schnappschuss (1 s) |
| `<scene>/eval/diff_<var>_vs_ref.png/.ply` | Differenzkarte: grau beide, rot Phantom, cyan Hindernis unbekannt, blau Hindernis als frei |
| `<scene>/eval/diff_A_vs_B.png/.ply` | grau beide, orange nur A, blau nur B |
| `<scene>/eval/map_over_time.png` | Vollständigkeit und Voxel-Flips über die Zeit |
| `summary/summary.md` | alle Metriken, Mittelwert ± Std über Szenen, und je Szene |
| `summary/map_table.tex` | fertige Tabelle fürs Paper |

### Differenzbilder in RViz

`map_eval.py --rviz` veröffentlicht `/map_eval/diff_<var>_vs_ref` und `/map_eval/diff_A_vs_B` als
`PointCloud2` im Frame `map` (latched). In RViz: PointCloud2 hinzufügen, *Color Transformer = RGB8*,
*Style = Boxes*, *Size = 0.02*. Gleiche Kamera-Ansicht für alle Varianten speichern
(RViz *Views → Save*), dann sind die Screenshots direkt vergleichbar. Die `.ply` lassen sich auch in
CloudCompare oder MeshLab öffnen.

### Metriken (Kurzfassung, Details im Kopf von `map_eval.py`)

- **completeness / accuracy / fscore @k:** Abdeckung der Referenz-Hindernisse bzw. Anteil richtiger
  Kartenvoxel, Toleranz k = 2 Voxel (4 cm)
- **phantom:** belegt, wo die Referenz freien Raum gesehen hat
- **missed_unknown:** Referenz-Hindernis ist in der Karte unbekannt. Im Betrieb gilt das als frei,
  ist also gefährlich.
- **flip_rate:** wie oft Voxel zwischen belegt und frei wechseln (statischer Teil, `--static-until`)
- **t90:** Sekunden bis 90 % der Endvollständigkeit

### Grenzen der Referenz

Die Referenz kommt aus **roher** Tiefe. Flächen, die die Kinect auch aus der Nähe nie misst
(z. B. Glas), fehlen auch in der Referenz. Dort kann das Füllen weder belohnt (completeness) noch
bestraft (phantom) werden. Solche Fälle über die Differenzbilder qualitativ zeigen oder mit `obstacles`
(s06) gezielt prüfen.
