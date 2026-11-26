# Mapping-Pipeline

Kamera → Tiefenvervollständigung → gefilterte Punktwolke → Karte → Kollisionsprüfung in Python.
Jeder Schritt ist ein eigener Prozess mit genau einer Aufgabe. Die Schritte kommunizieren
über ROS 2 Topics (Ausnahme: der Kamera-Eingang über LN).

```
 Azure Kinect ──(nur lokal)──► [1] ln_camera_publisher.py
                                      │  LN: camera.perception/{rgb,depth,cam_k}  (+ timestamp)
 echtes System: Kameratreiber ────────┤
                                      ▼
                               [2] ln_ros_camera_bridge.py
                                      │  /camera/color/image_raw, /camera/depth/image_raw,
                                      │  /camera/color/camera_info
                                      ▼
                               [3] depth_completion/depth_completion_node.py (GPU)
                                      │  /camera/completed/{color/image_raw, depth/image_raw, camera_info}
                                      ▼
  synchronization_buffer.py ──► [4] camera_frame_filter.py ◄── user_info (W_T_CAM)
  (Odometrie + q über die Zeit)       │  /tf  (map → Wt → camera, Wt → ground)
                                      │  /camera/filtered/depth/image_raw
                                      │  /camera/filtered/points
                         ┌────────────┴─────────────┐
                         ▼                          ▼
              [5] start_nvblox.py         start_octomap_mapping.py (Alternative)
                 /nvblox/occupied_points_rgb   /octomap_point_cloud_centers
                         └────────────┬─────────────┘
                                      ▼
                               [6] collision_map_provider.py  (Python-Klasse, kein eigener Prozess)
                                      COAL-Octree / AABBs / FROCC-Map für Arm_Collision_Model
```

## Die Schritte

| # | Datei | Aufgabe | Eingang | Ausgang |
|---|---|---|---|---|
| 1 | `ln_camera_publisher.py` | Azure Kinect per k4a auslesen, RGB, registrierte Tiefe und K mit Zeitstempel auf LN publishen. **Nur mit lokal angeschlossener Kamera**, auf dem echten System gibt es den Treiber schon. | k4a | LN `camera.perception/*` |
| 2 | `ln_ros_camera_bridge.py` | LN-Kameradaten 1:1 nach ROS. `read()` blockiert, also kein Timer: jeder Frame wird sofort weitergereicht. | LN | `/camera/*` (roh) |
| 3 | `depth_completion/depth_completion_node.py` | Monokulares Tiefenmodell (wählbar: MoGe-2/3 oder Depth Pro 2, Standard MoGe-2 B mit 600 Tokens, ~9 Hz auf M4 Pro) + Gradienten-Fusion (CG auf der GPU) + Kantenfilter füllt die Löcher der Sensortiefe. Gemessene Pixel bleiben unverändert. Keine Anzeige. | `/camera/depth/image_raw` (Trigger), `/camera/color/*` | `/camera/completed/*` |
| 4 | `camera_frame_filter.py` | TF zum Frame-Zeitstempel, Min/Max-Distanz, fliegende Pixel an Tiefenkanten, Menschen (`--mask_humans`) und Roboterarm (`--filter_robot`) maskieren, gefilterte Tiefe + Punktwolke publishen. `--depth_source raw` nimmt die rohe Tiefe (ohne Schritt 3), `--no-publish_pointcloud` schaltet die Punktwolke ab. | `/camera/completed/*` | `/tf`, `/camera/filtered/*` |
| 5 | `start_nvblox.py` | TSDF-Karte aus der gefilterten Tiefe, belegte Voxel um die Kamera als Punktwolke, isolierte Voxel (Speckles) entfernt. LN-Service `nvblox.reset_map`. | `/camera/filtered/depth/image_raw` | `/nvblox/occupied_points_rgb` |
| 5' | `start_octomap_mapping.py` | Alternative: ROS `octomap_server` auf der gefilterten Punktwolke. | `/camera/filtered/points` | `/octomap_point_cloud_centers` |
| 6 | `collision_map_provider.py` | `CollisionMapProvider`: holt die Karte (nvblox oder octomap, `MAP_BACKEND`) nach Python, baut COAL-Octree / AABBs / FROCC-Map und filtert optional ein Objekt heraus. | Karten-Topic | Python-API |

Alles zu Schritt 3 liegt im Unterordner `depth_completion/`: ROS-Knoten, Modelle
(`depth_models.py`), Fusion und Kantenfilter (`depth_fusion.py`), Live-Viewer (`viewer.py`) und
Offline-Benchmark (`bench/`). Wie das genau funktioniert, mit Zeichnungen:
[depth_completion/README.md](depth_completion/README.md).
Hilfsmodule: `synchronization_buffer.py` (LN-Service: Odometrie und Gelenkwinkel zu einem
Zeitstempel), `arm_filter.py` (Arm-Maske, `depth_to_pointcloud`), `ros_utils.py`
(Zeitstempel-Umrechnung, TF aus 4×4-Matrix).

## Starten

```bash
python3 -m navigation.mapping.synchronization_buffer
```

```bash
python3 -m navigation.mapping.ln_camera_publisher            # nur mit lokaler Kamera
```

```bash
python3 -m navigation.mapping.ln_ros_camera_bridge
```

```bash
python navigation/mapping/depth_completion/depth_completion_node.py   # conda-Env depthcompletion
```

Modell und Tokenzahl stehen oben im Skript (`MODEL`, `NUM_TOKENS`, Namen aus
`depth_models.MODELS`). Im LN-Prozessmanager das Env-Python direkt aufrufen
(`~/miniforge3/envs/depthcompletion/bin/python`), nicht `conda run`. Einrichten des Envs:
[depth_completion/README.md](depth_completion/README.md), Abschnitt „Installation“.

```bash
python3 -m navigation.mapping.camera_frame_filter --mask_humans --filter_robot
```

Optionen von Schritt 4:

- `--publish_pointcloud` / `--no-publish_pointcloud` — Punktwolke auf `/camera/filtered/points`
  publishen oder nicht (Default: aus; nur mit `--publish_pointcloud`). Die gefilterte Tiefe
  für nvblox wird immer publisht.
- `--depth_source completed` — die von MoGe gefüllte Tiefe aus `/camera/completed/depth/image_raw`
  verwenden (Default).
- `--depth_source raw` — die originale Tiefe aus `/camera/depth/image_raw` (ln_ros_camera_bridge)
  verwenden. Schaltet aus Zeitstempel-Gründen auch Farbe und `camera_info` auf die rohen Topics.
- `--raw` — Kurzform für `--depth_source raw`.

Hinweis: `start_nvblox` liest `camera_info`/Farbe standardmäßig von `/camera/completed/*`.
Ohne laufendes MoGe müssen `camera_info_topic`/`rgb_topic` auf `/camera/color/*` gesetzt werden.

```bash
python3 -m navigation.mapping.start_nvblox                   # oder: ros2 launch navigation/mapping/start_octomap_mapping.py
```

Schritt 6 wird nicht gestartet, sondern von `Arm_Collision_Model` instanziiert.

## Dev-Viewer

`depth_completion/viewer.py` zeigt in einem Rerun-Fenster oben RGB, Sensortiefe, Modelltiefe und
gemergte Tiefe, darunter die 3D-Punktwolke jeder dieser Tiefen (mit der Maus drehbar). Er rechnet
genau den Code von Schritt 3 (`depth_models` + `depth_fusion.fill`). Nicht Teil der
Produktivkette: Er abonniert die rohen Bridge-Topics, publisht nichts und kann parallel zur
Pipeline laufen (eigenes Modell im GPU-Speicher). Mit `--frames f.npz` spielt er stattdessen eine
Aufnahme ohne ROS ab, mit `--eval` zeigt er zusätzlich den Fehler unter 3 m als Kurve über die Zeit.

```bash
python navigation/mapping/depth_completion/viewer.py --model moge2-b --num-tokens 600
```

```bash
python navigation/mapping/depth_completion/viewer.py --model depthpro2-s --stream
```

```bash
python navigation/mapping/depth_completion/viewer.py --frames aufnahme.npz
```

```bash
python navigation/mapping/depth_completion/viewer.py --set ramp_dmin=0 support_px=0
```

Der letzte Aufruf schaltet die Kantenfilter zum Vergleich ab. Es gibt nur einen Knoten und einen
Viewer für alle Modelle; das Modell wird per Name gewählt. Die früheren Einzel-Skripte je Modell
(`moge_depth.py`, `moge_depth_optim.py`, `moge3_depth_viewer.py`, `depthpro_depth_optim.py`,
`fill_depth.py`) hatten je eine eigene Kopie der Fusion und sind darin aufgegangen.

## Designentscheidungen

**ROS 2 als Datenbus zwischen den Schritten.** ROS hat native Datentypen für alles, was hier
fließt (`Image`, `CameraInfo`, `PointCloud2`, TF, Octomap), und jeder Zwischenschritt lässt sich
direkt in RViz anschauen (rohe Tiefe neben gefüllter Tiefe, gefilterte Punktwolke, Karte).
nvblox, `octomap_server` und `message_filters` arbeiten direkt mit diesen Typen. LN bleibt dort,
wo es schon ist: Kameratreiber, Synchronisationspuffer, `user_info` und Reset-Services.

**Ein Prozess pro Aufgabe.** Das alte `azure_interface.py` hat Kamera, Bridge, Filter, TF und
Punktwolke in einer Schleife gemacht. Jetzt ist jeder Schritt einzeln startbar, austauschbar
(z. B. Schritt 3 weglassen mit `--depth_source raw`, nvblox gegen octomap tauschen) und in seiner eigenen
Umgebung lauffähig. Schritt 3 läuft im conda-Env `depthcompletion` mit torch, der Rest im System-Python mit LN und
`cv_bridge`.

**Lokal und echtes System haben denselben Eingang.** Schritt 1 publisht exakt die LN-Topics
des echten Kameratreibers. Deshalb gibt es in der Bridge nur noch den LN-Pfad (früher `--use_ln`),
und alles ab Schritt 2 ist in beiden Setups identisch.

**Der originale Zeitstempel wird durchgereicht.** Der Aufnahme-Zeitstempel aus Schritt 1
(`time.time()`-Domäne, dieselbe Uhr wie im `SynchronizationBuffer`) wird in Schritt 2 zu
`header.stamp` und bleibt bis zur Karte unverändert: MoGe übernimmt den Header des Eingangs,
der Camera-Frame-Filter stempelt TF, Tiefe und Punktwolke damit. So findet Schritt 4 Odometrie
und Gelenkwinkel *dieses* Frames im Puffer (nötig für die Armmaske, gerade wenn MoGe Latenz
dazubringt), und nvblox/octomap finden über TF genau die Kamerapose dieses Frames. Vorher
wurde mit der Publish-Zeit gestempelt.

**Keine Kometenschweife an Kanten.** Monokulare Modelle zeichnen Objektkanten als weiche Rampe,
und genau dort misst der ToF-Sensor oft nichts. Ohne Gegenmaßnahme landet die Rampe als Schweif
zwischen Vorder- und Hintergrund in der Karte. `depth_fusion.fill` verwirft deshalb gefüllte Pixel
(gemessene bleiben immer unverändert; fliegende Pixel in der Messung entfernt Schritt 4):

- *Rampenfilter* (`ramp_dmin = 0.5 m`): Für eine Ebene n·X = d gilt f·|∇(1/z)| ≤ 1/d. Geht die
  lokale Tangentialebene der Füllung näher als 0,5 m am Kamerazentrum vorbei, ist sie (fast)
  parallel zum Sehstrahl, also eine Rampe. Echte Flächen bleiben, z. B. der Boden mit
  d = Kamerahöhe (≈ 0,58 m) in jeder Entfernung. **`ramp_dmin` muss unter der Kamerahöhe liegen.**
- *Stützprüfung* (`support_px = 12`, `support_rtol = 2 %`, `support_edge = 10 %`): Ein Füllwert
  muss im Bereich der Messungen im Umkreis von 12 px liegen. Liegen dort zwei Tiefenebenen
  (Spanne > 10 %), muss er zu einer der beiden passen; Zwischenwerte werden verworfen.

Dazu wird das Korrekturfeld der Fusion kantentreu direkt auf volle Auflösung gebracht. Auf 6
Aufnahmen (`depth_completion/bench/README.md`) sinkt das Kanten-Bleeding damit von 3,9 % (alter 3×3-Kantenfilter)
auf 0,00–0,01 %, der Fehler unter 3 m in kleinen Löchern von 0,9 % auf 0,4 % (6 mm). Preis: Nur
noch 2–8 % der Kantenlöcher werden gefüllt; dort bleibt die Tiefe leer statt falsch.

**Exakter Abgleich statt Approximation.** Farbe und Tiefe eines Frames tragen denselben
Stempel, daher reicht ein exakter Abgleich über den Zeitstempel (kein `ApproximateTimeSynchronizer`).
`camera_info` ist statisch und wird nicht mitsynchronisiert.

**Neuester Frame statt Warteschlange in MoGe.** MoGe ist langsamer als die Kamera (15 Hz).
Der Knoten verarbeitet immer nur den neuesten Frame und verwirft ältere, sonst würde die
Latenz unbegrenzt wachsen. Die Tiefe ist der einzige Trigger, es gibt keinen Synchronizer:
Das RGB-Bild (MoGe braucht es als Eingang) wird aus einem kleinen Puffer über den identischen
Zeitstempel geholt. Ein verlorenes RGB-Bild kostet so nur diesen einen Frame. Das verwendete
RGB-Bild und `camera_info` werden mit der gefüllten Tiefe unter demselben Stempel neu publisht.
Die gefüllte Tiefe kommt ~100 ms nach dem Rohbild; so müssen nachfolgende Knoten (Menschenmaske,
nvblox-Einfärbung) nicht selbst über diese Latenz hinweg RGB puffern und zuordnen.

**Filtern nach dem Füllen.** Menschen und Arm werden in Schritt 4 maskiert, also *nach* MoGe.
MoGe sieht dadurch das ganze Bild (bessere Skalierung), und was in Arm- oder Menschbereiche
gefüllt wurde, wird danach wieder auf 0 (= ignorieren) gesetzt. Min/Max-Distanz greift
ebenfalls erst hier, damit auch gefüllte Werte begrenzt werden.

**nvblox integriert die gefilterte Tiefe, nicht die Punktwolke.** `nvblox_torch` integriert
Tiefenbilder projektiv (`add_depth_frame`). Schritt 4 publisht deshalb dieselben gefilterten
Daten zweimal: als Tiefenbild für nvblox und als `PointCloud2` für octomap, RViz und den
schnellen Pfad von `CollisionMapProvider` (`use_pointcloud_fast`).

**TF kommt aus Schritt 4.** Der Knoten, der ohnehin den Synchronisationspuffer abfragt, stempelt
`map → Wt` mit der Odometrie zum Frame-Zeitstempel und publisht sie vor Tiefe und Punktwolke.

**Fusion nur auf der GPU (CUDA oder Apple MPS).** Die CPU-Variante der Gradienten-Fusion ist
entfallen. Die Fusions-Mathematik stammt aus dem früheren `moge_depth_optim.py`; die Parameter
(`down=2`, `sigma_edge=0.05`, 40 CG-Iterationen) sind mit `depth_completion/bench/` auf 6
Aufnahmen bestimmt.

**Schwebende Voxel an zwei Stellen filtern, ohne die Rate zu drücken.** Die meisten
schwebenden Voxel stammen von fliegenden Pixeln an Tiefenkanten (Mischwerte aus Vorder- und
Hintergrund). Schritt 4 verwirft deshalb Pixel, deren Tiefenspanne im 3×3-Fenster mehr als
`EDGE_REL_THRESH` (10 %) der eigenen Tiefe beträgt. Das sind zwei Morphologie-Operationen,
ca. 3 ms pro Frame auf der CPU, und streifende Flächen wie der Boden bleiben erhalten. Was
danach noch übrig ist, entfernt Schritt 5 im Abfragefenster: belegte Voxel mit weniger als
`min_neighbors` (Standard 1) belegten Nachbarn in der 26er-Nachbarschaft. Das Zählen läuft als
3×3×3-Pooling auf der GPU und kostet deutlich weniger als die eigentliche Abfrage. Die Karte
selbst bleibt unverändert; der Filter wirkt nur auf die publizierte Punktwolke.

## Topics

| Topic | Typ | Von | Nach |
|---|---|---|---|
| `/camera/color/image_raw` | `Image` rgb8 | 2 | 3, (4 mit `--depth_source raw`) |
| `/camera/depth/image_raw` | `Image` 16UC1 mm | 2 | 3, (4 mit `--depth_source raw`) |
| `/camera/color/camera_info`, `/camera/depth/camera_info` | `CameraInfo` | 2 | 3, (4 mit `--depth_source raw`) |
| `/camera/completed/color/image_raw` | `Image` rgb8 | 3 | 4, 5 (Einfärbung) |
| `/camera/completed/depth/image_raw` | `Image` 16UC1 mm | 3 | 4 |
| `/camera/completed/camera_info` | `CameraInfo` | 3 | 4, 5 |
| `/tf` | `map → Wt → camera`, `Wt → ground` | 4 | 5, 5', RViz |
| `/camera/filtered/depth/image_raw` | `Image` 16UC1 mm | 4 | 5 |
| `/camera/filtered/points` | `PointCloud2` (Frame `camera`) | 4 | 5', RViz |
| `/nvblox/occupied_points_rgb` | `PointCloud2` (Frame `map`) | 5 | 6 |
| `/octomap_point_cloud_centers` | `PointCloud2` (Frame `map`) | 5' | 6 |
