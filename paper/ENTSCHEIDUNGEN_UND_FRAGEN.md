# Paper-Draft v0.2: Entscheidungen und offene Fragen

Datei: `paper/main.tex` + `paper/refs.bib` (IEEEtran, conference). Kompilieren mit
`cd paper && tectonic main.tex` (ergibt `main.pdf`, ca. 9 Seiten).
v0.2 enthält die Entwicklungsgeschichte aus dem claude.ai-Chat "RGB-to-Depth-Modelle zur Tiefenauffüllung".

## Getroffene Entscheidungen

| # | Entscheidung | Begründung |
|---|---|---|
| E1 | **Sprache Englisch** | IEEE-Venues sind englisch. |
| E2 | **LaTeX/IEEEtran conference** (2-spaltig) statt Word/Doc | Echter IEEE-Stil, Overleaf-fähig. |
| E3 | **Titel:** "Metric Monocular Priors for Depth Completion in Volumetric Environment Modelling of an Assistive Mobile Manipulator" | Stellt den Depth Filler ins Zentrum, Pipeline als Kontext. |
| E4 | **Autoren:** Simon Spielmann (DLR-RM, Wessling), Claude (Anthropic) | Affiliation aus `rmc-github.robotic.dlr.de` abgeleitet. |
| E5 | **Roboter = MAYA** (rollstuhlbasiert, 7-DoF-Arm). EDAN wird als unterstützter Vorgänger zitiert. | Im Chat steht `ROBOT_NAME=maya` in der Prozess-Umgebung. v0.1 hatte fälschlich EDAN. |
| E6 | **Kernthese:** Variante B (Optimierung) wird in der Pipeline genutzt, obwohl Variante A im Holdout besser ist (2.75 % vs. 3.27 %) | Deine Aussage, dass A ohne Messungen in der Umgebung springt. Analytisch begründet: A fällt nach ~r px auf den globalen Faktor zurück (Naht), ε=1e-3 + Clip [0.25,4] sind schlecht konditioniert, die Konfidenzrampe schaltet hart. B ist strikt konvex, also stetig, und klingt mit ℓ=√(λ/μ) ≈ 100 Zellen ≈ 400 px ab. |
| E7 | **Holdout-Bias** als eigene Diskussion: Holdout-Blöcke sind ≤64 px und rundum gemessen, das bevorzugt lokale Verfahren. Indiz: Abstand A↔B schrumpft von 0.9 auf 0.25 pp (16 → 64 px). | Macht den scheinbaren Widerspruch "A besser, aber B genutzt" wissenschaftlich sauber. |
| E8 | **Reihenfolge:** Camera → MoGe → Arm → Human → nvblox. Erst füllen, dann die Masken wieder auf 0 setzen, sonst füllt MoGe Mensch und Arm wieder ein. | Entspricht der im Chat besprochenen Zielarchitektur: eigener Füll-Knoten, `ignore_mask` als Topic, Masken nach der Fusion. Im aktuellen Code ist das noch nicht so verdrahtet. |
| E9 | **"Unknown = free"** in der nvblox-Abfrage als Hauptmotivation für Depth Completion | Steht so im Code (`start_nvblox.py`, `w == 0` → frei). |
| E10 | Zahlen 1:1 aus `moge_sweep_ergebnisse.md` / `moge_optim_sweep_ergebnisse.md` / CSVs. RMSE B = 0.27 m und Distanz-Bins B aus `moge_sweep_optim_out_final`. Bin 4–6 m ist leer (†). | Keine erfundenen Zahlen. Wo etwas fehlt, steht ein rotes `\todo{}`. |
| E11 | Zeiten in Tab. II als "CPU sweep harness" deklariert (9 ms / 85 ms). | Der Sweep nutzt laut Docstring die CPU-Klasse `GradientFusion`. GPU-Timing ist unbekannt (F6). |
| E12 | Abbildungen: Architektur als **echtes TikZ-Diagramm**, die übrigen 3 als beschriftete Platzhalter (qualitativ, zeitliche Stabilität, nvblox-Map) | Platzhaltertext beschreibt jeweils, was zu sehen sein soll. |
| E13 | Neuer Abschnitt **"Design Rationale and Evolution"**: Marigold-DC (zu langsam, s/Frame), PointDiT (DINOv3-Gewichte zugangsbeschränkt, nur anonym umschrieben), PromptDA-S (Nearest-Neighbour-Vorfüllung macht die Ferne "nah"), MoGe-2 (MIT, metrisch, s≈1.02). Stufe 2: Gauß-Glättung → lokal affin (live 13.2 → 8.7 %) → Optimierung (live 4.6 %, 5.9 cm, untuned). | Aus dem Chat. Live-Zahlen sind als grob markiert (andere Szene). |
| E14 | Plane-Fitter (Boden) aus der README **nicht** aufgenommen | Kein Code dazu im Repo, wirkt wie ein Vorgänger. |
| E15 | Downstream: nvblox-Punkte → 1-cm-Coal-Octree → Kollisionsprüfung für den Arm | Aus `coal_octree_getter.py`. |
| E16 | Danksagung nennt Claude als Werkzeug | Transparenz, zusätzlich zur Co-Autorschaft. Bitte prüfen, ob die Venue KI-Autoren erlaubt (viele verbieten das). |
| E17 | bib-Einträge aus dem Gedächtnis geschrieben | Vor der Einreichung gegen DOI/Publisher prüfen. |
| E18 | Neuer Abschnitt **Runtime** + Tab. `tab:runtime`: MoGe 93–101 → 40 ms (FP16, 1200 Tokens), Fusion CPU 30 ms (f=8) / 80 ms (f=4), 14.2 bzw. 8.2 Hz. GPU-Zeit ist `\todo`. | Zahlen aus deinen Konsolen-Prints im Chat. OMP/MKL=1 Thread durch den Prozessmanager. |
| E19 | Implementierung: Füller als **eigener ROS-2-Knoten** (getrennte Python-Umgebungen, Kamera-Timer nicht blockieren, abschaltbar) | Aus dem Chat. |
| E20 | Neue Limitations: Farbbild nicht entzerrt (MoGe nimmt Lochkamera an) und fliegende Pixel an Kanten | Aus dem Chat. |

## Offene Fragen

- **F1 Chatverlauf:** ERLEDIGT (eingearbeitet, siehe E5, E13, E18–E20).
- **F2 Hauptvariante:** Ist B wirklich die Variante, die produktiv in der Pipeline läuft? Oder soll
  das Paper beide gleichwertig präsentieren?
- **F3 Reihenfolge:** Laut Chat ist die Zielarchitektur: Masken im Kamera-Knoten, Fusion im eigenen
  Knoten, Masken danach wieder auf 0. Ist das inzwischen so umgesetzt (`ignore_mask`-Topic)?
  Und läuft im Betrieb der k4a- oder der ln-Pfad? Für ln ist offen, ob die Tiefe registriert ist.
- **F4 Venue / Seitenlimit:** Konferenz (IROS/ICRA/RA-L, 6–8 Seiten) oder Journal? Der Draft ist
  eher lang (~8–9 Seiten).
- **F5 Autoren / Affiliation:** Stimmt "DLR-RM, Wessling"? Weitere Ko-Autoren (Betreuer)?
  Claude als Autor oder nur in der Danksagung?
- **F6 Fehlende Zahlen:** Nur noch die **GPU-Fusionszeit** und die Gesamt-Hz mit GPU-Fusion
  (Prints aus `moge_depth_fill.py` mit `USE_GPU_FUSION = True`).
- **F13 Roboter:** Stimmt MAYA, und ist MAYA rollstuhlbasiert? Gibt es eine zitierfähige Quelle?
- **F7 Szene:** Was war in der 60-Frame-Aufnahme zu sehen, und stand der Roboter still?
- **F8 Zusatzexperimente:** ERLEDIGT: `moge_sweep_ergebnisse/moge_eval_stability.py` (große + einseitige Löcher, Fehler über Abstand, zeitliche Stabilität). Muss noch auf dem Robot-Cache laufen. Tab. `tab:stability` und Figs. sind vorbereitet.
- **F9 Gewichtung gefüllter Pixel:** Werden gefüllte Pixel in nvblox wirklich gleich gewichtet
  (ja laut Code)? Als Limitation drin.
- **F10 Human-Filter:** Ist der Rollstuhlnutzer selbst im Bild? Gibt es eine separate
  reaktive Sicherheitsschicht für Menschen (im Paper so behauptet)?
- **F11 Arm-Filter:** Default 3-D-Modus? Die Faktoren 2 / 7 und die Hand-Schwelle 1e-4 m³ habe ich
  so übernommen.
- **F12 Bilder:** Kannst du Screenshots aus den Viewern liefern (RGB | roh | gefüllt) und
  RViz-Screenshots der nvblox-Map, mit und ohne Filling?

## Nachtrag v0.3: Karten-Auswertung über mehrere Szenen

- **E21** Neuer Abschnitt *Map-Level Evaluation Across Scenes* mit Tab. `tab:maps` und Fig.
  `fig:mapdiff` (Platzhalter). Pipeline: `navigation/mapping/map_eval/` (README mit Aufnahmeprotokoll).
- **E22** Referenz = nvblox-Karte aus roher Tiefe eines Nah-Scans (REF-Segment im selben Bag, gleiche
  Odometrie). Grenze: Glas o. ä. fehlt auch in der Referenz, deshalb qualitativ über Differenzbilder.
- **E23** Metriken: completeness/accuracy/F@2 Voxel, phantom, missed_unknown (unbekannt = frei!),
  flip_rate im statischen Teil, t90.
- **F14** Gibt es inzwischen das `ignore_mask`-Topic? Sonst bei den Aufnahmen Filter aus, Arm aus dem
  Bild, keine Personen.
- **F15** Läuft `nvblox_torch` im selben Env wie MoGe? Die Skripte sind getrennt (`bag_fill.py` im
  moge-Env, `map_build.py` im nvblox-Env), funktionieren aber auch im selben.
- **E24** Branch `refactor` entfernt `moge_depth.py` (Variante A) und die CPU-Fusion. Die Auswertung
  nutzt deshalb eine eingefrorene Kopie aller Füll-Methoden (`map_eval/fill_methods.py`, Stand main
  5146946) und läuft auf main und refactor. `--b-impl deployed` nimmt B direkt aus
  `moge_depth_completion.py`. Auf refactor wird die Maske aus `/camera/filtered/depth/image_raw`
  abgeleitet (kein eigenes `ignore_mask`-Topic nötig). Die im Paper beschriebene Reihenfolge
  (MoGe vor Arm/Human-Filter) entspricht der refactor-Pipeline.
