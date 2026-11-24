#!/usr/bin/env bash
# Nimmt eine Szene für die Karten-Auswertung als ROS-2-Bag auf (siehe README.md).
#
#   ./record_scene.sh <szenen_name> [zusätzliche Topics ...]
#   z. B. ./record_scene.sh s01_monitor_tisch
#
# Ergebnis: $SCENE_ROOT/<szenen_name>/bag/  (mcap, zstd)  +  scene_log.yaml zum Ausfüllen.
# Vorher: Kamera-Knoten läuft (azure_interface.py), TF map -> camera wird publiziert.
set -euo pipefail

NAME="${1:?Szenen-Name angeben, z. B. s01_monitor_tisch}"
shift || true
SCENE_ROOT="${SCENE_ROOT:-$HOME/map_eval_scenes}"
OUT="$SCENE_ROOT/$NAME"

TOPICS=(
  /camera/color/image_raw
  /camera/depth/image_raw
  /camera/color/camera_info
  /camera/depth/camera_info
  /tf
  /tf_static
)
# optional, falls vorhanden: Maske der absichtlich entfernten Pixel (Mensch/Arm)
if ros2 topic list 2>/dev/null | grep -qx /camera/depth/ignore_mask; then
  TOPICS+=(/camera/depth/ignore_mask)
fi
TOPICS+=("$@")

if [[ -e "$OUT/bag" ]]; then
  echo "FEHLER: $OUT/bag existiert schon. Anderen Namen wählen." >&2
  exit 1
fi
mkdir -p "$OUT"

# Vorab-Check: kommen die Daten an?
echo "== Vorab-Check (je 3 s) =="
for t in /camera/color/image_raw /camera/depth/image_raw /tf; do
  hz=$(timeout 3 ros2 topic hz "$t" 2>/dev/null | grep -m1 "average rate" | awk '{print $3}' || true)
  echo "  $t : ${hz:-KEINE DATEN} Hz"
done

# Protokoll-Vorlage
cat > "$OUT/scene_log.yaml" <<EOF
# Bitte direkt nach der Aufnahme ausfüllen (siehe README.md, Abschnitt "Protokoll").
scene: $NAME
date: $(date -Iseconds)
host: $(hostname)
robot: maya
operator:
description:            # Was ist zu sehen? Raum, Möbel, Material
hard_surfaces: []       # z. B. [schwarzer Monitor, Glastür, Chromstuhlbein]
camera_path:            # azure_interface mit --use_ln oder k4a-Direktpfad?
filters_during_recording: {mask_humans: false, filter_robot: false}
arm_pose:               # z. B. "Transportpose, nicht im Bild"
people_in_view: false
lighting:               # z. B. Tageslicht / Kunstlicht / Gegenlicht
segments:               # Zeiten in Sekunden ab Start der Aufnahme
  eval:  {start: , end: , motion: }      # statisch | Kopf-Schwenk | Fahrt
  ref:   {start: , end: , motion: }      # Nah-Scan
obstacles:              # optional: bekannte Hindernisse für den Kollisionstest
  - {name: , position_map_xyz_m: [ , , ], size_m: [ , , ]}
notes:
EOF

echo
echo "== Aufnahme läuft: $OUT/bag  (Strg+C beendet) =="
echo "   Ablauf: 1) EVAL-Segment  2) 3 s Pause  3) REF-Nah-Scan  (Details in README.md)"
date +%s.%N > "$OUT/record_start_unix.txt"
ros2 bag record -o "$OUT/bag" -s mcap --storage-preset-profile zstd_fast "${TOPICS[@]}"

echo
ros2 bag info "$OUT/bag" | tee "$OUT/bag_info.txt"
echo
echo "Jetzt $OUT/scene_log.yaml ausfüllen (Segment-Zeiten!)."
