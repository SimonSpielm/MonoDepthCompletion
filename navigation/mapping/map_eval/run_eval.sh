#!/usr/bin/env bash
# Startet die Paper-Auswertungen am echten System (rmc-lx0481). Siehe README.md.
#
#   ./run_eval.sh stability                 # Zusatz-Auswertung auf dem vorhandenen Sweep-Cache
#   ./run_eval.sh fill  <szene> [<szene>…]  # Schritt 1: Bag -> 4 Tiefen-Varianten   (moge-Env)
#   ./run_eval.sh maps  <szene> [<szene>…]  # Schritt 2: nvblox-Karten              (nvblox-Env)
#   ./run_eval.sh eval  <szene> [<szene>…]  # Schritt 3: Metriken + Differenzkarten
#   ./run_eval.sh all   <szene> [<szene>…]  # 1 bis 3 nacheinander
#   ./run_eval.sh rviz  <szene>             # Differenzwolken für RViz-Screenshots
#
# <szene> = Name unter $SCENE_ROOT (z. B. s01_monitor_tisch) oder voller Pfad.
# Umgebung anpassen, falls nötig:
set -euo pipefail

MAIN="${MAIN:-$HOME/Documents/main}"
SCENE_ROOT="${SCENE_ROOT:-$HOME/map_eval_scenes}"
PY_MOGE="${PY_MOGE:-$HOME/miniforge3/envs/moge/bin/python}"     # MoGe + torch
PY_NVBLOX="${PY_NVBLOX:-python3}"                                # Env, in dem start_nvblox.py läuft
PY_EVAL="${PY_EVAL:-$PY_MOGE}"                                   # numpy + matplotlib (+ rclpy für rviz)
ROS_SETUP="${ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-48}"
SWEEP_CACHE="${SWEEP_CACHE:-$MAIN/navigation/mapping/moge_sweep_out/preds.npz}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROS_DOMAIN_ID
# ROS-Python sichtbar machen, Conan-Pakete (altes numpy) NICHT übernehmen
# shellcheck disable=SC1090
set +u; source "$ROS_SETUP"; set -u
export PYTHONPATH="/opt/ros/jazzy/lib/python3.12/site-packages:$MAIN"
export LD_LIBRARY_PATH="/opt/ros/jazzy/lib:${LD_LIBRARY_PATH:-}"

scene_path() { [[ "$1" = /* ]] && echo "$1" || echo "$SCENE_ROOT/$1"; }

cmd="${1:-}"; shift || true
case "$cmd" in
  stability)
    # moge_eval_stability.py muss neben moge_sweep.py / moge_sweep_optim.py liegen
    "$PY_MOGE" "$MAIN/navigation/mapping/moge_eval_stability.py" \
      --preds "$SWEEP_CACHE" --out-dir "$MAIN/navigation/mapping/moge_eval_out" --plots
    ;;
  fill)
    for s in "$@"; do "$PY_MOGE" "$HERE/bag_fill.py" --scene "$(scene_path "$s")" --save-pred; done ;;
  maps)
    for s in "$@"; do "$PY_NVBLOX" "$HERE/map_build.py" --scene "$(scene_path "$s")"; done ;;
  eval)
    paths=(); for s in "$@"; do paths+=("$(scene_path "$s")"); done
    "$PY_EVAL" "$HERE/map_eval.py" --scenes "${paths[@]}" --out "$SCENE_ROOT/summary" --static-until 15 ;;
  all)
    "$0" fill "$@"; "$0" maps "$@"; "$0" eval "$@" ;;
  rviz)
    "$PY_EVAL" "$HERE/map_eval.py" --scenes "$(scene_path "$1")" --out "$SCENE_ROOT/summary_rviz" --rviz ;;
  *)
    sed -n '2,13p' "$0"; exit 1 ;;
esac
