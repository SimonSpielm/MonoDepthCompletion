# conda-Env `moge` von Grund auf

Für `moge_depth_completion.py`, `moge3_depth_viewer.py`, die Sweeps und `map_eval/bag_fill.py`.
Der Ablauf fasst die bisherige Einrichtung auf rmc-lx0481 zusammen (RTX A6000, Treiber mit
CUDA 13.2, ROS 2 Jazzy). In genau dieser Reihenfolge wurde er noch nicht von Grund auf durchgespielt,
und der MoGe-3-Teil ist am System noch nicht erprobt.

Die Schritte berücksichtigen die Stolpersteine aus der bisherigen Einrichtung:
- `/run/user/...` ist zu klein für pip-Builds
- pip zieht mehrere torch-Versionen
- MoGe installiert `opencv-python-headless`, das keine Fenster öffnen kann
- das numpy 1.26 aus dem Conan-`PYTHONPATH` überschreibt das Env

## 0. Voraussetzungen

- Miniforge/conda installiert (`~/miniforge3`)
- `nvidia-smi` zeigt die GPU. Die CUDA-Version oben rechts muss ≥ 12.8 sein.
- ROS 2 Jazzy unter `/opt/ros/jazzy`. Jazzy nutzt **Python 3.12**, das Env muss dazu passen.

## 1. Bauordner auf die Festplatte legen

pip baut sonst in `/run/user/<uid>/.cache` (RAM-Disk), und das endet in `No space left on device`.

```bash
mkdir -p ~/pip_tmp/cache
export TMPDIR=~/pip_tmp PIP_CACHE_DIR=~/pip_tmp/cache
export HF_HOME=~/.cache/huggingface        # Modelle (1.3–5 GB) auf die Festplatte
```

Die Variablen gelten nur für die aktuelle Shell. Alle folgenden Schritte in **derselben** Shell
ausführen.

## 2. Env anlegen

Falls ein altes Env existiert, vorher entfernen: `conda deactivate && conda env remove -n moge`

```bash
conda create -n moge python=3.12 -y
conda activate moge
```

## 3. PyTorch mit CUDA installieren und festnageln

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # -> ... True
pip freeze | grep -E '^(torch|torchvision|triton)==' > ~/moge_constraints.txt
cat ~/moge_constraints.txt
```

Die Constraints-Datei sorgt dafür, dass pip diese Versionen bei den folgenden Schritten **nicht**
mehr verändert.

## 4. MoGe (v2 und v3) installieren

```bash
pip install -c ~/moge_constraints.txt git+https://github.com/microsoft/MoGe.git
python -c "from moge.model.v2 import MoGeModel; from moge.model.v3 import MoGeModel; print('MoGe v2+v3 ok')"
```

MoGe-3 braucht FlexGEMM (baut auf Triton auf, nur Linux). Meldet der zweite Befehl einen
FlexGEMM-Fehler, den Hinweisen im MoGe-README zur Installation von FlexGEMM folgen und dabei
ebenfalls `-c ~/moge_constraints.txt` angeben.

## 5. OpenCV mit Fenster-Unterstützung

MoGe zieht `opencv-python-headless`, damit funktioniert `cv2.imshow` nicht.

```bash
pip uninstall -y opencv-python-headless opencv-python
pip install -c ~/moge_constraints.txt opencv-python
```

## 6. Restliche Pakete

```bash
pip install -c ~/moge_constraints.txt huggingface_hub matplotlib pyyaml
```

- `huggingface_hub`: Laden der Checkpoints
- `matplotlib`, `pyyaml`: Plots und `scene_log.yaml` in der Auswertung

`rclpy`, `message_filters`, `rosbag2_py` **nicht** per pip installieren. Sie kommen aus
`/opt/ros/jazzy` (siehe Schritt 8).

## 7. Prüfen

```bash
python - <<'EOF'
import numpy, scipy, cv2, torch
print("numpy", numpy.__version__, "| scipy", scipy.__version__, "| cv2", cv2.__version__)
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("GUI:", "ok" if "GTK" in cv2.getBuildInformation() or "QT" in cv2.getBuildInformation() else "FEHLT")
EOF
```

Modelle einmal herunterladen. Beide Repos sind öffentlich (MIT), ein Login ist nicht nötig:

```bash
python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('Ruicheng/moge-3-vitl', 'model.pt'))"
python -c "from moge.model.v2 import MoGeModel; MoGeModel.from_pretrained('Ruicheng/moge-2-vitl'); print('MoGe-2 ok')"
```

## 8. Starten mit ROS

Die Skripte entfernen selbst die Conan-Pfade (altes numpy 1.26) aus `sys.path` und hängen
`/opt/ros/jazzy/lib/python3.12/site-packages` an. Trotzdem `PYTHONPATH` beim Start sauber setzen,
**ohne Semikolon**. Sonst trennt die Shell oder der Prozessmanager den Befehl auf, und python
läuft ohne ROS-Pfad:

```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=48
env PYTHONPATH=/opt/ros/jazzy/lib/python3.12/site-packages:$HOME/Documents/main \
    ~/miniforge3/envs/moge/bin/python -u navigation/mapping/moge3_depth_viewer.py
```

Im Prozessmanager (links_and_nodes) denselben Befehl als **eine** Zeile eintragen und direkt das
Python des Envs aufrufen (`~/miniforge3/envs/moge/bin/python`), nicht `conda run`.

## Häufige Fehler

| Meldung | Ursache | Lösung |
|---|---|---|
| `No space left on device` beim pip-Build | Bauordner in `/run/user` | Schritt 1 in derselben Shell |
| pip lädt mehrere torch-Versionen | Abhängigkeiten ohne Constraints | `-c ~/moge_constraints.txt` bei jedem `pip install` |
| `np.long` / "NumPy version >=2 required" | numpy 1.26 aus dem Conan-`PYTHONPATH` | `PYTHONPATH` wie in Schritt 8 setzen |
| `ModuleNotFoundError: rclpy` | ROS-Pfad fehlt (z. B. durch `;` im Startbefehl) | Schritt 8 |
| Fenster öffnet sich nicht / `cv2.imshow` Fehler | `opencv-python-headless` | Schritt 5 |
| `cuda False` | CPU-torch installiert | Schritt 3 mit `--index-url .../cu128` wiederholen |
