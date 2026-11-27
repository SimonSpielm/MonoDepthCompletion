# Offline benchmark for the depth completion

Tests **exactly the production code** (`../depth_fusion.py` and `../depth_models.py`, which the
ROS node `depth_completion_node.py` also imports) without ROS, camera or robot. Parameters and
code changes can thus be compared deterministically before they go to the robot.

## Setup

The same conda env as for the ROS node, see [../README.md](../README.md), section
"Installation". ROS is not needed for the benchmark. The Depth Pro 2 weights are licensed for
non-commercial research only.

## Test suite on the robot PC (`run_suite.py`)

Runs all model configurations on recorded scenes and writes a report for the analysis. Measured
per configuration: runtime (model, completion, total, Hz), GPU memory, accuracy (32 and 128 px
holes, gross errors, edge band) and flicker with a static camera. Default grid (`--list`, 27
configurations):

- MoGe-2 S, B, L, L-normal with 600, 1200 and 2400 tokens
- MoGe-3 L with refinement (600, 1200, 2400 tokens) and without (1200), MoGe-3 G (600, 1200)
- Depth Pro 2 S, B, L, each single frame, streaming window 4 and 16 (H with `--with-h`)

First a short run (5 configurations, about 10 minutes), then everything (rather 1–2 hours). Do
not run the ROS pipeline at the same time, it shares the GPU.

```bash
python run_suite.py --quick --bags ~/Desktop/rosbags/depth_estimation ~/Desktop/rosbags/octomap/octomap ~/Desktop/rosbags/local_map/azure_kinect
```

```bash
python run_suite.py --bags ~/Desktop/rosbags/depth_estimation ~/Desktop/rosbags/octomap/octomap ~/Desktop/rosbags/local_map/azure_kinect
```

- Bags are converted once to `~/depth_suite_data` (30 consecutive frames, `--n`, `--step`). ROS 1
  bags (`.bag`) work as well.
- The report goes to `bench/reports/<host>_<time>/`: `report.md` (table), `summary.csv`,
  `results.jsonl` (including per-recording values and error messages), `env.json` (GPU, driver,
  versions, commit), `suite.log`.
- If the run aborts, `--resume bench/reports/<folder>` continues where it stopped.
- Single configurations: `--only moge2-b:600 moge3-l:1200:refine3 depthpro2-s:stream4`.
- MoGe-3 with refinement needs `flex_gemm` (installation step 6); if it is missing, these
  configurations are logged as failed and the rest continues.

For the analysis, commit and push the report folder, or send it zipped.

## Data

| Source | From | For |
|---|---|---|
| `--synthetic N` | generated | floor, slanted wall, boxes, thin poles. ToF-like sensor (holes at edges, dark spots, range falloff, mixed pixels), MoGe-like prediction (soft edge ramps, smooth scale error). Ground truth known everywhere. Fast, good for edge experiments. |
| `--frames F.npz --preds P.npz` | `moge_sweep.py capture` and `infer` on the robot | real scenes. `preds.npz` contains the MoGe predictions, so MoGe does not have to run locally. |
| `--frames F.npz --predict depthpro2-s` | as above, without a prediction cache | runs the model locally (CUDA, Apple MPS or CPU) and writes `F_preds_<model>.npz`. |
| `bag_to_frames.py BAG --out F.npz` | ROS 1 or ROS 2 bag (`pip install rosbags`, no ROS needed) | RGB/depth pairs with the same stamp, FoV from `camera_info`. |

`--frames` takes several files and averages over all of them. The recordings are in
`rosbags/frames/` (not in git).

## Metrics

Each frame is evaluated with a fresh fusion (no warm start and no EMA), so runs are comparable.

- `hold`: general accuracy. For recordings, random 32 px blocks of measured pixels are hidden
  (protocol of the earlier sweeps); for synthetic data, all real holes away from edges count.
- `edge`: behaviour at edges. For recordings, a 6 px band around measured depth edges is hidden
  (what the ToF actually loses at edges); for synthetic data, all real holes up to 6 px next to an
  edge count.
  - `bleed`: fraction of pixels filled with > 10 % error, i.e. floating points and comet tails in
    the map. Lower is better.
  - `cov`: fraction of pixels that get a value at all. Always read together with `bleed`: a
    filler that fills nothing has `bleed` 0, but also `cov` 0.
- `<3m`: the same values only for pixels with depth below 3 m (`AbsRel`, mean error in `cm`).
- `--hold-px`: edge length of the hidden blocks (default 32). With 128, large holes appear as on
  black objects or glass, far from any measurement.
- `ms`: median runtime of the completion per frame on `--device`. Not comparable with the A6000,
  only a relative measure. Total runtime model + completion: `speed.py`.
- `--temporal`: flicker instead of accuracy. Each `--frames` file runs as a sequence with memory
  as in the live node (smoothed scale, warm start). Measured is the mean relative change between
  consecutive frames: on filled pixels (`filled`, also below 3 m), for comparison for the model
  alone (`model`) and for the sensor (`sensor`). Only meaningful with recordings of a static
  camera.

## Examples

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

`--set` and `--sweep` set any parameter of `GradientFusion`
(`down, lam, mu_near, mu_far, far_band, sigma_edge, z_ref, outlier, iters, tol`) and the
parameters of `fill` (`ema, edge_jump, edge_grow`). `edge_jump=0` switches the edge filter off.
Without them, the defaults of `depth_fusion.py` (= production) apply.

## Runtime

```bash
python speed.py --frames f.npz --models moge2-s:600 moge2-b:600 moge:1200 depthpro2-s
```

## Comparing models directly

```bash
python compare_models.py image.jpg --models depthpro2-s depthpro2-b
```

```bash
python compare_models.py --frames frames.npz --index 0
```

Writes `compare_models.png`: RGB, and per model the depth and the edges (white = hard jump,
orange = soft ramp), plus runtime and the fraction of ramp pixels. In outdoor scenes, distant
floor seen at a grazing angle also counts as a ramp; the number is mainly meaningful indoors.

Measured on an M4 Pro (MPS): Depth Pro 2 S 130 ms, B 326 ms per frame at 1280×720.

## Results (6 recordings, fusion on MPS)

### Methods compared (MoGe-2 B, 600 tokens)

Accuracy with 2 frames per recording, flicker with 15 consecutive frames per recording (13 for
`rosbag`). Errors below 3 m.

| Method | small holes | large holes (128 px) | coverage small / large | edge bleeding | filled edge holes | flicker filled | ms |
|---|---|---|---|---|---|---|---|
| no edge filter (Depth Pro 2 S) | 2.48 % | – | – | 20.9 % | 100 % | – | – |
| old 3×3 edge filter (Depth Pro 2 S) | 0.86 % | – | – | 3.95 % | 68 % | – | – |
| fusion + plane distance + support check (until Nov. 2026) | 0.40 % | 3.06 % | 87 / 89 % | 0.00 % | 2 % | 1.49 % | 24 |
| simple: smoothed correction factor + edge test | 0.33–0.35 % | 5.7–7.7 % | 52–67 / 45–60 % | 0.15–0.64 % | 3–24 % | 1.31 % | 4 |
| fusion + edge test in 1/depth, zone 8 px | 0.30 % | 2.07 % | 81 / 82 % | 0.00 % | 19 % | 1.29 % | 22 |
| **fusion + edge test in 1/depth, zone 4 px (default)** | 0.36 % | 2.40 % | **85 / 86 %** | 0.04 % | **21 %** | 1.36 % | 22 |

- Zone 4 px instead of 8 px: 66 % instead of 59 % of the real sensor holes are filled; in return
  a small residue of edge bleeding remains (0.04 %, 0.01 % below 3 m) and the error rises
  slightly.
- The simple method averages the correction factor across edges and has no guidance along
  surfaces in large holes; therefore the fusion stays.
- The edge test measures the jump in 1/depth instead of depth: with the relative depth jump, the
  floor seen at a grazing angle counted as an edge from about 2 m on, and coverage dropped to
  24–73 %.
- Flicker: the model alone changes by 1.4–1.6 % from frame to frame, the sensor by 0.2 %. No
  variant of the completion calms the model noticeably; the flicker comes from the model.
- Streaming mode of Depth Pro 2 S (`--stream --window W`, 30 frames per recording): flicker of the
  completed depth 1.91 % (single frame) → 0.89 % (window 4) → 0.80 % (window 16), model alone
  2.40 → 0.74 → 0.57 %. Model runtime on the M4 Pro 134 → 184 → 312 ms.
- Snapping intermediate values to the nearer side instead of discarding them was tested and
  rejected (21 % bleeding, often lands on the wrong side).
- The fusion parameters (`down`, `lam`, `sigma_edge`) hardly change the error in large holes; it
  is determined by the model.

### Models

Measured still with the earlier edge filter (plane distance + support check) and not recomputed;
for MoGe-2 B with the current filter see above. The runtimes include the completion of that time
(24 ms); the current one needs 22 ms with real predictions and without warm start.

Models, runtime on the M4 Pro (MPS, 1280×720, model + completion with 40 CG iterations), error and
bleeding below 3 m:

| Model | Tokens | ms | Hz | <3m small holes | <3m large holes (128 px) | edge bleed |
|---|---|---|---|---|---|---|
| MoGe-2 S (vits-normal) | 600 | 85 | 11.8 | 0.43 % / 0.6 cm | 3.80 % / 4.5 cm | 0.00 % |
| MoGe-2 S | 1200 | 134 | 7.5 | 0.41 % / 0.6 cm | 3.42 % / 4.1 cm | 0.00 % |
| **MoGe-2 B (vitb-normal)** | **600** | **107** | **9.4** | 0.40 % / 0.6 cm | 3.06 % / 3.7 cm | 0.00 % |
| MoGe-2 B | 700 | 117 | 8.5 | – | – | – |
| MoGe-2 B | 1200 | 182 | 5.5 | 0.39 % / 0.6 cm | 2.79 % / 3.2 cm | 0.00 % |
| MoGe-2 L (vitl) | 1200 | 306 | 3.3 | 0.37 % / 0.6 cm | 1.89 % / 2.6 cm | 0.00 % |
| Depth Pro 2 S | – | 153 | 6.5 | 0.43 % / 0.7 cm | 3.35 % / 4.4 cm | 0.00 % |
| Depth Pro 2 B | – | 348 | 2.9 | 0.41 % / 0.6 cm | 2.24 % / 3.3 cm | 0.00 % |
| Depth Pro 2 L | – | 1029 | 1.0 | 0.38 % / 0.6 cm | 1.59 % / 2.4 cm | 0.00 % |

On the M4 Pro, MoGe-2 B with 600 tokens is the choice for 8 Hz; the node adds conversion and
publishing. MoGe-3 needs CUDA for its refinement and was not compared here. On the robot, the
selection below applies.

## Results on the robot (RTX A6000, `run_suite.py`)

Report: [reports/rmc-lx0481_20261127_1009](reports/rmc-lx0481_20261127_1009/) with all 30
configurations, 4 recordings with 30 frames each (three bags from the robot and the tuning
recording), errors below 3 m. Plots: `plot_tradeoff.png` (error, gross errors and flicker over
runtime) and `plot_scenes.png` (per recording); RGB of the scenes in `scenes_rgb/`.

| Model | Variant | ms | Hz | 128 px AbsRel | gross | flicker |
|---|---|---|---|---|---|---|
| MoGe-2 S | 600 | 24 | 42.6 | 2.55 % | 5.14 % | 1.66 % |
| MoGe-2 B | 600 | 28 | 36.3 | 1.98 % | 3.70 % | 1.54 % |
| **MoGe-2 L** | **600** | **34** | **29.2** | **1.39 %** | **1.11 %** | **1.24 %** |
| MoGe-2 L | 1200 | 48 | 20.7 | 1.77 % | 2.35 % | 1.17 % |
| MoGe-3 L | 1200, refinement | 102 | 9.8 | 1.43 % | 1.03 % | 1.21 % |
| MoGe-3 G | 600, refinement | 104 | 9.6 | 1.25 % | 1.20 % | 1.13 % |
| Depth Pro 2 B | stream 4 | 52 | 19.2 | 1.56 % | 1.03 % | 0.94 % |
| Depth Pro 2 L | single frame | 98 | 10.2 | 1.24 % | 0.90 % | 1.83 % |
| Depth Pro 2 H | stream 4 | 195 | 5.1 | 1.16 % | 1.05 % | 0.78 % |

- The completion needs 9 ms on the A6000; 24 of the 30 configurations reach 8 Hz.
- All models fill small holes (32 px) with 0.40–0.49 %.
- MoGe-2 S and B make many gross errors, mainly on the tuning recording (up to 17 %).
- More tokens bring no consistent gain; the MoGe-3 refinement brings little at twice the time.
- The streaming mode of Depth Pro 2 halves the flicker; for S it costs accuracy.
- **The default of the ROS node is therefore MoGe-2 L with 600 tokens.** If flicker matters more
  than the licence: Depth Pro 2 B in streaming mode.
- Differences below about 0.3 percentage points between the large models are not reliable with
  2 held-out frames per recording.
