# Mapping pipeline

Camera → depth completion → filtered depth → map → collision checking in Python.
Every step is a process of its own with exactly one task. The steps communicate over
ROS 2 topics (exception: the camera input via LN), following the rules in
[Communication](#communication).

```
 Azure Kinect ──(local only)──► [1] ln_camera_publisher.py
                                      │  LN: camera.perception/{rgb,depth,cam_k}  (+ timestamp)
 real system: camera driver ──────────┤
                                      ▼
                               [2] ln_ros_camera_bridge.py
                                      │  /camera/color/image_raw ────────────────┐ colour, by stamp
                                      │  /camera/depth/image_raw                 │
                                      │  /camera/color/camera_info (latched) ────┤ intrinsics
                                      ▼                                          │
                               [3] depth_completion/depth_completion_node.py (GPU)
                                      │  /camera/completed/depth/image_raw       │
                                      ▼                                          │
  synchronization_buffer.py ──► [4] camera_frame_filter.py ◄─────────────────────┤ (--mask_humans)
  (odometry + q over time)            │  /tf  (map → Wt → camera, Wt → ground)   │
          user_info (W_T_CAM) ──►     │  /camera/filtered/depth/image_raw        │
                                      │  /camera/filtered/points (optional)      │
                         ┌────────────┴─────────────┐                            │
                         ▼                          ▼                            │
              [5] start_nvblox.py ◄─────────────────┼────────────────────────────┘ (point colours)
                 /nvblox/occupied_points_rgb    start_octomap_mapping.py (alternative)
                         │                          /octomap_point_cloud_centers
                         └────────────┬─────────────┘
                                      ▼
                               [6] collision_map_provider.py  (Python class, no process of its own)
                                      COAL octree / AABBs / FROCC map for Arm_Collision_Model
```

## Steps

| # | File | Task | Trigger / inputs | Output |
|---|---|---|---|---|
| 1 | `ln_camera_publisher.py` | Reads the Azure Kinect via k4a and publishes RGB, registered depth and K with a timestamp on LN. **Only with a locally connected camera**; on the real system the driver does this. | k4a | LN `camera.perception/*` |
| 2 | `ln_ros_camera_bridge.py` | LN camera data 1:1 to ROS. `read()` blocks, so no timer: every frame is forwarded at once. `camera_info` is sent once, latched. | LN | `/camera/color/*`, `/camera/depth/image_raw` |
| 3 | `depth_completion/depth_completion_node.py` | Monocular depth model (MoGe-2/3 or Depth Pro 2, default MoGe-2 L with 600 tokens, 29 Hz on the robot's RTX A6000) + gradient fusion (CG on the GPU) + edge filter fill the holes of the sensor depth. Measured pixels stay unchanged. Publishes only the completed depth. | `/camera/depth/image_raw` (trigger), `/camera/color/image_raw`, `camera_info` | `/camera/completed/depth/image_raw` |
| 4 | `camera_frame_filter.py` | TF at the frame stamp, min/max range, flying pixels at depth edges, masks humans (`--mask_humans`) and the robot arm (`--filter_robot`), publishes the filtered depth (+ point cloud). `--depth_source raw` uses the raw depth (without step 3). | `/camera/completed/depth/image_raw` (trigger), colour (`--mask_humans`), `camera_info` | `/tf`, `/camera/filtered/*` |
| 5 | `start_nvblox.py` | TSDF map from the filtered depth; per frame: integrate, query the occupied voxels around the camera, remove isolated voxels (speckles), publish. LN service `nvblox.reset_map`. | `/camera/filtered/depth/image_raw` (trigger), colour, `camera_info` | `/nvblox/occupied_points_rgb` |
| 5' | `start_octomap_mapping.py` | Alternative: ROS `octomap_server` on the filtered point cloud. | `/camera/filtered/points` | `/octomap_point_cloud_centers` |
| 6 | `collision_map_provider.py` | `CollisionMapProvider`: fetches the map (nvblox or octomap, `MAP_BACKEND`) into Python, builds a COAL octree / AABBs / FROCC map and optionally filters out an object. | map topic | Python API |

Everything about step 3 is in the subfolder `depth_completion/`: ROS node, models
(`depth_models.py`), fusion and edge filter (`depth_fusion.py`), live viewer (`viewer.py`) and
offline benchmark (`bench/`); details with drawings in
[depth_completion/README.md](depth_completion/README.md).
Helper modules: `pipeline_io.py` (QoS profiles, colour lookup by stamp, status line),
`synchronization_buffer.py` (LN service: odometry and joint angles at a timestamp),
`arm_filter.py` (arm mask, `depth_to_pointcloud`), `ros_utils.py` (timestamp conversion, TF from
a 4×4 matrix).

## Communication

All stages follow the same rules (implemented in `pipeline_io.py`):

1. **One stamp per frame.** Every message of a frame carries the LN capture time as
   `header.stamp` and `frame_id = camera`, through the whole pipeline. The stamp is the only key
   used to match messages.
2. **A new depth image is the only trigger.** No ROS timers, no `TimeSynchronizer`, one thread
   per node: each frame is processed in the depth callback.
3. **Latest wins.** The triggering depth topics use reliable QoS with keep last 1. While a stage
   is busy, DDS keeps only the newest depth image, so a slow stage drops old frames instead of
   building up latency, and nothing is lost silently on the way.
4. **Colour comes directly from the bridge.** Step 3 does not re-publish it. Stages that need it
   look it up by the stamp of their depth image:
   - Step 3 receives raw depth and colour at the same time and keeps the last one of each; when
     both have the same stamp, the frame is processed. A depth image that arrives first waits
     for its colour image.
   - Steps 4 (`--mask_humans`) and 5 (point colours) get their depth 100–200 ms after the colour
     image. They keep the last 10 colour images in a `message_filters.Cache` (QoS keep last 10)
     and look up the one with the identical stamp. In step 4 a depth image whose colour image is
     missing waits for it; step 5 then colours the points grey.
5. **`camera_info` is static.** The bridge sends it once with transient-local ("latched") QoS,
   so stages that start later still receive it. Colour and depth share one K (the depth is
   registered to the colour camera).
6. **Same status line everywhere.** Every stage logs every 5 s:

   ```
   completion: in 14.2 Hz | out 8.1 Hz | latency 180 ms | convert 0.6 | model 53.0 | fusion 12.1 | publish 1.4 ms
   ```

   `in` = trigger images received, `out` = frames published, `latency` = publish time minus
   capture time. Frames dropped because a stage is busy are the difference between `in` and
   `out`; frames lost between two stages are the difference between the `out` of the earlier
   and the `in` of the later stage.

| QoS profile | Settings | Used for |
|---|---|---|
| `qos_trigger()` | reliable, keep last 1 | depth images (trigger) |
| `qos_buffer(10)` | reliable, keep last 10 | colour images |
| `qos_latched()` | reliable, keep last 1, transient local | `camera_info` |

**Rosbags:** a reliable subscriber receives nothing from a best-effort publisher. If a bag was
recorded with best-effort QoS (`reliability: 2` in `offered_qos_profiles` of `metadata.yaml`),
play it with a QoS override to reliable (`ros2 bag play --qos-profile-overrides-path`).

## Starting

```bash
python3 -m navigation.mapping.synchronization_buffer
```

```bash
python3 -m navigation.mapping.ln_camera_publisher            # only with a local camera
```

```bash
python3 -m navigation.mapping.ln_ros_camera_bridge
```

```bash
python navigation/mapping/depth_completion/depth_completion_node.py   # conda env depthcompletion
```

Model and token count are set at the top of the script (`MODEL`, `NUM_TOKENS`, names from
`depth_models.MODELS`). In the LN process manager, call the env's Python directly
(`~/miniforge3/envs/depthcompletion/bin/python`), not `conda run`. Setting up the env:
[depth_completion/README.md](depth_completion/README.md), section "Installation".

```bash
python3 -m navigation.mapping.camera_frame_filter --mask_humans --filter_robot
```

Options of step 4:

- `--publish_pointcloud` / `--no-publish_pointcloud`: publish the point cloud on
  `/camera/filtered/points` or not (default: off). The filtered depth for nvblox is always
  published.
- `--depth_source completed`: use the completed depth from `/camera/completed/depth/image_raw`
  (default).
- `--depth_source raw`: use the original depth from `/camera/depth/image_raw` (bridge), without
  step 3.
- `--raw`: short for `--depth_source raw`.

```bash
python3 -m navigation.mapping.start_nvblox                   # or: ros2 launch navigation/mapping/start_octomap_mapping.py
```

Step 6 is not started; `Arm_Collision_Model` instantiates it.

## Dev viewer

`depth_completion/viewer.py` shows in one Rerun window RGB, sensor depth, model depth and merged
depth, and below them the 3D point cloud of each of these depths (rotatable with the mouse). It
runs exactly the code of step 3 (`depth_models` + `depth_fusion.fill`). It is not part of the
production chain: it subscribes the raw bridge topics, publishes nothing and can run next to the
pipeline (own model in GPU memory). With `--frames f.npz` it plays a recording without ROS
instead, with `--eval` it also shows the error below 3 m as a curve over time.

```bash
python navigation/mapping/depth_completion/viewer.py --model moge --num-tokens 600
```

```bash
python navigation/mapping/depth_completion/viewer.py --model depthpro2-s --window 4
```

```bash
python navigation/mapping/depth_completion/viewer.py --frames recording.npz
```

```bash
python navigation/mapping/depth_completion/viewer.py --set edge_jump=0
```

The last call switches the edge filter off for comparison. There is one node and one viewer for
all models; the model is chosen by name.

## Design decisions

**ROS 2 as the data bus between the steps.** ROS has native types for everything that flows here
(`Image`, `CameraInfo`, `PointCloud2`, TF, Octomap), and every intermediate step can be inspected
in RViz (raw next to completed depth, filtered point cloud, map). nvblox, `octomap_server` and
`message_filters` work with these types directly. LN stays where it already is: camera driver,
synchronisation buffer, `user_info` and reset services.

**One process per task.** The old `azure_interface.py` did camera, bridge, filter, TF and point
cloud in one loop. Now every step can be started on its own, replaced (e.g. skip step 3 with
`--depth_source raw`, swap nvblox for octomap) and run in its own environment. Step 3 runs in the
conda env `depthcompletion` with torch, the rest in the system Python with LN and `cv_bridge`.

**Local and real system have the same input.** Step 1 publishes exactly the LN topics of the real
camera driver. The bridge therefore only has the LN path, and everything from step 2 on is
identical in both setups.

**The original timestamp is passed through.** The capture timestamp of step 1 (`time.time()`
domain, the same clock as in the `SynchronizationBuffer`) becomes `header.stamp` in step 2 and
stays unchanged up to the map: step 3 takes over the header of its input, the frame filter
stamps TF, depth and point cloud with it. So step 4 finds odometry and joint angles of *this*
frame in the buffer (needed for the arm mask, especially since step 3 adds latency), and
nvblox/octomap find exactly the camera pose of this frame via TF.

**Trigger plus lookup instead of a synchroniser.** A `TimeSynchronizer` over colour, depth and
`camera_info` drops a whole frame as soon as one of the three messages is missing. With large
images and best-effort QoS this happened often enough that the frame filter ran at 0.3–3 Hz
while step 3 published at ~8 Hz. Now only the depth triggers, colour is looked up by stamp and
`camera_info` is latched, so one lost colour image costs at most that one frame.

**Colour is not routed through step 3.** Re-publishing the colour image next to the completed
depth cost 2.7 MB per frame only to forward it. Steps 4 and 5 take it from the bridge and bridge
the 100–200 ms latency with a buffer of 10 images.

**No timers.** Every stage works when a new image arrives and only then. nvblox therefore
publishes its point cloud once per integrated frame instead of at a fixed rate, and serves the
reset service while frames arrive.

**No comet tails at edges.** Monocular models draw object edges as a soft ramp, and exactly there
the ToF sensor often measures nothing. Without countermeasure, the ramp ends up in the map as a
tail between foreground and background. `depth_fusion.fill` therefore finds edges in the estimated
depth (jump in 1/depth > `edge_jump`), dilates them by `edge_grow` pixels and fills nothing there
(measured pixels always stay; flying pixels in the measurement are removed by step 4). 1/depth
instead of depth, because a floor seen at a grazing angle is equally steep in it at every
distance and does not count as an edge. On 6 recordings this reduces edge bleeding from 32 % to
0.04 % (0 % with `edge_grow=8`, at less filled area); details in
`depth_completion/bench/README.md`.

**Filtering after completion.** Humans and the arm are masked in step 4, i.e. *after* step 3. The
model thus sees the whole image (better scale), and whatever was filled into arm or human areas
is set back to 0 (= ignore) afterwards. Min/max range also applies only here, so that filled
values are limited as well.

**nvblox integrates the filtered depth, not the point cloud.** `nvblox_torch` integrates depth
images projectively (`add_depth_frame`). Step 4 therefore publishes the same filtered data as a
depth image for nvblox and, optionally, as `PointCloud2` for octomap, RViz and the fast path of
`CollisionMapProvider` (`use_pointcloud_fast`).

**TF comes from step 4.** The node that queries the synchronisation buffer anyway stamps
`map → Wt` with the odometry at the frame stamp and publishes it before depth and point cloud.

**Fusion only on the GPU (CUDA or Apple MPS).** The fusion parameters (`down=2`,
`sigma_edge=0.05`, 40 CG iterations) were determined with `depth_completion/bench/` on 6
recordings.

**Floating voxels are filtered in two places without lowering the rate.** Most floating voxels
come from flying pixels at depth edges (mixed values of foreground and background). Step 4
therefore drops pixels whose depth range in a 3×3 window exceeds `EDGE_REL_THRESH` (10 %) of their
own depth: two morphology operations, about 3 ms per frame on the CPU, and grazing surfaces such
as the floor are kept. What remains is removed by step 5 in the query window: occupied voxels with
fewer than `min_neighbors` (default 1) occupied neighbours in the 26-neighbourhood. The counting
runs as 3×3×3 pooling on the GPU and costs much less than the query itself. The map itself stays
unchanged; the filter only acts on the published point cloud.

## Topics

| Topic | Type | QoS | From | To |
|---|---|---|---|---|
| `/camera/color/image_raw` | `Image` rgb8 | buffer (10) | 2 | 3, 4 (`--mask_humans`), 5 (colours) |
| `/camera/depth/image_raw` | `Image` 16UC1 mm | trigger | 2 | 3, (4 with `--depth_source raw`) |
| `/camera/color/camera_info` | `CameraInfo` | latched | 2 | 3, 4, 5 |
| `/camera/completed/depth/image_raw` | `Image` 16UC1 mm | trigger | 3 | 4 |
| `/tf` | `map → Wt → camera`, `Wt → ground` | default | 4 | 5, 5', RViz |
| `/camera/filtered/depth/image_raw` | `Image` 16UC1 mm | trigger | 4 | 5 |
| `/camera/filtered/points` | `PointCloud2` (frame `camera`) | default | 4 | 5', RViz |
| `/nvblox/occupied_points_rgb` | `PointCloud2` (frame `map`) | default | 5 | 6 |
| `/octomap_point_cloud_centers` | `PointCloud2` (frame `map`) | default | 5' | 6 |

`/camera/completed/color/image_raw`, `/camera/completed/camera_info` and
`/camera/depth/camera_info` no longer exist.
