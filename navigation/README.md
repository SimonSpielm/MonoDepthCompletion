
# How to Run

## 1. Start `odometry_buffer`

- Creates synchronized odometry-timestamp pairs that can be subscribed to via LN topic.
- Command:

```bash
python3 map/odometry_buffer.py
```

---

## 2. Start `camera_ros_publisher`

- Bridges LN to ROS topics.
- Publishes RGB, depth, and odometry data.
- Masks humans and uses a plane fitter to fill depth image gaps (floor).
- Command:

```bash
python3 -m navigation.map.map_publisher
```

- Add `--use_ln` flag to use the LN camera topic, otherwise reads directly from the camera.

---

## 3. Start `rmc_local_map`

- Generates a 2.5D map of the environment.
- More info: [rmc_local_mapping](https://rmc-github.robotic.dlr.de/moro/rmc_local_mapping)
- Command:

```bash
rosrun rmc_local_mapping rmc_local_mapping __name:=rmc_local_mapping _output:=screen
```

- **Important:** Make sure `rmc_local_mapping_params` has started — this should run automatically. If needed, run:

```bash
rosparam load $(main)/navigation/map/rmc_local_mapping_params.yaml
```

---

## 4. Visualize the map in RViz

- Ensure `grid_map/[1.6.2]@3rdparty/unstable` is installed as a dependency.
- In RViz:
  - Panels → Display → Add → By topic → `local_map`

---

## 5. Start `reset_map_service`

- LN service wrapper to call a ROS service to reset the local map.
- Command:

```bash
python3 -m navigation.map.service.reset_map_service
```

---

## 6. Start `safety_limits`

- Computes safe velocity limits to avoid driving into obstacles (only forward).
- Can be enabled/disabled via LN service.
- Sends limits to the platform model, which constrains the desired velocity.
- Command:

```bash
python3 -m navigation.map.safety_limits
```

---

## 7. Run `trajectory_manager (if needed)`

- Calculates wheelchair trajectories to target poses.
- Also considers the obstacle map (not yet tested on the real system).


# How to integrate veloctiy limits to scts

```yaml
external:
  wheelchair_velocity_limits:
    active: True
```


# How to integrate trajectory planning to scts

```yaml
states:

 go_1m_door:
  parameters:
   odometry_active: False
   tsm.targetCM: wheelchair
   tsm.wheelchairCM: position # this doesnt work
  external:
   wheelchair_trajectory_planner:
    go_to_wsr: target.wheelchair_relocalization_frame    # target pose (defined in odb), get obj pose from go_to_anchoring, go_to_wsr
    continuous: True                                     # choose if the wheelchair should not stop at that point but rather drive smoothly through that pose to reach next target
    obstacle_avoidance: False                            # use rmc_local_map and obstacle avoidance
    v_end: 0.2                                           # 0.2 m/s at trajectory end since we dont want to stop here
    # path planning parameters (optional, defaults in brackets):
    min_length_straight_end: 0.2                         # minimal length of the straight segment at the END of the path [m] (0.2)
    min_length_straight_start: 0.0                       # length of the straight segment at the BEGINNING of the path, driven along current heading [m] (0.0 = off)
    max_curvature: 2.5                                   # maximum curvature of the dubins path [1/m] (2.5)
    min_curvature: 0.1                                   # minimum curvature of the dubins path [1/m] (0.1)
    v_max: 0.15                                          # maximum velocity on a straight path [m/s] (0.15)
    a_max: 0.025                                         # maximum acceleration on a straight path [m/s^2] (0.025)
    a_min: -0.025                                        # minimum (braking) acceleration on a straight path [m/s^2] (-0.025)
    distance_slow_start: 0                               # distance at the beginning over which v_max_start applies [m] (0)
    distance_slow_end: 0.1                               # distance at the end over which v_max_end applies [m] (0.1)
    v_max_start: 0.05                                    # max velocity during distance_slow_start [m/s] (0.05)
    v_max_end: 0.05                                      # max velocity during distance_slow_end [m/s] (0.05)
  transitions:
  - to: go_in_front_of_door
    predicates:
    - external:
       wheelchair:
        arrived_at: target.wheelchair_relocalization_frame
 go_in_front_of_door:                                  # wbc !
  parameters:
    wbc_box: [0.015,0.25,-999,999]
    odometry_active: True
    wbc_mode: 7
    tsm.targetCM: arm
    tsm.wheelchairCM: manual
  external:
   wheelchair_trajectory_planner:
    go_to_anchoring: target.wheelchair_approach_frame
    continuous: True 
    wbc_command_gain: 1
  input_mapping:
   - mapping: [.4tx,.1ty,.1tz] 
  transitions: # arm control phase so using arm transition
   - to: rotate
     predicates:
     - aux_fct:
        frame: EE
        ref: target
        function: StraightMap
        mapping: [1,0,0,0,0,0]
        range: [0.25, inf]
```

you can also only update wbc gain without defining new target:
```yaml
    wheelchair_trajectory_planner:
     wbc_command_gain: 0.2
     update_only_wbc_gain: True  #this is important if no new target is definied !
```

### go_to_static / go_to_relative (pose based, no object / frame lookup)

Semantics (C = wheelchair pose, Ws = world frame):

- `go_to_static: [x, y, theta]` — specifies where **C shall be in Ws**: an absolute target pose in the world frame.
- `go_to_relative: [x, y, theta]` — specifies where the **wheelchair shall be relative to the current C**: x/y in the current local frame of C, theta (in rad) added to the current heading.

The pose is given as `[x, y, theta]`: x/y in meters, theta as an angle in radians (the trajectory planner converts it to degrees for the platform model):

```yaml
  drive_to_point:
    parameters:
      odometry_active: False
    events:
      tsm.targetCM: wheelchair
      tsm.wheelchairCM: position
    external:
      wheelchair_trajectory_planner:
        go_to_static: [1.0, 0.5, 1.570796]    # absolute pose in Ws, theta in rad, mode 'path' only
        # go_to_relative: [0.5, 0, 0.785398]  # same pose relative to the current pose, all modes
    transitions:
      - to: next
        predicates:
          - external:
             wheelchair:
             arrived_at: go_to_static.drive_to_point
```

- `go_to_static` is path mode only; `go_to_relative` works with all modes.
- a pose list has no frame name of its own, so the `arrived_at` name is `<key>.<state_name>` (unique per state).

### heading / rotation commands (no path planning)

Both modes run the position controller directly instead of dubins path planning. The value is
extracted from the `[x, y, theta]` pose by the TP:

- `mode: heading` — drive **straight** by the distance `x` (m). Value = first element.
- `mode: rotation` — rotate **in place** by the angle `theta` (rad). Value = third element.
- `mode: heading` / `mode: rotation` always run with `automatic: True`; the `automatic` flag can only be disabled for path commands.

```yaml
  drive_straight:
    parameters:
      odometry_active: False
    events:
      tsm.targetCM: wheelchair
      tsm.wheelchairCM: position
    external:
      wheelchair_trajectory_planner:
        go_to_relative: [0.5, 0, 0]
        mode: heading             # path (default) | heading | rotation
    transitions:
      - to: next
        predicates:
          - external:
             wheelchair:
             arrived_at: go_to_relative.drive_straight
```

For in place rotation use `mode: rotation` with the angle in rad in the third element
(e.g. `go_to_relative: [0, 0, 1.570796]`).

- the TP waits until the state is actually active before executing, and triggers `arrived_at` when done.
- put the heading state directly after a wheelchair state (waiting across arm phases is not supported, same as for path commands).

### Trigger naming (`arrived_at`)

When the motion of a state is finished, the TP sends one trigger named after the command's
`original_frame_name`. The `arrived_at` value in the transition must match that name exactly:

| `go_to_*` key (as written in the state) | `arrived_at` name |
|---|---|
| `go_to_anchoring: target.wheelchair_approach_frame` | `target.wheelchair_approach_frame` (the name as written) |
| `go_to_wsr: target.wheelchair_relocalization_frame` | `target.wheelchair_relocalization_frame` (the name as written) |
| `go_to_static` in state `drive_to_point` | `go_to_static.drive_to_point` |
| `go_to_relative` in state `turn_left` | `go_to_relative.turn_left` |

- **Frame name keys** (`go_to_anchoring` / `go_to_wsr`): the frame name is used verbatim, including the
  `target` prefix. `target` is **not** resolved to the actual target id for the trigger name — the id
  (from the grounding params) is only used for the pose lookup. So the `arrived_at` name stays the same
  for every object that uses the skill.
- **Pose keys** (`go_to_static` / `go_to_relative`): a pose list has no frame name, so the name is built
  automatically as `<key>.<state_name>`.
- The trigger **latches**: once set, it stays set until the task is reloaded. Give every state that must
  wait for its own motion a unique name (automatic for pose keys via the state name). Reusing a name in a
  later state on purpose makes that state's transition fire immediately, because the motion is already
  considered done (this is how e.g. `align_wheelchair_sct.yml` reuses `target.wheelchair_approach_frame`).
