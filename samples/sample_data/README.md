# Sample 1: sample data (offline, library API only)

Run the planner on a short clip cut out of a vehicle rosbag, without ROS:
frames, odometry and a route on disk in, plans (`.npy`) and an optional video
out. `run_clip.cpp` is the smallest complete use of the library -- the same
pipeline the vehicle node runs, in about a hundred lines.

## 1. Make a clip

```bash
uv run --with mcap --with mcap-ros2-support --with numpy --with opencv-python-headless \
    python3 samples/sample_data/export_clip.py <bag_dir> <clip_dir> --start 60 --duration 30
```

`<bag_dir>` is the vehicle's directory of `.mcap` files (no `metadata.yaml`
or ROS install needed). Options:

| option | default | |
|---|---|---|
| `--start`, `--duration` | `0`, `30` | seconds after the bag start; 2 s before `--start` are kept as history |
| `--views` | `camera0,camera3,camera4` | camera names or topics, view 0 first (CAM_FRONT, FRONT_LEFT, FRONT_RIGHT on prd_jt) |
| `--full-res` | off | keep the recorded JPEGs instead of 512x256 crops (~10x larger) |
| `--lanelet_map` | none | the map's centerlines (`scripts/convert_map.py`): `route.npy` is the bag's mission-planning route, as the vehicle builds it, instead of the driven path |
| `--route-ahead` | `30` | seconds of driven path past the clip end kept in `route.npy` (without `--lanelet_map`) |

A 30 s three-camera clip takes ~30 s to cut and ~65 MB on disk.

### Clip format

```
clip.json          views (topic, directory, per-frame stamps), time origin t0_ns,
                   map origin origin_xy, view-0 calibration for the overlay
view0/ view1/ ...  one JPEG per frame (512x256: the model's crop + resize already
                   applied, which the window's preprocessing then leaves alone)
odometry.npy       float32 [M, 8]: t, x, y, yaw, vx, vy, ax, ay (50 Hz)
route.npy          float32 [N, 2]: route polyline in the map frame: the mission route
                   (--lanelet_map) or the driven path; clip.json route_kind says which
route_points.npy   float32 [T, 1 + 2K]: t, x1, y1, ... -- the route points the vehicle
                   fed its model, when the bag recorded them (LatentDrive runs)
```

Times are seconds since `t0_ns`, positions metres relative to `origin_xy`.

## 2. Run it

```bash
build/bin/latentdrive-sample-clip \
    --engine_file=models/engine/camfront3-rptw-lane2-e1-mixed.engine \
    --clip_dir=<clip_dir> --out_video=<clip_dir>/replay.mp4
```

```
[info] Deserialized ... (3 views, status [1, 16] = 6 route points, 40 waypoints, output 'plan')
[info] Clip ...: 30 s, 321 front frames, 3 views, status [1, 16] with route points from route.npy
[info] t  20.07 s  v  2.7 m/s  |4s|  19.1 m  route point 50 m at (49.7, 2.2)  engine 65.6 ms
[info] 300 plans -> <clip_dir>/plans_camfront3-rptw-lane2-e1-mixed.npy (0 warm-up steps, 0 skipped, 0 blank side-view slots)
```

| flag | default | |
|---|---|---|
| `--route_source` | `file` for a mission-route clip, else `recorded` if the clip has `route_points.npy`, else `file` | `recorded`: the vehicle's route points; `file`: build them from `route.npy` |
| `--missing_view_fill` | `skip` | a side slot without a frame within the tolerance: skip the step, or `zeros` (the blank frame the camfront3 training used) |
| `--view_tolerance_s` | `0.05` | max side/front stamp difference per history slot |
| `--max_rate_hz` | `10` | plans per second at most, paced on the front stamps |
| `--out_video` | none | camera ribbon + BEV mp4 |

Outputs, next to the clip: `plans_<engine>.npy` float32 `[T, N, 3]` (x
forward, y left, heading, in the ego frame of each step) and
`plans_<engine>_t.npy` with the step times.

The engine decides the contract, the clip only has to cover it: a
single-camera engine reads `view0` alone, a camfront3 one all three; a
single-subgoal engine (status `[1, 6]`) runs on the same clip with
`--route_source=file`.

## What the loop does

1. Every camera frame goes into `preprocess::ViewWindow` as it "arrives"
   (preprocessed once there). Side frames up to 50 ms past the front frame are
   pushed first: offline every frame is at hand, so nothing waits.
2. Every odometry sample goes into `status::StatusBuilder`.
3. On each CAM_FRONT frame (at most 10 Hz): the status (route points + motion
   terms), the `[V, 3, 4, 256, 512]` window assembled straight into the
   engine's pinned input, one `ForwardPrepared`.

With `--lanelet_map` the clip's `route.npy` is the vehicle's route, built from
the bag's `/planning/mission_planning/route` exactly as its planner does, and
run_clip uses it by default: on a LatentDrive run the plans match those from
the recorded route points (`--route_source=recorded`) to millimetres. Without
it `route.npy` is the driven path, so the route points follow where the car
went; where the car stayed on its route the two agree.
