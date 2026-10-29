# LatentDrive-TRT

Standalone TensorRT C++ inference framework for the LatentDrive e2e planner
(a V-JEPA2 ViT-L encoder feeding a planner transformer). It is the layer the
ROS 2 node sits on: the library knows nothing about ROS, files or JSON —
it takes camera frames and ego status, and returns a trajectory. One binary
serves every checkpoint -- one camera or three (CAM_FRONT + FRONT_LEFT +
FRONT_RIGHT; the earlier two-view wide + telephoto too), one 50 m subgoal or
six route points -- and the engine file decides which.

Two ways to use it, each with a sample (see [Samples](#samples)):

- **sample data**: a clip cut out of a rosbag, run offline through the
  library API alone (`samples/sample_data`, no ROS needed);
- **rosbag**: a ROS 2 node that plans live while the bag plays, with RViz
  (`samples/rosbag` + `ros2/latentdrive_ros`).

## Demo

![LatentDrive-TRT open-loop replay on an x2dev whole-scene clip](docs/demo.gif)

Open-loop replay of a *decel → red-light → turn* scene. The TensorRT fp16 plan
(rainbow ribbon, green = now → red = +4 s) is projected onto the front camera
and drawn in the bird's-eye view, alongside the ground truth and the 50 m
subgoal. ~5 ms engine latency per 10 Hz step; playback shown at 5× speed.

## Environment

Two ways to build and run — Docker (matches the vehicle exactly) or a native
build on the host. Both need the same stack:

- **CUDA 12.8** and **TensorRT 10.8.0.43** — match the version the engine will
  run against; a serialized engine is not portable across TensorRT versions, so
  build and run must use the same one.
- **cuDNN — not required.** TensorRT 10 does not depend on it and the planner
  uses no cuDNN path, so there is nothing to install.
- **OpenCV 4**, **gflags**, **nlohmann-json**, **CMake ≥ 3.22**, **g++** (C++17).

### Option A — Docker (recommended)

Build and run inside the **pilot-auto.xx1** deployment container, so everything
links the same CUDA 12.8 / TensorRT 10.8 the vehicle runs. Setup and the full
build → engine → validate → visualize walkthrough are in
[`docker/README.md`](docker/README.md):

```bash
./docker/build.sh                                 # pilot-auto-xx1:v0.53-cuda (once)
DATA=/path/to/workspace ./docker/run.sh           # shell at /workspace, data at /data
```

Every dependency is already in that image; jump straight to `cmake` inside the
shell it opens. The Autoware development image
`ghcr.io/autowarefoundation/autoware:universe-devel-cuda` (ROS 2 humble,
TensorRT 10.8.0.43) works as well -- `IMAGE=... ./docker/run.sh` -- and is
where the samples below were last verified.

### Option B — Native (no Docker)

Distro packages cover everything except the NVIDIA stack:

```bash
sudo apt install libopencv-dev libgflags-dev nlohmann-json3-dev cmake g++
```

Install **CUDA 12.8** and **TensorRT 10.8.0.43** from NVIDIA (apt repo or
tarball) — the same versions the deployment container pins, so an engine built
here loads on the vehicle; cuDNN is not needed. Then build:

```bash
cmake -B build -DTENSORRT_ROOT=/path/to/TensorRT   # or export TENSORRT_ROOT; omit if system-wide
cmake --build build -j
```

`TENSORRT_ROOT` may be a `-D` flag or an environment variable (a tarball's
headers and libs are picked up from `$TENSORRT_ROOT/{include,lib}`). If the
`/usr/local/cuda` symlink points at a toolkit other than the one TensorRT was
built for, pin it explicitly, e.g. `-DCUDA_DIR=/usr/local/cuda-12.8`. The
framework has no ROS or Docker dependency, so this is also how it builds on the
target vehicle.

## Samples

Both samples run the same pipeline the vehicle node does (autoware_tensorrt_e2e
with `planning_setting:=latentdrive`): frames preprocessed once on arrival,
the four history slots picked by stamp, side views paired within 50 ms,
route points from the route, motion terms from differenced poses, one plan per
CAM_FRONT frame at 10 Hz. Build an engine for the checkpoint first (on the GPU
that will run it; see [Commands](#commands)). `scripts/build.sh [onnx ...]` does
all of it -- library, unit tests, the ROS 2 package and an engine per ONNX
(default: the camfront3 lane2 export) -- or by hand:

```bash
cmake -B build && cmake --build build -j
build/bin/latentdrive-trt --mode=build-engine \
    --onnx_file=models/onnx/multi-camera/closed-loop-best/onnx/camfront3-rptw-lane2-e1.onnx \
    --engine_file=models/engine/camfront3-rptw-lane2-e1-mixed.engine   # ~7 min (3 views), ~1.5 min (1 view)
```

### 1. Sample data (offline, no ROS) -- [`samples/sample_data`](samples/sample_data)

```bash
# cut 30 s out of a vehicle bag (reads the .mcap files directly)
uv run --with mcap --with mcap-ros2-support --with numpy --with opencv-python-headless \
    python3 samples/sample_data/export_clip.py /bags/<bag_dir> /data/clips/my_clip --start 60 --duration 30

# plan on every CAM_FRONT frame; writes plans_<engine>.npy and an mp4
build/bin/latentdrive-sample-clip --engine_file=models/engine/camfront3-rptw-lane2-e1-mixed.engine \
    --clip_dir=/data/clips/my_clip --out_video=/data/clips/my_clip/replay.mp4
```

[`run_clip.cpp`](samples/sample_data/run_clip.cpp) is the reference for
embedding the library: ~100 lines from frames + odometry to a plan.

### 2. Rosbag (ROS 2) -- [`samples/rosbag`](samples/rosbag)

```bash
cd ros2 && colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release && cd ..   # once
samples/rosbag/prepare_bag.sh /bags/<bag_dir> /data/bags/my_bag          # once per bag: metadata.yaml
scripts/convert_map.py /maps/<map_dir> /data/maps/my_map.json                 # once per map
samples/rosbag/run_replay.sh /data/bags/my_bag models/engine/camfront3-rptw-lane2-e1-mixed.engine \
    lanelet_map:=/data/maps/my_map.json start_offset:=60
```

The node subscribes to the cameras, odometry and acceleration the bag plays,
builds the route points from the bag's mission-planning route on the map's
lanelet centerlines like the vehicle does (or takes the ones a LatentDrive
run recorded, or a `route.npy`), and publishes the plan plus a camera + BEV
image; RViz shows it next to the plan the vehicle drove with.

## Commands

One binary, `build/bin/latentdrive-trt`, three modes. Paths below use the
Docker mount points (`/data` = the hand-off package, `/workspace` = this repo);
natively, just pass the real host paths.

```bash
# ONNX -> engine (fp16, planner stage pinned to fp32 via IBuilder;
#   --prec=fp32 for the control engine, --pure_fp16 disables the pin)
latentdrive-trt --mode=build-engine --onnx_file=E.onnx --engine_file=E.engine

# validate + batch-1 benchmark against the hand-off package
latentdrive-trt --mode=run-package --engine_file=E.engine --package_dir=/data \
    [--ref_name=ref_output_fp32_wp10hz.npy]      # 40-wp model; default is the 8-wp refs

# camera ribbon + BEV mp4
latentdrive-trt --mode=visualize --engine_file=E.engine --package_dir=/data \
    --clip=follow_lane --out_video=viz/follow_lane.mp4 \
    [--smooth_alpha=0.35]                        # temporal smoothing of the DRAWN plan
    [--timeline=false]                           # drop the route + speed band
    [--plan_drift]                               # add the plan-drift strip (development)
```

`run-package` writes the `.npy` outputs the package's `validate.py` scores
(`python3 validate.py` → 11/11 for the 8-waypoint model). The waypoint count
(8 at 2 Hz, 40 at 10 Hz), the view count and the status size are read from
the engine at load, so the same binary serves every checkpoint. These modes
replay the hand-off packages (one or two views, `status.npy` precomputed); a
camfront3 engine runs on rosbag data through the [samples](#samples). The fp16
build's fp32-pin recipe lives in `engine_builder.cpp`; an ONNX whose batch
axis is symbolic (the closed-loop and two-view exports) is frozen to batch 1
by an optimization profile there, so it builds like a fixed-shape one.

A two-view engine needs the telephoto stream as well. Offline that is a second
clip directory, `<clip>/far` by default or `--far_clip_dir=DIR`, with the same
layout as the wide clip (`frames/*.jpg` + `sequence.json` carrying `t_us` per
frame); `--view_tolerance_s` (default 0.06) bounds how far a telephoto frame
may be from the wide frame it is paired with, and a step with no such frame
is skipped and reported — never filled with a substitute frame. See
[Two-view mode](#two-view-mode).

`visualize` renders the camera ribbon and the BEV over a timeline band:

- **route** -- the clips ship no route file, but the subgoal the model is fed
  is a point riding the route a fixed distance ahead, so lifting every step's
  subgoal into the map frame traces out the route and the driven path fills in
  the rest. Where the two meet the clip reaches its destination; where they do
  not the band says so instead of inventing one. Milestones are cut from that
  trace at a round spacing, and the ledger counts down the metres still owed to
  the next few and to the goal.
- **speed** -- straight from `status.npy`, in km/h, against the clip's own
  timeline.
- **plan drift**, behind `--plan_drift` -- how far the +4 s end of the plan
  moves per step. The reach is ego-relative, so a steady drive holds it flat
  and what is left is the plan disagreeing with itself; the raw and drawn
  traces are overlaid, which is what makes the smoothing below visible rather
  than merely claimed. It answers a question about the model rather than about
  the drive, so it is off unless asked for.

`visualize` draws a temporally smoothed plan: consecutive frames disagree by
about a metre on how far the plan reaches, which reads as a tail that pulses
in and out. `--smooth_alpha` is the weight of the fresh plan in that blend
(`1.0` draws the raw output; lower is smoother and laggier). It is display
only -- the accuracy readout and every `.npy` this binary writes stay on the
raw engine output.

## Model contract

| tensor | shape | meaning |
|---|---|---|
| `video` | `[1,3,4,256,512]` | single-view: 4 front-camera frames, oldest→newest, 0.5 s apart, RGB, ImageNet-normalized |
| `video` | `[1,V,3,4,256,512]` | V views, the same per camera, frame-aligned in time: camfront3 = front, front_left, front_right; the older two-view = wide front, telephoto |
| `status` | `[1,6]` | `[subgoal_x/10, subgoal_y/10, v_fwd, v_left, a_fwd, a_left]`, ego frame; subgoal = the route point 50 m ahead |
| `status` | `[1,16]` | `[p1x/10, p1y/10, …, p6x/10, p6y/10, v_fwd, v_left, a_fwd, a_left]`: route points 5, 10, 15, 20, 30, 50 m ahead |
| `traj` / `plan` | `[1,N,3]` | N waypoints (8 at 2 Hz, 40 at 10 Hz) over the next 4 s: `(x fwd [m], y left [m], heading [rad])` |

V, the route point count K (status = 2K + 4) and N are read from the engine at
load (`Planner::NumViews()`, `NumRoutePoints()`, `NumWaypoints()`), so one
binary serves every checkpoint.

The single-view ONNX comes from `deployment/export_onnx.py` in the training
repo (`kalfazed/LatentDrive`, opset 17, fixed batch 1, output `traj`); the
newer exports (closed-loop single-view, two-view) are opset 18 with a symbolic
batch axis and the output named `plan`. The planner finds the output by IO
mode and the engine build freezes the batch axis, so all of them load the same
way. The engine builds either there via `trtexec` or here via
`--mode=build-engine` (TensorRT C++ builder API). Both apply the same
precision recipe — fp16 with the planner stage pinned to fp32 — and produce
numerically identical results: validated 11/11 against the PyTorch reference,
~4.7 ms engine call / ~5.8 ms end-to-end per step on an RTX 4090 — 6% of the
10 Hz budget (single-view; a V-view model runs the encoder V times: the
camfront3 engines take ~35 ms on the vehicle's RTX 5070, ~63 ms on a laptop
RTX 4060). The fp32 pin covers one `image_fc` per view (`/m/image_fc*`): 407
layers on a camfront3 export, 403 on a single-view one -- the build log says
how many matched.

## Multi-view mode

The camfront3 checkpoints read CAM_FRONT, CAM_FRONT_LEFT and CAM_FRONT_RIGHT --
on the prd_jt vehicles `/sensing/camera/camera0`, `camera3` (+49.5°) and
`camera4` (-49.6°), all 57° HFOV, 2880x1860. (The earlier two-view checkpoints
read the wide front camera1 and the telephoto camera2.) The input contract,
matching `planner_io.py` and the camfront3 training:

- **Preprocessing** is the same for every view (`PreprocessFrame`): keep the
  full width, centre-crop the rows to a 2:1 band, bilinear resize to 512x256,
  /255, ImageNet mean/std, CHW. A side frame is a separate sensor's full
  frame, not a crop of the front image.
- **History** (`PickHistory`): view 0 anchors the four slots. For the newest
  front frame at t, each slot takes the front frame nearest in *time* to
  t-1.5, t-1.0, t-0.5, t -- never by counting frames, so a dropped frame moves
  one slot rather than every older one.
- **Alignment** (`ViewWindow`): for each chosen front frame, every other view's
  frame nearest in time, within `view_tolerance_s` (default 0.05 s: the
  training paired the side cameras within 50 ms; on this vehicle they run
  14 ms either side of the front). A slot without one fails the selection --
  that inference is **skipped and reported** -- unless
  `fill_missing_views` is set: then it gets the image the camfront3 training
  fed for a missing view, all zeros before normalization, and the selection
  lists it in `filled`. A stale or duplicated frame is never substituted.
- **Layout**: `[1, V, 3, 4, 256, 512]` is V contiguous `[3, 4, H, W]`
  channel-major blocks, view 0 first; `Planner::VideoNumel()` is V times the
  single-view size.

A live node does not plan the moment the front frame arrives: the side frames
of the same instant may still be in flight, so the plan waits for them up to
the tolerance past the front stamp (the ROS 2 node does this).

## Route points and status

`status::StatusBuilder` builds the `status` input the way the vehicle node
does (and `planner_io.ego_status` in training), with no ROS dependency:

- **Route points** (K = 6): the route polyline is cleaned of vertices that
  turn back by more than 120° (a mission route that doubles back at a junction
  entry otherwise puts a point behind the car), resampled every 1 m; the ego is
  projected onto it with a forward hint carried across calls (a route that
  crosses itself cannot snap it back; a jump of more than 10 m falls back to a
  global search); each point is 5/10/15/20/30/50 m of arc length past that
  vertex, clamped at the route end, then rotated into the ego frame. K = 1 is
  the original single-subgoal rule, 50 m ahead.
  `BuildWithRoutePoints` takes map-frame points directly instead -- e.g. the
  ones the vehicle recorded on `/tensorrt_e2e/debug/latentdrive/route_points`.
  On the vehicle the polyline is the mission route's preferred lanelets'
  centerlines (5 m fine centerlines from the map loader), joined and cut at
  the goal; `scripts/convert_map.py` writes those centerlines from a
  `lanelet2_map.osm`, and the ROS node (`route_source:=mission`) and
  `export_clip.py --lanelet_map` build the same polyline from them.
- **Motion terms**: the training differenced recorded positions: velocity is
  the chord over the last 0.2 s and acceleration its change over 0.1 s, both in
  the current ego frame (`MotionSource::kPoseDifference`, the default). On a
  turning car this carries the lateral acceleration (~v x yaw rate) that the
  odometry twist and the acceleration topic report as zero; `kTwist` uses those
  topics instead, and they stand in until 0.3 s of poses are there.

`tests/test_view_window.cpp` and `tests/test_status_builder.cpp` cover the
history pick, the tolerance rejection, the blank fill, the tensor layout, the
route points and the motion terms (`ctest --test-dir build`).

## Library API

```cpp
#include "frame_window.hpp"
#include "planner.hpp"
#include "status_builder.hpp"

auto planner = model::planner::CreatePlanner("camfront3-rptw-lane2-e1-mixed.engine");

preprocess::ViewWindowParams window_params;
window_params.num_views = planner->NumViews();           // 1 or 3, from the engine
auto window = preprocess::CreateViewWindow(window_params);

status::StatusBuilderParams status_params;
status_params.num_route_points = planner->NumRoutePoints();  // 1 or 6, from the engine
auto builder = status::CreateStatusBuilder(status_params);
builder->SetRoute(route_map_xy);                         // map-frame polyline, driving order

// camera callbacks: each frame is preprocessed exactly once
window->Push(view, bgr, stamp_s);                        // any resolution; crop + resize + normalize
// odometry callback
builder->PushPose({stamp_s, x, y, yaw, vx, vy, ax, ay});

// planning tick, on each view-0 frame
const preprocess::ViewSelection sel = window->Assemble(planner->VideoInput());
if (sel.ok) {                                            // else sel.reason says which view/slot lacks a frame
  const status::StatusResult st = builder->Build(stamp_s, ego_pose);
  common::Trajectory plan = planner->ForwardPrepared(st.ego.Encode());
}
```

A multi-threaded caller runs `preprocess::PreprocessFrame` on its own thread
and only takes its lock for `PushPreprocessed`. `FrameWindow` (count-based,
single view) is still there for the hand-off package, whose clips define the
history by frame index. The visualizers (`BevVisualizer`, `CameraProjector`)
draw on plain `cv::Mat`, so a node can publish the same canvas as
`sensor_msgs/Image`.

## Codebase layout

```
include/
  common/       logger, timer, cuda RAII helpers, trajectory types
  latentdrive/  planner interface + params (the model contract), view window, status builder
  visualizer/   BEV renderer and camera projector interfaces
  utils/        npy reader/writer
src/cpp/
  common/       logger, timer
  latentdrive/  PlannerImpl (TensorRT), FrameWindowImpl / ViewWindowImpl (preprocessing),
                StatusBuilderImpl (route points, motion terms)
  visualizer/   BevVisualizerImpl, CameraProjectorImpl
  main/         gflags mode dispatch (build-engine, run-package, visualize)
scripts/        build.sh (library, tests, ROS 2 package, engines), convert_map.py (lanelet2 map -> route centerlines)
tests/          ctest unit tests (view window, status builder)
samples/
  sample_data/  export_clip.py (rosbag -> clip), run_clip.cpp (latentdrive-sample-clip)
  rosbag/       prepare_bag.sh, run_replay.sh (around ros2/latentdrive_ros)
docker/         pilot-auto.xx1 deployment container (Dockerfile, build/run)
ros2/           ROS 2 open-loop rosbag replay package (latentdrive_ros)
```

Every module is an interface class plus a `CreateX()` factory; the
implementation classes stay inside the .cpp files, so consumers see neither
TensorRT nor CUDA types. GPU resources are held by `common::cuda` unique
pointers (RAII), errors go through the `logger` macros.

## Roadmap

- [x] TensorRT planner wrapper + preprocessing ring buffer
- [x] BEV trajectory visualization
- [x] Camera-frame waypoint overlay
- [x] In-framework engine build from ONNX (TensorRT C++ builder API)
- [x] ROS 2 node — open-loop rosbag replay ([`ros2/latentdrive_ros`](ros2/latentdrive_ros))
- [x] Two-view planner (wide front + telephoto), history by timestamp
- [x] Three-camera planner (camfront3), side views paired within 50 ms, optional blank fill
- [x] Route-point status (six points), motion terms from differenced poses
- [x] Samples: offline clip from a rosbag, live rosbag replay with RViz
