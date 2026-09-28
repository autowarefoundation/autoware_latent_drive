# LatentDrive-TRT

Standalone TensorRT C++ inference framework for the LatentDrive e2e planner
(a V-JEPA2 ViT-L encoder feeding a planner transformer). It is the layer the
ROS 2 node sits on: the library knows nothing about ROS, files or JSON —
it takes camera frames and ego status, and returns a trajectory. It serves the
single-view checkpoints (wide front camera) and the two-view ones (wide front
+ telephoto) from one binary; the engine file decides which.

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
shell it opens.

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
(8 at 2 Hz, 40 at 10 Hz) and the view count (1 or 2 cameras) are read from
the engine at load, so the same binary serves every checkpoint. The fp16
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
| `video` | `[1,2,3,4,256,512]` | two-view: the same for 2 cameras — view 0 = wide front, view 1 = telephoto, frame-aligned in time |
| `status` | `[1,6]` | `[subgoal_x/10, subgoal_y/10, vel_x, vel_y, acc_x, acc_y]`, ego frame |
| `traj` / `plan` | `[1,N,3]` | N waypoints (8 at 2 Hz, 40 at 10 Hz) over t+0.5…4.0 s: `(x fwd [m], y left [m], heading [rad])` |

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
10 Hz budget (single-view; the two-view model runs the encoder twice).

## Two-view mode

The two-view checkpoints (`models/onnx/multi-camera/`) read the wide front
camera and the telephoto camera — on this vehicle
`/sensing/camera/camera1/image_raw/compressed` and
`/sensing/camera/camera2/image_raw/compressed`, both 2880x1860; camera2 has
fx 5448 against camera1's 1486 (HFOV 29.6°). The input contract, matching
`planner_io.py` in the closed-loop hand-off:

- **Preprocessing** is the same for both views (`PreprocessFrame`): keep the
  full width, centre-crop the rows to a 2:1 band, bilinear resize to 512x256,
  /255, ImageNet mean/std, CHW. The telephoto frame is a separate sensor's
  full frame, not a crop of the wide image.
- **History** (`PickHistory`): the wide camera anchors the four slots. For the
  newest wide frame at t, each slot takes the wide frame nearest in *time* to
  t-1.5, t-1.0, t-0.5, t — never by counting frames, so a dropped frame moves
  one slot rather than every older one.
- **Alignment** (`ViewWindow`): for each chosen wide frame, the telephoto frame
  nearest in time, within `view_tolerance_s` (default 0.06 s, what the offline
  alignment used; the cameras are not hardware-synchronised). A slot with no
  telephoto frame inside the tolerance fails the whole selection: that
  inference is **skipped and reported**. No grey, black or duplicated frame is
  ever substituted — the model has only seen real telephoto frames.
- **Layout**: `[1, V, 3, 4, 256, 512]` is V contiguous `[3, 4, H, W]`
  channel-major blocks, view 0 first; `Planner::VideoNumel()` is V times the
  single-view size.

The mode is chosen by configuration alone: the engine's `video` rank sets the
view count (`Planner::NumViews()`), the ROS 2 node subscribes to the telephoto
topic only when that is 2, and the offline tools read the telephoto clip only
then. `tests/test_view_window.cpp` covers the history pick, the tolerance
rejection and the tensor layout (`ctest --test-dir build`).

## ROS 2 interface

```cpp
#include "frame_window.hpp"
#include "planner.hpp"

auto planner = model::planner::CreatePlanner("e2e-fp16-mixed.engine");
preprocess::ViewWindowParams params;
params.num_views = planner->NumViews();          // 1 or 2, from the engine
auto window = preprocess::CreateViewWindow(params);

// camera callbacks (10 Hz): each frame is preprocessed exactly once
window->Push(0, wide_bgr, stamp_s);              // any resolution; crop + resize + normalize
window->Push(1, tele_bgr, stamp_s);              // two-view engines only

// planning tick: pick the history by timestamp, align the views, run
const preprocess::ViewSelection sel = window->Assemble(planner->VideoInput());
if (sel.ok) {                                    // else sel.reason says which view/slot lacks a frame
  common::EgoStatus ego{.subgoal_x_m = sx, .subgoal_y_m = sy,
                        .vel_x = vx, .vel_y = vy, .acc_x = ax, .acc_y = ay};
  common::Trajectory traj = planner->ForwardPrepared(ego.Encode());
}
```

`FrameWindow` (count-based, single view) is still there for the hand-off
package, whose clips define the history by frame index.

The visualizers (`BevVisualizer`, `CameraProjector`) draw on plain `cv::Mat`,
so a node can publish the same canvas as `sensor_msgs/Image`.

## Codebase layout

```
include/
  common/       logger, timer, cuda RAII helpers, trajectory types
  latentdrive/  planner interface + params (the model contract)
  visualizer/   BEV renderer and camera projector interfaces
  utils/        npy reader/writer
src/cpp/
  common/       logger, timer
  latentdrive/  PlannerImpl (TensorRT), FrameWindowImpl / ViewWindowImpl (preprocessing)
  visualizer/   BevVisualizerImpl, CameraProjectorImpl
  main/         gflags mode dispatch (build-engine, run-package, visualize)
tests/          ctest unit tests (history pick, view alignment, tensor layout)
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
