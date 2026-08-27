# LatentDrive-TRT

Standalone TensorRT C++ inference framework for the LatentDrive e2e planner
(`e2e-idx8-hist4`: V-JEPA2 ViT-L encoder + planner transformer). This is the
layer the ROS 2 node will sit on: the library knows nothing about ROS, files
or JSON — it takes camera frames and ego status, and returns a trajectory.

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
    --clip=follow_lane --out_video=viz/follow_lane.mp4
```

`run-package` writes the `.npy` outputs the package's `validate.py` scores
(`python3 validate.py` → 11/11 for the 8-waypoint model). The engine build
reproduces the validated trtexec recipe in the TensorRT C++ API: the planner
stage (`kFp32LayerPrefixes` in `planner_params.hpp`) is held at fp32 under
`kOBEY_PRECISION_CONSTRAINTS`, and index/shape layers are skipped since they
reject a float constraint. The waypoint count (8 at 2 Hz, 40 at 10 Hz) is read
from the engine at load, so the same binary serves both checkpoints.

## Model contract

| tensor | shape | meaning |
|---|---|---|
| `video` | `[1,3,4,256,512]` | 4 front-camera frames, oldest→newest, 0.5 s apart, RGB, ImageNet-normalized |
| `status` | `[1,6]` | `[subgoal_x/10, subgoal_y/10, vel_x, vel_y, acc_x, acc_y]`, ego frame |
| `traj` | `[1,N,3]` | N waypoints (8 at 2 Hz, 40 at 10 Hz) over t+0.5…4.0 s: `(x fwd [m], y left [m], heading [rad])` |

The ONNX comes from `deployment/export_onnx.py` in the training repo
(`kalfazed/LatentDrive`, opset 17); the engine builds either there via
`trtexec` or here via `--mode=build-engine` (TensorRT C++ builder API). Both
apply the same precision recipe — fp16 with the planner stage pinned to fp32 —
and produce numerically identical results: validated 11/11 against the
PyTorch reference, ~4.7 ms engine call / ~5.8 ms end-to-end per step on an
RTX 4090 — 6% of the 10 Hz budget.

## ROS 2 interface

```cpp
#include "frame_window.hpp"
#include "planner.hpp"

auto planner = model::planner::CreatePlanner("e2e-fp16-mixed.engine");
auto window  = preprocess::CreateFrameWindow();  // 16-frame ring buffer @ 10 Hz

// camera callback (10 Hz): each frame is preprocessed exactly once
window->Push(bgr_frame);          // any resolution; crop + resize + normalize

// planning tick: assemble the 4-slot window and run the engine
if (window->Ready()) {
  window->Assemble(planner->VideoInput());   // zero-copy into pinned memory
  common::EgoStatus ego{.subgoal_x_m = sx, .subgoal_y_m = sy,
                        .vel_x = vx, .vel_y = vy, .acc_x = ax, .acc_y = ay};
  common::Trajectory traj = planner->ForwardPrepared(ego.Encode());
}
```

`visualizer::CameraProjector` overlays the plan on the camera image (pinhole,
T4 calibrated_sensor convention, mandatory 0.5 m depth gate — at a standstill
the whole plan sits behind the camera and is correctly not drawn):
`DrawRibbon` fills a ground-plane corridor colored along the horizon
(green = now → red = +4 s, the `HorizonColor` ramp), `DrawTrajectory` adds the
thin white ground-truth line. `visualizer::BevVisualizer` renders the same plan
in a bird's-eye view with `DrawTrajectoryHorizon` (matching gradient), plus the
subgoal marker and velocity arrow. Both draw on plain `cv::Mat`, so the ROS 2
node can publish the same canvases as `sensor_msgs/Image`.

## Codebase layout

```
include/
  common/       logger, timer, cuda RAII helpers, trajectory types
  latentdrive/  planner interface + params (the model contract)
  visualizer/   BEV renderer and camera projector interfaces
  utils/        npy reader/writer
src/cpp/
  common/       logger, timer
  latentdrive/  PlannerImpl (TensorRT), FrameWindowImpl (preprocessing)
  visualizer/   BevVisualizerImpl, CameraProjectorImpl
  main/         gflags mode dispatch (build-engine, run-package, visualize)
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
