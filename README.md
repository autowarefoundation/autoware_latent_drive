# LatentDrive-TRT

Standalone TensorRT C++ inference framework for the LatentDrive e2e planner
(`e2e-idx8-hist4`: V-JEPA2 ViT-L encoder + planner transformer). This is the
layer the ROS 2 node will sit on: the library knows nothing about ROS, files
or JSON — it takes camera frames and ego status, and returns a trajectory.

## Model contract

| tensor | shape | meaning |
|---|---|---|
| `video` | `[1,3,4,256,512]` | 4 front-camera frames, oldest→newest, 0.5 s apart, RGB, ImageNet-normalized |
| `status` | `[1,6]` | `[subgoal_x/10, subgoal_y/10, vel_x, vel_y, acc_x, acc_y]`, ego frame |
| `traj` | `[1,8,3]` | 8 waypoints at t+0.5…4.0 s: `(x fwd [m], y left [m], heading [rad])` |

The ONNX comes from `deployment/export_onnx.py` in the training repo
(`kalfazed/JEPA-World-model`, opset 17); the engine builds either there via
`trtexec` or here via `--mode=build-engine` (TensorRT C++ builder API). Both
apply the same precision recipe — fp16 with the planner stage pinned to fp32 —
and produce numerically identical results: validated 11/11 against the
PyTorch reference, ~4.7 ms engine call / ~5.8 ms end-to-end per step on an
RTX 4090 — 6% of the 10 Hz budget.

## Layout

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
  main/         gflags mode dispatch (run-package, visualize)
```

Every module is an interface class plus a `CreateX()` factory; the
implementation classes stay inside the .cpp files, so consumers see neither
TensorRT nor CUDA types. GPU resources are held by `common::cuda` unique
pointers (RAII), errors go through the `logger` macros.

## Interface

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

## Build

Dependencies: CUDA 12.x, TensorRT 10.x, OpenCV 4, gflags, nlohmann-json.
The deployment container from the training repo has all of them:

```bash
cmake -B build     # -DTENSORRT_ROOT=... for tarball installs
cmake --build build -j
```

## Build the engine from ONNX

`build-engine` parses the ONNX export with nvonnxparser and builds through
`IBuilder` (TensorRT 10 API). The fp16 build pins the planner-stage layers
(`kFp32LayerPrefixes` in `planner_params.hpp`) to fp32 under
`kOBEY_PRECISION_CONSTRAINTS` — the in-framework equivalent of the validated
trtexec recipe; index/shape layers are skipped since they reject a float
constraint:

```bash
./build/bin/latentdrive-trt --mode=build-engine \
    --onnx_file=e2e-idx8-hist4-e3.onnx --engine_file=e2e-fp16-mixed.engine
    # --prec=fp32 for the control engine; --pure_fp16 disables the pin (fails validation)
```

## Run against the hand-off package

`run-package` replays the `workspace/` test package from the training repo —
9 alignment samples, a preprocessing self-test, 2 streaming clips and the
batch-1 latency benchmark — and writes the `.npy` outputs its `validate.py`
scores:

```bash
./build/bin/latentdrive-trt --mode=run-package \
    --engine_file=e2e-fp16-mixed.engine --package_dir=/path/to/workspace
(cd /path/to/workspace && python3 validate.py)     # expect 11/11
```

`visualize` streams one clip and writes camera (with the waypoint overlay when
`calibration.json` covers the scene) + HUD + BEV as an mp4:

```bash
./build/bin/latentdrive-trt --mode=visualize \
    --engine_file=e2e-fp16-mixed.engine --package_dir=/path/to/workspace \
    --clip=stop_to_launch --out_video=viz/stop_to_launch.mp4 --png_every=40
```

## Roadmap

- [x] TensorRT planner wrapper + preprocessing ring buffer
- [x] BEV trajectory visualization
- [x] Camera-frame waypoint overlay
- [x] In-framework engine build from ONNX (TensorRT C++ builder API)
- [ ] ROS 2 node
