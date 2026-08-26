# LatentDrive-TRT

Standalone TensorRT C++ inference framework for the LatentDrive e2e planner
(`e2e-idx8-hist4`: V-JEPA2 ViT-L encoder + planner transformer). This is the
layer a ROS 2 node will sit on: the library knows nothing about ROS, files, or
JSON — it takes camera frames and ego status, and returns a trajectory.

## Model contract

| tensor | shape | meaning |
|---|---|---|
| `video` | `[1,3,4,256,512]` | 4 front-camera frames, oldest→newest, 0.5 s apart, RGB, ImageNet-normalized |
| `status` | `[1,6]` | `[subgoal_x/10, subgoal_y/10, vel_x, vel_y, acc_x, acc_y]`, ego frame |
| `traj` | `[1,8,3]` | 8 waypoints at t+0.5…4.0 s: `(x fwd [m], y left [m], heading [rad])` |

The engine is produced by `deployment/` in the training repo
(`kalfazed/JEPA-World-model`): ONNX opset 17 → `trtexec --fp16` with the
planner stage pinned to fp32 (`build_engine.sh`). Validated 11/11 against the
PyTorch reference; ~5 ms engine call / ~7 ms end-to-end per step on an
RTX 4090 — 8% of the 10 Hz budget.

## Interface

```cpp
#include <latentdrive/planner.hpp>
#include <latentdrive/frame_window.hpp>

latentdrive::Planner planner("e2e-fp16-mixed.engine");
latentdrive::FrameWindow window;              // 16-frame ring buffer @ 10 Hz

// camera callback (10 Hz): preprocess each frame exactly once
window.push(bgr_frame);                       // any resolution; crop+resize+normalize

// planning tick: assemble the 4-slot window and run the engine
if (window.ready()) {
  window.assemble(planner.videoInput());      // zero-copy into pinned memory
  latentdrive::EgoStatus ego{.subgoal_x_m = sx, .subgoal_y_m = sy,
                             .vel_x = vx, .vel_y = vy, .acc_x = ax, .acc_y = ay};
  latentdrive::Trajectory traj = planner.inferPrepared(ego.encode());
}
```

`Planner` hides all TensorRT/CUDA types behind a pImpl, so consumers only link
`latentdrive` and include these headers. Layout:

```
include/latentdrive/   public interface (types.hpp, planner.hpp, frame_window.hpp)
src/                   library implementation + internal utils (npy.hpp)
src/apps/              CLI tools (run_package)
```

## Build

Dependencies: CUDA 12.x, TensorRT 10.x, OpenCV 4, nlohmann-json (apps only).
The deployment container from the training repo has all of them:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release   # -DTENSORRT_ROOT=... for tarball installs
cmake --build build -j
```

## Validate & benchmark against the hand-off package

`run_package` replays the `workspace/` test package from the training repo —
9 alignment samples, a preprocessing self-test, 2 streaming clips, and the
batch-1 latency benchmark — and writes the `.npy` outputs its `validate.py`
scores:

```bash
./build/run_package --engine e2e-fp16-mixed.engine --package /path/to/workspace
(cd /path/to/workspace && python3 validate.py)     # expect 11/11
```

## Visualization

`BevVisualizer` renders trajectories in a bird's-eye view (x forward = up,
dark theme matching the benchmark report): prediction vs PyTorch reference vs
ground truth, subgoal marker, velocity arrow, per-waypoint heading ticks. It
is pure drawing (`cv::Mat` in, `cv::Mat` out) so a ROS 2 node can publish the
same canvas as `sensor_msgs/Image`.

`visualize_sequence` streams a package clip and writes camera + HUD + BEV
as an mp4 (PNG fallback when OpenCV lacks an encoder):

```bash
./build/visualize_sequence --engine e2e-fp16-mixed.engine \
    --package /path/to/workspace --clip stop_to_launch \
    --out viz/stop_to_launch.mp4 --png-every 40
```

## Roadmap

- [x] TensorRT planner wrapper + preprocessing ring buffer
- [x] BEV trajectory visualization
- [ ] Camera-frame waypoint overlay (needs camera calibration)
- [ ] ROS 2 node
