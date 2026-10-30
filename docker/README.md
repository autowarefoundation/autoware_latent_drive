# Docker environment

LatentDrive-TRT builds and runs inside the **pilot-auto.xx1** deployment
container, so the engine is built and executed against the exact CUDA 12.8 /
TensorRT 10.8 libraries the vehicle runs. This is the recommended (and only
supported) way to build and run the framework.

| | |
|---|---|
| Base | `ghcr.io/tier4/pilot-auto.xx1/base-image:v0.53` |
| OS / ROS | Ubuntu 24.04 / ROS 2 jazzy |
| CUDA | 12.8 |
| TensorRT | 10.8.0.43 |

The base image ships ROS 2 and every rosdep (OpenCV 4, gflags, nlohmann-json,
CMake, g++ 13 are all present) but no GPU stack; `install-nvidia-stack.sh` adds
CUDA and TensorRT at the versions `amd64.env` pins. It is the same image the
training repo's `deployment/` builds — if you already have
`pilot-auto-xx1:v0.53-cuda`, reuse it and skip step 1.

## Prerequisites

- Docker + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  (`docker run --rm --gpus all ubuntu nvidia-smi` must work)
- A GHCR credential that can pull `tier4/pilot-auto.xx1` (the base image is
  private): `gh auth login`, or `docker login ghcr.io`. `build.sh` picks it up.
- Host NVIDIA driver new enough for CUDA 12.8 (>= 570 native, >= 525 with
  forward compat). The container ships no driver — it comes from the host.

## 1. Build the image (once)

```bash
./docker/build.sh                 # -> pilot-auto-xx1:v0.53-cuda
```

## 2. Enter the container

`run.sh` mounts this repo at `/workspace` and the hand-off data package
(checkpoint / ONNX / `samples/` / `sequence/` / `calibration.json`) at `/data`:

```bash
DATA=/path/to/workspace ./docker/run.sh          # interactive shell at /workspace
```

`DATA` is the directory holding the ONNX and the test package (in the training
repo that is `JEPA-World-model/workspace/`). Everything below runs from the
shell this opens; `NAME=ltrt` keeps the container alive between calls.

## 3. Compile the framework

```bash
cmake -S core -B build && cmake --build build -j
```

Produces `build/bin/latentdrive-trt`. One binary, three `--mode`s. The unit
tests (history pick, wide/telephoto alignment, tensor layout; no GPU needed)
build alongside it:

```bash
ctest --test-dir build --output-on-failure
```

## 4. Build a TensorRT engine from ONNX

```bash
./build/bin/latentdrive-trt --mode=build-engine \
    --onnx_file=/data/e2e-idx8-hist4-wp10hz-e3.onnx \
    --engine_file=/data/e2e-wp10hz-fp16-mixed.engine
```

fp16 with the planner stage pinned to fp32 (the validated recipe). `--prec=fp32`
builds the control engine; `--pure_fp16` disables the pin (fails validation).
Takes ~1 min on an RTX 4090.

The same command builds the two-view export
(`models/onnx/multi-camera/closed-loop-best/onnx/idx9-far-ego-jepa9-e1.onnx`,
`video [batch, 2, 3, 4, 256, 512]`): its symbolic batch axis is frozen to 1 by
an optimization profile, and the build log says `Video input is 6-D: 2 camera
views`. Everything downstream (`run-package`, `visualize`, the ROS 2 node)
reads the view count from the engine.

## 5. Validate + benchmark

```bash
./build/bin/latentdrive-trt --mode=run-package \
    --engine_file=/data/e2e-wp10hz-fp16-mixed.engine --package_dir=/data \
    --ref_name=ref_output_fp32_wp10hz.npy --out_name=engine_output_wp10hz.npy
```

Runs the 9 alignment samples, the preprocessing self-test and both streaming
clips, reports per-step deviation vs the PyTorch reference and the batch-1
latency (engine + preprocessing). `--ref_name` selects the reference set
(defaults to `ref_output_fp32.npy`, the 8-waypoint model); omit the wp10hz name
for the 2 Hz checkpoint. Score against the package's own `validate.py`:

```bash
# on the host, in the data package dir
python3 validate.py --engine-file engine_output_wp10hz.npy
```

## 6. Visualize

```bash
./build/bin/latentdrive-trt --mode=visualize \
    --engine_file=/data/e2e-wp10hz-fp16-mixed.engine --package_dir=/data \
    --clip=follow_lane --ref_name=ref_output_fp32_wp10hz.npy \
    --out_video=/workspace/viz/follow_lane.mp4
```

Writes an mp4 (camera panel with the rainbow prediction ribbon + ground-truth
line + a deployment HUD showing live latency and accuracy, beside the BEV with
the subgoal). `--out_video` under `/workspace` lands in the repo on the host.
`--clip=stop_to_launch` for the other clip; `--png_every=N` also dumps stills.

## Notes

- **Files written inside the container are root-owned** on the host (the image
  has no matching user). `sudo chown -R $USER:$USER build viz` if that gets in
  the way, or `rm -rf build` from inside the container before reconfiguring.
- **A `build/` from a different mount path** carries stale absolute paths in its
  CMake cache. If you previously built with the repo mounted elsewhere, wipe
  `build/` before the first `cmake -S core -B build` here.
- **A `build/` from before the move of `CMakeLists.txt` to `core/`** points at
  the old source directory, and CMake refuses it. Wipe `build/` once.
- The ONNX itself is produced by the training repo (`deployment/export_onnx.py`);
  this container only consumes it. Everything from ONNX onward stays in C++.
