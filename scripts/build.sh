#!/usr/bin/env bash
# Build everything needed to run the samples: the library and its apps
# (build/), the unit tests, the autoware_latent_drive package (ros2/install/) and
# one mixed-precision engine per ONNX export (models/engine/).
#
#   scripts/build.sh [onnx ...]
#
#   scripts/build.sh                                  # camfront3-rptw-lane2-e1.onnx, found under models/onnx/
#   scripts/build.sh models/onnx/single-camera/.../camfront-rptw-lane2-e1.onnx
#   SKIP_ROS=1 scripts/build.sh                       # no ROS install: library + engines only
#
# An engine is written next to the others as models/engine/<onnx name>-mixed.engine
# (fp16, planner stage pinned to fp32), with the builder's output in
# <name>-mixed.build.log. An existing engine is kept unless FORCE=1: engines are
# tied to the GPU (and TensorRT version) they were built on, so rebuild them on
# a new machine. Three views take ~7 min, one view ~1.5 min.
#
#   BUILD_DIR   CMake build directory, default <repo>/build
#   ENGINE_DIR  default <repo>/models/engine
#   SKIP_ROS    1: skip the colcon build
#   SKIP_TESTS  1: skip ctest
#   FORCE       1: rebuild engines that already exist
set -eo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
BUILD_DIR=${BUILD_DIR:-$REPO/build}
ENGINE_DIR=${ENGINE_DIR:-$REPO/models/engine}
DEFAULT_ONNX_NAME=camfront3-rptw-lane2-e1.onnx
JOBS=$(nproc)

step() { echo -e "\n=== $*"; }

# Check the ONNX files before spending minutes on the builds. The default is
# looked up by name anywhere under models/onnx/, whose layout varies by machine
ONNX=("$@")
if [ ${#ONNX[@]} -eq 0 ]; then
  default=$(find -L "$REPO/models/onnx" -name "$DEFAULT_ONNX_NAME" 2>/dev/null | head -1)
  [ -n "$default" ] || { echo "no $DEFAULT_ONNX_NAME under $REPO/models/onnx: pass the ONNX files to build" >&2; exit 1; }
  ONNX=("$default")
fi
for onnx in "${ONNX[@]}"; do
  [ -f "$onnx" ] || { echo "no ONNX at $onnx" >&2; exit 1; }
done

step "library and apps -> $BUILD_DIR"
cmake -S "$REPO/core" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" -j "$JOBS"

if [ "${SKIP_TESTS:-0}" != 1 ]; then
  step "unit tests"
  (cd "$BUILD_DIR" && ctest --output-on-failure)
fi

if [ "${SKIP_ROS:-0}" != 1 ]; then
  step "autoware_latent_drive -> $REPO/ros2/install"
  # ROS_DISTRO alone is no sign of a sourced ROS: the Autoware images set it
  if [ -z "${AMENT_PREFIX_PATH:-}" ]; then
    setup=/opt/ros/${ROS_DISTRO:-}/setup.bash
    [ -f "$setup" ] || setup=$(ls /opt/ros/*/setup.bash 2>/dev/null | head -1)
    [ -n "$setup" ] || { echo "no ROS 2 under /opt/ros (SKIP_ROS=1 to build without it)" >&2; exit 1; }
    # shellcheck disable=SC1090
    source "$setup"
  fi
  (cd "$REPO/ros2" && colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release)
fi

mkdir -p "$ENGINE_DIR"
for onnx in "${ONNX[@]}"; do
  engine=$ENGINE_DIR/$(basename "$onnx" .onnx)-mixed.engine
  if [ -f "$engine" ] && [ "${FORCE:-0}" != 1 ]; then
    step "engine $engine exists (FORCE=1 to rebuild)"
    continue
  fi
  step "engine $onnx -> $engine"
  start=$(date +%s)
  "$BUILD_DIR/bin/latentdrive-trt" --mode=build-engine --onnx_file="$onnx" --engine_file="$engine" \
    2>&1 | tee "${engine%.engine}.build.log"
  echo "built in $(( $(date +%s) - start )) s"
done

step "done"
echo "engines in $ENGINE_DIR:"
ls -1 "$ENGINE_DIR"/*.engine
