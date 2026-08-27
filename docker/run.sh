#!/usr/bin/env bash
# Run the LatentDrive-TRT workflow inside the pilot-auto.xx1 deployment
# container. Mounts this repo at /workspace and the hand-off data package
# (checkpoint / ONNX / samples / sequences / calibration.json) at /data.
#
# Usage:
#   DATA=/path/to/workspace ./docker/run.sh                 # interactive shell
#   DATA=/path/to/workspace ./docker/run.sh cmake --build build   # one command
#   NAME=ltrt DATA=/path/to/workspace ./docker/run.sh       # persistent container
#
# Environment:
#   IMAGE   image to run (default pilot-auto-xx1:v0.53-cuda)
#   REPO    host dir mounted at /workspace (default: this repository)
#   DATA    host dir mounted at /data (the hand-off package); required for the
#           build-engine / run-package / visualize commands
#   BAG     rosbag dir mounted at /bag (for the ROS 2 replay node)
#   GPUS    value for --gpus (default all)
#   NAME    container name; when set the container is kept after exit and a
#           second invocation execs into the running one. Unset means --rm.
#   NO_X11=1  skip X11 forwarding (no GUI windows like rqt_image_view)
#
# For live GUI windows (rqt_image_view) run `xhost +local:docker` on the host
# once, then start with X11 forwarded (the default here).
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE=${IMAGE:-pilot-auto-xx1:v0.53-cuda}
REPO=${REPO:-$PWD}
GPUS=${GPUS:-all}

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "error: image '$IMAGE' not found. Build it first: ./docker/build.sh" >&2
    exit 1
fi

# Reuse an already-running named container instead of starting a second one.
if [ -n "${NAME:-}" ] && [ -n "$(docker ps -q -f "name=^${NAME}$")" ]; then
    exec docker exec -it "$NAME" "${@:-bash}"
fi

args=(
    --gpus "$GPUS"
    # ROS 2 nodes + rqt talk over localhost/shared-memory DDS; host net + shm
    # keep discovery and large image transport happy
    --network host
    --ipc host
    --shm-size 8g
    -v "$REPO:/workspace"
    -w /workspace
)

if [ -n "${DATA:-}" ]; then
    args+=(-v "$DATA:/data")
else
    echo "note: DATA is unset; the build-engine/run-package/visualize commands" >&2
    echo "      need the hand-off package mounted, e.g. DATA=/path/to/workspace" >&2
fi

if [ -n "${BAG:-}" ]; then
    args+=(-v "$BAG:/bag")
fi

# X11 for GUI windows (rqt_image_view). Run `xhost +local:docker` once on the
# host if the display is refused.
if [ -z "${NO_X11:-}" ] && [ -n "${DISPLAY:-}" ]; then
    args+=(-e "DISPLAY=$DISPLAY" -v /tmp/.X11-unix:/tmp/.X11-unix:ro)
    if [ -n "${XAUTHORITY:-}" ] && [ -f "$XAUTHORITY" ]; then
        args+=(-e XAUTHORITY=/tmp/.docker.xauth -v "$XAUTHORITY:/tmp/.docker.xauth:ro")
    fi
fi

if [ -n "${NAME:-}" ]; then
    args+=(--name "$NAME")
else
    args+=(--rm)
fi

tty_args=(-i)
[ -t 0 ] && tty_args+=(-t) # no -t when piped, so this works from scripts too

exec docker run "${tty_args[@]}" "${args[@]}" "$IMAGE" "${@:-bash}"
