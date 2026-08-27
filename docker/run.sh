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
#   GPUS    value for --gpus (default all)
#   NAME    container name; when set the container is kept after exit and a
#           second invocation execs into the running one. Unset means --rm.
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
    -v "$REPO:/workspace"
    -w /workspace
)

if [ -n "${DATA:-}" ]; then
    args+=(-v "$DATA:/data")
else
    echo "note: DATA is unset; the build-engine/run-package/visualize commands" >&2
    echo "      need the hand-off package mounted, e.g. DATA=/path/to/workspace" >&2
fi

if [ -n "${NAME:-}" ]; then
    args+=(--name "$NAME")
else
    args+=(--rm)
fi

tty_args=(-i)
[ -t 0 ] && tty_args+=(-t) # no -t when piped, so this works from scripts too

exec docker run "${tty_args[@]}" "${args[@]}" "$IMAGE" "${@:-bash}"
