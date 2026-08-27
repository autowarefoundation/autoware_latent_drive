#!/usr/bin/env bash
# Build the deployment image (pilot-auto.xx1 beta/v0.53 + CUDA 12.8 / TensorRT 10.8).
#
# Usage:
#   ./docker/build.sh                          # build pilot-auto-xx1:v0.53-cuda
#   ./docker/build.sh --no-cache               # extra args go to docker build
#
# The same image backs the training repo's deployment/ container; if you already
# built it there, skip this and reuse it -- run.sh defaults to the same tag.
#
# Environment:
#   IMAGE          image tag to build (default pilot-auto-xx1:v0.53-cuda)
#   GITHUB_TOKEN   token used to log in to ghcr.io; falls back to `gh auth token`.
#                  Skipped entirely if you are already logged in.
#
# The base image is private to tier4, so the daemon needs a GHCR credential that
# can pull tier4/pilot-auto.xx1. A classic PAT or a gh CLI token both work.
set -euo pipefail
cd "$(dirname "$0")"

IMAGE=${IMAGE:-pilot-auto-xx1:v0.53-cuda}

if ! docker info >/dev/null 2>&1; then
    echo "error: cannot talk to the docker daemon. Is it running, and is your user in the docker group?" >&2
    exit 1
fi

# Log in only if we cannot already read the private base image's manifest.
if ! docker manifest inspect ghcr.io/tier4/pilot-auto.xx1/base-image:v0.53 >/dev/null 2>&1; then
    token=${GITHUB_TOKEN:-$(gh auth token 2>/dev/null || true)}
    if [ -z "$token" ]; then
        echo "error: no GHCR credential. Set GITHUB_TOKEN, or run 'gh auth login', or 'docker login ghcr.io'." >&2
        exit 1
    fi
    echo "$token" | docker login ghcr.io --username "$(id -un)" --password-stdin
fi

set -x
docker build -t "$IMAGE" -f Dockerfile "$@" .
set +x

echo
echo "Built $IMAGE. Sanity check:"
echo "  docker run --rm --gpus all $IMAGE bash -lc 'nvcc --version | tail -1; nvidia-smi -L'"
