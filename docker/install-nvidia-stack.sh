#!/usr/bin/env bash
# Install the NVIDIA stack that pilot-auto.xx1 beta/v0.53 pins in amd64.env.
#
# ghcr.io/tier4/pilot-auto.xx1/base-image:v0.53 is built by tier4's
# create-base-image workflow with
#     SETUP_ARGS=local_dev_env --no-cuda-drivers --no-nvidia
# which makes ansible skip the cuda / tensorrt / spconv roles entirely. The
# image therefore carries ROS 2 jazzy and every rosdep dependency, but no CUDA,
# no TensorRT and no spconv. This script adds them back at the exact versions
# the branch specifies, so a container built here matches what the vehicle runs.
#
# Mirrors: autoware.dev_env.cuda and .tensorrt (autoware pinned at f508cb1c)
#          plus pilot-auto.xx1's own x2_spconv role.
#
# Usage:
#   Normally run by ./Dockerfile during `docker build`. It can also be run by
#   hand as root inside an already-running container, e.g. when adding the
#   stack to an enroot/pyxis image on a Slurm cluster:
#       enroot start --root --rw --mount .:/setup <container> \
#           bash /setup/install-nvidia-stack.sh
#
# Environment (all optional, defaults track beta/v0.53):
#   CUDA_VERSION        default 12.8
#   TENSORRT_VERSION    default 10.8.0.43-1+cuda12.8
#   CUMM_VERSION        default 0.5.3
#   SPCONV_VERSION      default 2.3.8
#   SPCONV_CUDA_TAG     default cu128-rev1
#
# Note for Jetson/aarch64 targets: arm64.env overrides TensorRT to
# 10.3.0.26-1+cuda12.5, and the spconv debs need the -jetson suffix.
set -euxo pipefail

CUDA_VERSION=${CUDA_VERSION:-12.8}
TENSORRT_VERSION=${TENSORRT_VERSION:-10.8.0.43-1+cuda12.8}
CUMM_VERSION=${CUMM_VERSION:-0.5.3}
SPCONV_VERSION=${SPCONV_VERSION:-2.3.8}
SPCONV_CUDA_TAG=${SPCONV_CUDA_TAG:-cu128-rev1}
CUDA_DASH=${CUDA_VERSION//./-}

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl gnupg

# NVIDIA apt repo. cuda_repo_distro for Ubuntu 24.04 is ubuntu2404 (cuda role default).
curl -fsSL -o /tmp/cuda-keyring.deb \
  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i /tmp/cuda-keyring.deb
rm -f /tmp/cuda-keyring.deb
apt-get update

# --- role: autoware.dev_env.cuda (install_devel=y, cuda_install_drivers=false) ---
# The driver is deliberately NOT installed. It comes from the host at runtime via
# the NVIDIA container runtime; installing nvidia-open here would only conflict.
apt-get install -y --no-install-recommends \
  cuda-command-line-tools-"${CUDA_DASH}" \
  cuda-minimal-build-"${CUDA_DASH}" \
  libcusparse-dev-"${CUDA_DASH}" \
  libcublas-dev-"${CUDA_DASH}" \
  libcurand-dev-"${CUDA_DASH}" \
  cuda-nvml-dev-"${CUDA_DASH}" \
  cuda-nvrtc-dev-"${CUDA_DASH}" \
  libnpp-dev-"${CUDA_DASH}" \
  libnvjpeg-dev-"${CUDA_DASH}"

# --- role: autoware.dev_env.tensorrt ---
# TensorRT 10 dropped libnvparsers (the Caffe parser); only the ONNX parser is left.
apt-get install -y --no-install-recommends --allow-downgrades \
  libnvinfer10="${TENSORRT_VERSION}" \
  libnvinfer-plugin10="${TENSORRT_VERSION}" \
  libnvonnxparsers10="${TENSORRT_VERSION}" \
  libnvinfer-dev="${TENSORRT_VERSION}" \
  libnvinfer-plugin-dev="${TENSORRT_VERSION}" \
  libnvinfer-headers-dev="${TENSORRT_VERSION}" \
  libnvinfer-headers-plugin-dev="${TENSORRT_VERSION}" \
  libnvonnxparsers-dev="${TENSORRT_VERSION}"

apt-mark hold \
  libnvinfer10 libnvinfer-plugin10 libnvonnxparsers10 \
  libnvinfer-dev libnvinfer-plugin-dev libnvinfer-headers-dev \
  libnvinfer-headers-plugin-dev libnvonnxparsers-dev

# --- role: x2_spconv ---
# Upstream autoware.dev_env.spconv skips Ubuntu 24.04 (the cu130-rev1 release is
# not published yet), so pilot-auto.xx1 installs the cu128-rev1 debs itself.
# Needed by the sparse-convolution detectors (CenterPoint, TransFusion, ...).
BASE_URL="https://github.com/autowarefoundation/spconv_cpp/releases/download/spconv_v${SPCONV_VERSION}%2Bcumm_v${CUMM_VERSION}%2B${SPCONV_CUDA_TAG}"
curl -fsSL -o /tmp/cumm.deb   "${BASE_URL}/cumm_${CUMM_VERSION}_amd64.deb"
curl -fsSL -o /tmp/spconv.deb "${BASE_URL}/spconv_${SPCONV_VERSION}_amd64.deb"
apt-get install -y /tmp/cumm.deb /tmp/spconv.deb
rm -f /tmp/cumm.deb /tmp/spconv.deb

# --- GPU vendor ICDs (from the cuda role), so OpenGL / OpenCL see the GPU ---
mkdir -p /etc/vulkan/icd.d /etc/glvnd/egl_vendor.d /etc/OpenCL/vendors
curl -fsSL -o /etc/vulkan/icd.d/nvidia_icd.json \
  https://gitlab.com/nvidia/container-images/vulkan/raw/dc389b0445c788901fda1d85be96fd1cb9410164/nvidia_icd.json
curl -fsSL -o /etc/glvnd/egl_vendor.d/10_nvidia.json \
  https://gitlab.com/nvidia/container-images/opengl/raw/5191cf205d3e4bb1150091f9464499b076104354/glvnd/runtime/10_nvidia.json
echo libnvidia-opencl.so.1 > /etc/OpenCL/vendors/nvidia.icd

# --- env: login shells, interactive shells, and non-interactive exec ---
cat > /etc/profile.d/zz-cuda.sh <<'EOF'
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
EOF
chmod 644 /etc/profile.d/zz-cuda.sh
# The base image's Dockerfile truncates /etc/bash.bashrc to just the ROS source
# line (`>` not `>>`), so append rather than overwrite.
cat >> /etc/bash.bashrc <<'EOF'
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
EOF

# /etc/environment is what enroot reads for the container environment. Harmless
# under docker (the Dockerfile's ENV covers that path), required under enroot:
# its 98-nvidia.sh hook injects no driver at all unless NVIDIA_VISIBLE_DEVICES
# is set, leaving the container without even /dev/nvidia*.
cat >> /etc/environment <<'EOF'
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,video
PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
LD_LIBRARY_PATH=/usr/local/cuda/lib64
EOF

apt-get clean
rm -rf /var/lib/apt/lists/*

echo "=== installed ==="
/usr/local/cuda/bin/nvcc --version | tail -2
dpkg -l | grep -E "libnvinfer|libnvonnxparsers|^ii  spconv|^ii  cumm" | awk '{print $2, $3}'
