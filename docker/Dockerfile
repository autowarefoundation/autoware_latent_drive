# Deployment image for the JEPA world-model E2E node, matching pilot-auto.xx1
# beta/v0.53.
#
# The base carries ROS 2 jazzy on Ubuntu 24.04 with every rosdep dependency of
# the branch, but tier4 builds it with `--no-nvidia`, so it ships no CUDA, no
# TensorRT and no spconv. install-nvidia-stack.sh adds them at the versions
# amd64.env pins: CUDA 12.8 / TensorRT 10.8.0.43 / spconv 2.3.8.
#
# The base image is private; `docker login ghcr.io` with a token that can pull
# from tier4/pilot-auto.xx1 before building. See ./build.sh.
FROM ghcr.io/tier4/pilot-auto.xx1/base-image:v0.53@sha256:0b757b93965c27f4ffc7144609168978ef91d0a1de900e5437ea11a3f82a2d6a

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Defaults live in install-nvidia-stack.sh; override here to retarget a version,
# e.g. --build-arg TENSORRT_VERSION=10.3.0.26-1+cuda12.5 for Jetson/arm64.
ARG CUDA_VERSION
ARG TENSORRT_VERSION
ARG CUMM_VERSION
ARG SPCONV_VERSION
ARG SPCONV_CUDA_TAG

COPY install-nvidia-stack.sh /tmp/install-nvidia-stack.sh
RUN CUDA_VERSION=${CUDA_VERSION:-12.8} \
    TENSORRT_VERSION=${TENSORRT_VERSION:-10.8.0.43-1+cuda12.8} \
    CUMM_VERSION=${CUMM_VERSION:-0.5.3} \
    SPCONV_VERSION=${SPCONV_VERSION:-2.3.8} \
    SPCONV_CUDA_TAG=${SPCONV_CUDA_TAG:-cu128-rev1} \
    bash /tmp/install-nvidia-stack.sh \
  && rm -f /tmp/install-nvidia-stack.sh

ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# Read by the NVIDIA container runtime (`docker run --gpus`) and by enroot's
# 98-nvidia.sh hook, which injects nothing when they are unset.
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,video

WORKDIR /workspace
CMD ["/bin/bash"]
