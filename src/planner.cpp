#include "latentdrive/planner.hpp"

#include <cuda_runtime_api.h>

#include <NvInfer.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace latentdrive {
namespace {

void cudaCheck(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
  }
}

class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cerr << "[TRT] " << msg << "\n";
    }
  }
};

Logger& logger() {
  static Logger instance;
  return instance;
}

// Pinned host + device pair for one IO tensor.
struct Buffer {
  float* host = nullptr;
  void* device = nullptr;
  std::size_t bytes = 0;

  void alloc(std::size_t numel) {
    bytes = numel * sizeof(float);
    cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&host), bytes), "cudaMallocHost");
    cudaCheck(cudaMalloc(&device, bytes), "cudaMalloc");
  }
  void free() {
    if (host) cudaFreeHost(host);
    if (device) cudaFree(device);
    host = nullptr;
    device = nullptr;
  }
};

}  // namespace

struct Planner::Impl {
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  cudaStream_t stream = nullptr;
  Buffer video, status, traj;

  ~Impl() {
    video.free();
    status.free();
    traj.free();
    if (stream) cudaStreamDestroy(stream);
  }
};

Planner::Planner(const std::string& engine_path) : impl_(std::make_unique<Impl>()) {
  std::ifstream f(engine_path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open engine: " + engine_path);
  const std::streamsize size = f.tellg();
  f.seekg(0);
  std::vector<char> blob(static_cast<std::size_t>(size));
  if (!f.read(blob.data(), size)) throw std::runtime_error("short read: " + engine_path);

  impl_->runtime.reset(nvinfer1::createInferRuntime(logger()));
  if (!impl_->runtime) throw std::runtime_error("createInferRuntime failed");
  impl_->engine.reset(impl_->runtime->deserializeCudaEngine(blob.data(), blob.size()));
  if (!impl_->engine) throw std::runtime_error("deserializeCudaEngine failed: " + engine_path);
  impl_->context.reset(impl_->engine->createExecutionContext());
  if (!impl_->context) throw std::runtime_error("createExecutionContext failed");

  cudaCheck(cudaStreamCreate(&impl_->stream), "cudaStreamCreate");
  impl_->video.alloc(kVideoNumel);
  impl_->status.alloc(kStatusDim);
  impl_->traj.alloc(kNumWaypoints * 3);

  // The export fixed these names; fail loudly if handed some other engine.
  for (const char* name : {"video", "status", "traj"}) {
    if (impl_->engine->getTensorShape(name).nbDims < 0) {
      throw std::runtime_error(std::string("engine has no tensor '") + name +
                               "' -- not a LatentDrive planner engine?");
    }
  }
  impl_->context->setTensorAddress("video", impl_->video.device);
  impl_->context->setTensorAddress("status", impl_->status.device);
  impl_->context->setTensorAddress("traj", impl_->traj.device);
}

Planner::~Planner() = default;

float* Planner::videoInput() { return impl_->video.host; }

Trajectory Planner::inferPrepared(const std::array<float, kStatusDim>& status) {
  auto& im = *impl_;
  std::memcpy(im.status.host, status.data(), im.status.bytes);
  cudaCheck(cudaMemcpyAsync(im.video.device, im.video.host, im.video.bytes,
                            cudaMemcpyHostToDevice, im.stream), "H2D video");
  cudaCheck(cudaMemcpyAsync(im.status.device, im.status.host, im.status.bytes,
                            cudaMemcpyHostToDevice, im.stream), "H2D status");
  if (!im.context->enqueueV3(im.stream)) throw std::runtime_error("enqueueV3 failed");
  cudaCheck(cudaMemcpyAsync(im.traj.host, im.traj.device, im.traj.bytes,
                            cudaMemcpyDeviceToHost, im.stream), "D2H traj");
  cudaCheck(cudaStreamSynchronize(im.stream), "stream sync");

  Trajectory out;
  static_assert(sizeof(Trajectory) == kNumWaypoints * 3 * sizeof(float));
  std::memcpy(out.data(), im.traj.host, sizeof(Trajectory));
  return out;
}

Trajectory Planner::infer(const float* video, const std::array<float, kStatusDim>& status) {
  std::memcpy(impl_->video.host, video, impl_->video.bytes);
  return inferPrepared(status);
}

}  // namespace latentdrive
