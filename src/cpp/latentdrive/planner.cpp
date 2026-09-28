#include "planner.hpp"

#include <cstring>
#include <string>

#include "NvInfer.h"
#include "logger.hpp"
#include "utils.hpp"

namespace model {
namespace planner {

// @brief: This function is used to print a Dims as "[1, 2, 3]" for the load
// diagnostics; a -1 (symbolic axis) prints as such
static std::string DimsToString(const nvinfer1::Dims &dims) {
  std::string out = "[";
  for (int i = 0; i < dims.nbDims; ++i) {
    out += (i ? ", " : "") + std::to_string(dims.d[i]);
  }
  return out + "]";
}

// `PlannerImpl` is an implementation class, which implement the detailed
// `Forward`. It owns the deserialized engine, one CUDA stream and one pinned
// host + device buffer pair per IO tensor. All GPU resources are released
// through RAII (common::cuda unique pointers), the stream in the destructor
class PlannerImpl : public Planner {
 public:
  explicit PlannerImpl() {}

  PlannerImpl(const PlannerImpl &)            = delete;
  PlannerImpl &operator=(const PlannerImpl &) = delete;

  // @brief Device synchronization before the RAII members release the
  // engine and the buffers
  ~PlannerImpl() override {
    if (stream_) {
      cudaStreamSynchronize(stream_);
      cudaStreamDestroy(stream_);
    }
  }

  // @brief Deserialize the engine, allocate the IO buffers and bind their
  // device addresses to the execution context
  bool Init(const std::string &engine_path) {
    logger_ = logger::CreateLogger(logger::Level::kINFO);

    auto engine_data = common::LoadFile(engine_path);
    if (engine_data.empty()) return false;

    runtime_ = common::cuda::nv_unique_ptr<nvinfer1::IRuntime>(
        nvinfer1::createInferRuntime(*logger_));
    if (!runtime_) {
      LOGE("Failed to create the TensorRT runtime");
      return false;
    }
    engine_ = common::cuda::nv_unique_ptr<nvinfer1::ICudaEngine>(
        runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    if (!engine_) {
      LOGE("Failed to deserialize %s", engine_path.c_str());
      return false;
    }
    context_ = common::cuda::nv_unique_ptr<nvinfer1::IExecutionContext>(
        engine_->createExecutionContext());
    if (!context_) {
      LOGE("Failed to create the execution context");
      return false;
    }

    // The export fixed the input names; refuse an engine with another IO
    // contract instead of binding garbage
    for (const char *name : {kVideoTensorName, kStatusTensorName}) {
      if (engine_->getTensorShape(name).nbDims < 0) {
        LOGE("Engine has no tensor '%s' -- not a LatentDrive planner engine", name);
        return false;
      }
    }

    // The view count is a property of the checkpoint: a 5-D video input is
    // the single-camera contract, a 6-D one carries V camera views as
    // contiguous [3, T, H, W] blocks. The trailing four dims are fixed
    nvinfer1::Dims video_dims = engine_->getTensorShape(kVideoTensorName);
    if (video_dims.nbDims == 5) {
      num_views_ = 1;
    } else if (video_dims.nbDims == 6) {
      num_views_ = static_cast<int>(video_dims.d[1]);
    } else {
      LOGE("Unexpected video rank %d (want 5-D [1,3,4,H,W] or 6-D [1,V,3,4,H,W])",
           video_dims.nbDims);
      return false;
    }
    const int tail = video_dims.nbDims - 4;
    if (num_views_ < 1 || video_dims.d[tail] != 3 ||
        video_dims.d[tail + 1] != kNumHistFrames || video_dims.d[tail + 2] != kInputHeight ||
        video_dims.d[tail + 3] != kInputWidth) {
      LOGE("Unexpected video shape %s (want [1, (V,) 3, %d, %d, %d])",
           DimsToString(video_dims).c_str(), kNumHistFrames, kInputHeight, kInputWidth);
      return false;
    }
    nvinfer1::Dims status_dims = engine_->getTensorShape(kStatusTensorName);
    if (status_dims.nbDims != 2 || status_dims.d[1] != kStatusDim) {
      LOGE("Unexpected status shape %s (want [1, %d])", DimsToString(status_dims).c_str(),
           kStatusDim);
      return false;
    }

    // An engine built from an export with a symbolic batch axis reports -1
    // there; the build froze its profile at batch 1, and the context needs
    // the concrete shape before the output shape can be resolved
    for (nvinfer1::Dims *dims : {&video_dims, &status_dims}) {
      if (dims->d[0] == -1) dims->d[0] = 1;
      if (dims->d[0] != 1) {
        LOGE("Engine batch dimension is %ld, this planner runs batch 1", static_cast<long>(dims->d[0]));
        return false;
      }
    }
    if (!context_->setInputShape(kVideoTensorName, video_dims) ||
        !context_->setInputShape(kStatusTensorName, status_dims)) {
      LOGE("setInputShape failed (video %s, status %s)", DimsToString(video_dims).c_str(),
           DimsToString(status_dims).c_str());
      return false;
    }

    // The single output is `plan` in the newer exports and `traj` in the
    // older ones; find it by IO mode rather than by name
    int num_outputs = 0;
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
      const char *name = engine_->getIOTensorName(i);
      if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) {
        output_name_ = name;
        ++num_outputs;
      }
    }
    if (num_outputs != 1) {
      LOGE("Engine has %d outputs, want exactly one ([1, N, 3] plan)", num_outputs);
      return false;
    }

    // The waypoint count is a property of the checkpoint (8 at 2 Hz, 40 at
    // 10 Hz), so it is read from the engine instead of being compiled in
    const nvinfer1::Dims traj_dims = context_->getTensorShape(output_name_.c_str());
    if (traj_dims.nbDims != 3 || traj_dims.d[0] != 1 || traj_dims.d[2] != 3 || traj_dims.d[1] < 1) {
      LOGE("Unexpected %s shape %s (want [1, N, 3])", output_name_.c_str(),
           DimsToString(traj_dims).c_str());
      return false;
    }
    num_waypoints_ = static_cast<int>(traj_dims.d[1]);
    video_numel_   = static_cast<std::size_t>(num_views_) * kVideoNumel;

    CUDA_CHECK(cudaStreamCreate(&stream_));

    const std::size_t traj_bytes = static_cast<std::size_t>(num_waypoints_) * 3 * sizeof(float);
    host_video_                  = common::cuda::MakeHostMemory<float>(video_numel_ * sizeof(float));
    host_status_                 = common::cuda::MakeHostMemory<float>(kStatusDim * sizeof(float));
    host_traj_                   = common::cuda::MakeHostMemory<float>(traj_bytes);
    dev_video_                   = common::cuda::MakeDeviceMemory<float>(video_numel_ * sizeof(float));
    dev_status_                  = common::cuda::MakeDeviceMemory<float>(kStatusDim * sizeof(float));
    dev_traj_                    = common::cuda::MakeDeviceMemory<float>(traj_bytes);

    context_->setTensorAddress(kVideoTensorName, dev_video_.get());
    context_->setTensorAddress(kStatusTensorName, dev_status_.get());
    context_->setTensorAddress(output_name_.c_str(), dev_traj_.get());

    LOG("Deserialized %s (%d view%s, %d waypoints, output '%s')", engine_path.c_str(),
        num_views_, num_views_ == 1 ? "" : "s", num_waypoints_, output_name_.c_str());
    return true;
  }

  common::Trajectory Forward(const float                         *in_video,
                             const std::array<float, kStatusDim> &in_status) final {
    std::memcpy(host_video_.get(), in_video, video_numel_ * sizeof(float));
    return ForwardPrepared(in_status);
  }

  float *VideoInput() final { return host_video_.get(); }

  common::Trajectory ForwardPrepared(const std::array<float, kStatusDim> &in_status) final {
    std::memcpy(host_status_.get(), in_status.data(), kStatusDim * sizeof(float));

    const std::size_t traj_bytes = static_cast<std::size_t>(num_waypoints_) * 3 * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(dev_video_.get(), host_video_.get(),
                               video_numel_ * sizeof(float),
                               cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(dev_status_.get(), host_status_.get(),
                               kStatusDim * sizeof(float),
                               cudaMemcpyHostToDevice, stream_));
    if (!context_->enqueueV3(stream_)) {
      LOGF("enqueueV3 failed");
    }
    CUDA_CHECK(cudaMemcpyAsync(host_traj_.get(), dev_traj_.get(), traj_bytes,
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    static_assert(sizeof(common::Waypoint) == 3 * sizeof(float));
    common::Trajectory out_traj(num_waypoints_);
    std::memcpy(out_traj.data(), host_traj_.get(), traj_bytes);
    return out_traj;
  }

  int         NumWaypoints() const final { return num_waypoints_; }
  int         NumViews() const final { return num_views_; }
  std::size_t VideoNumel() const final { return video_numel_; }

 private:
  cudaStream_t stream_        = nullptr;
  int          num_waypoints_ = 0;
  int          num_views_     = 0;
  std::size_t  video_numel_   = 0;
  std::string  output_name_;

  common::cuda::cuda_host_unique_ptr<float> host_video_  = nullptr;
  common::cuda::cuda_host_unique_ptr<float> host_status_ = nullptr;
  common::cuda::cuda_host_unique_ptr<float> host_traj_   = nullptr;
  common::cuda::cuda_unique_ptr<float>      dev_video_   = nullptr;
  common::cuda::cuda_unique_ptr<float>      dev_status_  = nullptr;
  common::cuda::cuda_unique_ptr<float>      dev_traj_    = nullptr;

  std::shared_ptr<logger::Logger>                          logger_  = nullptr;
  common::cuda::nv_unique_ptr<nvinfer1::IRuntime>          runtime_ = nullptr;
  common::cuda::nv_unique_ptr<nvinfer1::ICudaEngine>       engine_  = nullptr;
  common::cuda::nv_unique_ptr<nvinfer1::IExecutionContext> context_ = nullptr;
};

std::shared_ptr<Planner> CreatePlanner(const std::string &engine_path) {
  std::shared_ptr<PlannerImpl> instance(new PlannerImpl());
  if (!instance->Init(engine_path)) {
    instance.reset();
  }
  return instance;
}

}  // namespace planner
}  // namespace model
