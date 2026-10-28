#include "planner.hpp"

#include <cstring>

#include "NvInfer.h"
#include "logger.hpp"
#include "utils.hpp"

namespace model {
namespace planner {

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

    // The export fixed the tensor names; refuse an engine with another IO
    // contract instead of binding garbage
    for (const char *name : {kVideoTensorName, kStatusTensorName, kTrajTensorName}) {
      if (engine_->getTensorShape(name).nbDims < 0) {
        LOGE("Engine has no tensor '%s' -- not a LatentDrive planner engine", name);
        return false;
      }
    }

    // The waypoint count is a property of the checkpoint (8 at 2 Hz, 40 at
    // 10 Hz), so it is read from the engine instead of being compiled in
    const nvinfer1::Dims traj_dims = engine_->getTensorShape(kTrajTensorName);
    if (traj_dims.nbDims != 3 || traj_dims.d[0] != 1 || traj_dims.d[2] != 3) {
      LOGE("Unexpected traj shape (want [1, N, 3])");
      return false;
    }
    num_waypoints_ = static_cast<int>(traj_dims.d[1]);

    CUDA_CHECK(cudaStreamCreate(&stream_));

    const std::size_t traj_bytes = static_cast<std::size_t>(num_waypoints_) * 3 * sizeof(float);
    host_video_                  = common::cuda::MakeHostMemory<float>(kVideoNumel * sizeof(float));
    host_status_                 = common::cuda::MakeHostMemory<float>(kStatusDim * sizeof(float));
    host_traj_                   = common::cuda::MakeHostMemory<float>(traj_bytes);
    dev_video_                   = common::cuda::MakeDeviceMemory<float>(kVideoNumel * sizeof(float));
    dev_status_                  = common::cuda::MakeDeviceMemory<float>(kStatusDim * sizeof(float));
    dev_traj_                    = common::cuda::MakeDeviceMemory<float>(traj_bytes);

    context_->setTensorAddress(kVideoTensorName, dev_video_.get());
    context_->setTensorAddress(kStatusTensorName, dev_status_.get());
    context_->setTensorAddress(kTrajTensorName, dev_traj_.get());

    LOG("Deserialized %s (%d waypoints)", engine_path.c_str(), num_waypoints_);
    return true;
  }

  common::Trajectory Forward(const float                         *in_video,
                             const std::array<float, kStatusDim> &in_status) final {
    std::memcpy(host_video_.get(), in_video, kVideoNumel * sizeof(float));
    return ForwardPrepared(in_status);
  }

  float *VideoInput() final { return host_video_.get(); }

  common::Trajectory ForwardPrepared(const std::array<float, kStatusDim> &in_status) final {
    std::memcpy(host_status_.get(), in_status.data(), kStatusDim * sizeof(float));

    const std::size_t traj_bytes = static_cast<std::size_t>(num_waypoints_) * 3 * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(dev_video_.get(), host_video_.get(),
                               kVideoNumel * sizeof(float),
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

  int NumWaypoints() const final { return num_waypoints_; }

 private:
  cudaStream_t stream_        = nullptr;
  int          num_waypoints_ = 0;

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
