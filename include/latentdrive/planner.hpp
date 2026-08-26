// TensorRT wrapper around the LatentDrive e2e planner engine.
//
// TensorRT/CUDA headers stay out of this interface (pImpl) so downstream
// consumers -- a ROS 2 node, a visualizer -- only need this header and a link
// against the latentdrive library.
#pragma once

#include <array>
#include <memory>
#include <string>

#include "latentdrive/types.hpp"

namespace latentdrive {

class Planner {
 public:
  // Deserializes the engine, allocates pinned host + device buffers, warms
  // nothing up (call infer() a few times before trusting latency numbers).
  // Throws std::runtime_error on any TensorRT/CUDA failure.
  explicit Planner(const std::string& engine_path);
  ~Planner();
  Planner(const Planner&) = delete;
  Planner& operator=(const Planner&) = delete;

  // Copies `video` ([3,4,256,512] normalized CHW, oldest..newest) into the
  // pinned input buffer and runs H2D -> enqueueV3 -> D2H, synchronized.
  Trajectory infer(const float* video, const std::array<float, kStatusDim>& status);

  Trajectory infer(const float* video, const EgoStatus& status) {
    return infer(video, status.encode());
  }

  // Zero-copy path: write the window straight into this pinned buffer
  // (e.g. FrameWindow::assemble(planner.videoInput())) then call
  // inferPrepared(). Saves one 6 MB host memcpy per step.
  float* videoInput();
  Trajectory inferPrepared(const std::array<float, kStatusDim>& status);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace latentdrive
