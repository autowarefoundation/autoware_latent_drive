#ifndef __PLANNER_HPP__
#define __PLANNER_HPP__

#include <array>
#include <memory>
#include <string>

#include "planner_params.hpp"
#include "trajectory.hpp"

namespace model {
namespace planner {

// @brief `Planner` class is a interface class, which only exposure the
// `Forward` pass of the e2e planner engine. TensorRT/CUDA types stay inside
// the implementation, so a consumer (the ROS 2 node, the visualizer tools)
// only links against this header
class Planner {
 public:
  Planner()          = default;
  virtual ~Planner() = default;

  // @brief: Run one plan. `in_video` is the [3, 4, 256, 512] normalized CHW
  // window (oldest..newest), copied into the pinned input buffer before the
  // H2D -> enqueueV3 -> D2H round trip (synchronized on return)
  virtual common::Trajectory Forward(const float                         *in_video,
                                     const std::array<float, kStatusDim> &in_status) = 0;

  // @brief: Zero-copy path. Write the window straight into the pinned buffer
  // returned by `VideoInput` (e.g. FrameWindow::Assemble(planner->VideoInput()))
  // and call `ForwardPrepared`, saving one 6 MB host memcpy per step
  virtual float             *VideoInput()                                                    = 0;
  virtual common::Trajectory ForwardPrepared(const std::array<float, kStatusDim> &in_status) = 0;

  // @brief: Waypoint count of the loaded engine ([1, N, 3] output): 8 for the
  // 2 Hz checkpoints, 40 for the 10 Hz ones. The horizon is 4 s either way
  virtual int NumWaypoints() const = 0;
};

// @brief: This function is used as a factory to deserialize the engine and
// generate a Planner object. Returns nullptr when the engine cannot be loaded
std::shared_ptr<Planner> CreatePlanner(const std::string &engine_path);

}  // namespace planner
}  // namespace model

#endif  //__PLANNER_HPP__
