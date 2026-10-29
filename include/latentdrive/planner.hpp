#ifndef __PLANNER_HPP__
#define __PLANNER_HPP__

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

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

  // @brief: Run one plan. `in_video` is the normalized CHW window
  // (oldest..newest) of every view the engine takes, laid out as NumViews()
  // contiguous [3, 4, 256, 512] blocks (view 0 first) -- VideoNumel() floats.
  // It is copied into the pinned input buffer before the H2D -> enqueueV3 ->
  // D2H round trip (synchronized on return). `in_status` is the encoded
  // status (EgoStatus::Encode), StatusDim() floats; a status of another size
  // is refused with an error and an empty trajectory
  virtual common::Trajectory Forward(const float *in_video, const std::vector<float> &in_status) = 0;

  // @brief: Zero-copy path. Write the window straight into the pinned buffer
  // returned by `VideoInput` (e.g. window->Assemble(planner->VideoInput()))
  // and call `ForwardPrepared`, saving one 6 MB-per-view host memcpy per step
  virtual float             *VideoInput()                                         = 0;
  virtual common::Trajectory ForwardPrepared(const std::vector<float> &in_status) = 0;

  // @brief: Waypoint count of the loaded engine ([1, N, 3] output): 8 for the
  // 2 Hz checkpoints, 40 for the 10 Hz ones. The horizon is 4 s either way
  virtual int NumWaypoints() const = 0;

  // @brief: Camera views the loaded engine takes: 1 for a 5-D `video` input,
  // V for a 6-D [1, V, 3, 4, H, W] one (view 0 anchors the history; for the
  // camfront3 models the order is front, front_left, front_right).
  // VideoNumel() is the float count of one complete input
  virtual int         NumViews() const   = 0;
  virtual std::size_t VideoNumel() const = 0;

  // @brief: Length of the `status` input (2K + 4) and its route point count
  // K: 1 for the single-subgoal models (status [1, 6]), 6 for the route-point
  // ones (status [1, 16])
  virtual int StatusDim() const      = 0;
  virtual int NumRoutePoints() const = 0;
};

// @brief: This function is used as a factory to deserialize the engine and
// generate a Planner object. Returns nullptr when the engine cannot be loaded
std::shared_ptr<Planner> CreatePlanner(const std::string &engine_path);

}  // namespace planner
}  // namespace model

#endif  //__PLANNER_HPP__
