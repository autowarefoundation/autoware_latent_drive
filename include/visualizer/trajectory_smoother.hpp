#ifndef __TRAJECTORY_SMOOTHER_HPP__
#define __TRAJECTORY_SMOOTHER_HPP__

#include <memory>

#include "trajectory.hpp"

namespace visualizer {

struct SmootherParams {
  // Weight of the fresh plan in the blend. 1.0 passes the plan through
  // untouched; 0.35 keeps the displayed tail readable while the visible lag
  // stays under a waypoint's spacing
  double alpha = 0.35;

  // Prediction horizon the waypoints span [s], used together with the
  // waypoint count to place them on a time axis
  double horizon_s = 4.0;

  // A plan that jumps further than this between two steps is treated as a new
  // situation (clip restart, engine swap, a genuine re-plan) and replaces the
  // filter state instead of being blended into it [m]
  double reset_jump_m = 8.0;
};

// @brief `TrajectorySmoother` class is a DISPLAY-ONLY temporal filter for the
// planner output. The engine is deterministic per frame, but consecutive
// frames disagree about how far the plan reaches -- on this clip the +4 s
// waypoint moves ~1.2 m (p95) between steps, almost all of it along-track,
// which reads as a tail that pulses in and out. Nothing here feeds control:
// the raw plan is what gets logged and compared against the fp32 reference,
// this is what gets drawn
//
// Each step carries the previous displayed plan forward by dt along itself --
// the plan already says where the ego will be at +dt, so that pose becomes the
// new origin -- and blends the fresh plan into it on the shared time axis.
// Working on the time axis rather than the waypoint index keeps it correct for
// both the 8-waypoint (2 Hz) and 40-waypoint (10 Hz) engines
class TrajectorySmoother {
 public:
  TrajectorySmoother()          = default;
  virtual ~TrajectorySmoother() = default;

  // @brief: Blend `in_traj` into the filter state and return the trajectory to
  // draw. `in_dt_s` is the wall time since the previous call; pass the clip's
  // step period offline, the measured callback period online
  virtual common::Trajectory Update(const common::Trajectory &in_traj, double in_dt_s) = 0;

  // @brief: Drop the filter state, so the next Update passes its plan through
  virtual void Reset() = 0;
};

// @brief: This function is used as a factory to generate a TrajectorySmoother
// object
std::shared_ptr<TrajectorySmoother> CreateTrajectorySmoother(const SmootherParams &params = {});

}  // namespace visualizer

#endif  //__TRAJECTORY_SMOOTHER_HPP__
