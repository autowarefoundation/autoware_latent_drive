#ifndef __TRAJECTORY_HPP__
#define __TRAJECTORY_HPP__

#include <array>

namespace common {

constexpr static int kNumWaypoints = 8;  // planner horizon: t+0.5 .. t+4.0 s

// Ego frame convention throughout: x forward [m], y left [m], heading [rad],
// heading 0 = straight ahead. The frame is the ego pose at inference time
struct Waypoint {
  float x;
  float y;
  float heading;
};

// One plan: 8 waypoints, 0.5 s apart
using Trajectory = std::array<Waypoint, kNumWaypoints>;

// @brief Ego status in physical units. `Encode` applies the training-time
// layout: the subgoal (route point 50 m of arc length ahead, ego frame) is
// divided by 10, velocity and acceleration pass through unscaled
struct EgoStatus {
  float subgoal_x_m = 0.f;
  float subgoal_y_m = 0.f;
  float vel_x       = 0.f;  // [m/s]
  float vel_y       = 0.f;
  float acc_x       = 0.f;  // [m/s^2]
  float acc_y       = 0.f;

  std::array<float, 6> Encode() const {
    return {subgoal_x_m / 10.f, subgoal_y_m / 10.f, vel_x, vel_y, acc_x, acc_y};
  }
};

}  // namespace common

#endif  //__TRAJECTORY_HPP__
