#ifndef __TRAJECTORY_HPP__
#define __TRAJECTORY_HPP__

#include <array>
#include <cstddef>
#include <vector>

namespace common {

// Ego frame convention throughout: x forward [m], y left [m], heading [rad],
// heading 0 = straight ahead. The frame is the ego pose at inference time
struct Waypoint {
  float x;
  float y;
  float heading;
};

// One plan over a 4 s horizon. The waypoint count is a property of the
// engine, not of this framework: 8 at 2 Hz for e2e-idx8-hist4, 40 at 10 Hz
// for the -wp10hz variant. Planner::NumWaypoints() reports it after load
using Trajectory = std::vector<Waypoint>;

// @brief A route point in the ego frame [m]
struct RoutePoint {
  float x_m = 0.f;  // forward
  float y_m = 0.f;  // left
};

// @brief Ego status in physical units. `Encode` applies the training-time
// layout [p1x, p1y, ..., pKx, pKy, v_fwd, v_left, a_fwd, a_left]: every route
// point is divided by 10, velocity and acceleration pass through unscaled.
// One route point (the point 50 m of arc length ahead) is the original
// single-subgoal contract, status [1, 6]; the route-point models take six
// (5, 10, 15, 20, 30, 50 m), status [1, 16]. Planner::StatusDim() says which
struct EgoStatus {
  std::vector<RoutePoint> route_points;  // nearest first
  float                   vel_x = 0.f;   // [m/s]
  float                   vel_y = 0.f;
  float                   acc_x = 0.f;  // [m/s^2]
  float                   acc_y = 0.f;

  int Dim() const { return static_cast<int>(2 * route_points.size() + 4); }

  std::vector<float> Encode() const {
    std::vector<float> out;
    out.reserve(static_cast<std::size_t>(Dim()));
    for (const RoutePoint &p : route_points) {
      out.push_back(p.x_m / 10.f);
      out.push_back(p.y_m / 10.f);
    }
    out.insert(out.end(), {vel_x, vel_y, acc_x, acc_y});
    return out;
  }

  // @brief The inverse of Encode, for status vectors read from disk
  // (status.npy rows). `dim` must be 2K + 4 with K >= 1
  static EgoStatus Decode(const float *status, int dim) {
    EgoStatus  ego;
    const int k = (dim - 4) / 2;
    for (int i = 0; i < k; ++i) ego.route_points.push_back({status[2 * i] * 10.f, status[2 * i + 1] * 10.f});
    ego.vel_x = status[2 * k];
    ego.vel_y = status[2 * k + 1];
    ego.acc_x = status[2 * k + 2];
    ego.acc_y = status[2 * k + 3];
    return ego;
  }

  // @brief The farthest route point (the 50 m one in both contracts)
  RoutePoint Subgoal() const { return route_points.empty() ? RoutePoint{} : route_points.back(); }
};

}  // namespace common

#endif  //__TRAJECTORY_HPP__
