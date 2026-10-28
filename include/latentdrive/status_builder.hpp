#ifndef __STATUS_BUILDER_HPP__
#define __STATUS_BUILDER_HPP__

#include <array>
#include <cstddef>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "planner_params.hpp"
#include "trajectory.hpp"

namespace status {

// Map-frame point [m]
struct Point2d {
  double x = 0.0;
  double y = 0.0;
};

// One localization sample in the map frame. Velocity and acceleration are the
// odometry twist / acceleration topic values (ego frame); they are only used
// when the motion terms come from the twist instead of the pose difference
struct PoseSample {
  double stamp_s = 0.0;
  double x       = 0.0;
  double y       = 0.0;
  double yaw     = 0.0;
  double vel_x   = 0.0;
  double vel_y   = 0.0;
  double acc_x   = 0.0;
  double acc_y   = 0.0;
};

/****************************************/
/*****planner_io, one function each******/
/****************************************/

// @brief: Drop consecutive duplicates, then repeatedly remove the vertex at
// which the polyline turns by more than `max_turn_rad` (a route that doubles
// back at a junction entry). `max_turn_rad <= 0` only drops the duplicates
std::vector<Point2d> RemoveReversals(const std::vector<Point2d> &points, double max_turn_rad);

// @brief: planner_io.resample_route: the polyline resampled every `step` m of
// arc length, starting at its first point
std::vector<Point2d> ResamplePolyline(const std::vector<Point2d> &points, double step);

// @brief: Index of the vertex of a 1 m route nearest to `position`. With a
// hint the search covers [hint - 2, hint - 2 + window) only, so a route that
// crosses itself cannot snap the projection backwards
std::size_t ProjectArclength(const std::vector<Point2d> &route_1m, const Point2d &position,
                             std::optional<std::size_t> hint, std::size_t window = 80);

// @brief: The point `ahead_m` of arc length past vertex `s_now` of a 1 m
// route, linearly interpolated and clamped at the route's end
Point2d PointAhead(const std::vector<Point2d> &route_1m, std::size_t s_now, double ahead_m);

// @brief: The single-subgoal models' rule: the point `ahead_m` of arc length
// past the reference vertex nearest to `position`, interpolated inside the
// segment that crosses the distance, clamped at the end
Point2d SubgoalAlong(const std::vector<Point2d> &reference, const Point2d &position, double ahead_m);

// @brief: planner_io.ego_status's motion terms. `samples` are four poses
// kMotionSampleDtS apart, oldest first, ending at "now": the velocity is the
// chord over the last two intervals, the acceleration its change over one,
// both rotated into the ego frame of `yaw_now`. Returns
// {v_fwd, v_left, a_fwd, a_left}. On a turning car this gives the lateral
// acceleration of about v * yaw_rate the training saw, which the twist and
// acceleration topics report as zero
std::array<double, 4> MotionFromPoses(const std::array<PoseSample, 4> &samples, double yaw_now);

// @brief: The four samples nearest to now - 3*dt, now - 2*dt, now - dt, now,
// or nothing when one of them is more than `tolerance_s` from its target
std::optional<std::array<PoseSample, 4>> PickMotionSamples(const std::deque<PoseSample> &history,
                                                           double now_s, double dt_s,
                                                           double tolerance_s);

/****************************************/
/*************StatusBuilder**************/
/****************************************/

enum class MotionSource {
  kPoseDifference,  // planner_io: differenced positions (the training convention, default)
  kTwist,           // the odometry twist and the acceleration topic
};

struct StatusBuilderParams {
  // Route point count of the engine (Planner::NumRoutePoints()) and their
  // distances; empty distances pick the defaults for 1 (50 m) or 6 points
  int                 num_route_points = 6;
  std::vector<double> route_points_m;
  double              route_reversal_deg = model::planner::kDefaultRouteReversalDeg;
  MotionSource        motion_source      = MotionSource::kPoseDifference;
  double              motion_dt_s        = model::planner::kMotionSampleDtS;
  double              motion_tolerance_s = model::planner::kMotionSampleToleranceS;
  // Pose history kept for the motion terms
  double retain_s = 2.0;
};

// @brief What `Build` produced, beside the status itself: the route points in
// the map frame (for drawing / debugging) and where the motion terms came from
struct StatusResult {
  bool                 ok = false;
  std::string          error;
  common::EgoStatus    ego;
  std::vector<Point2d> route_points_map;
  bool                 motion_from_poses = false;
};

// @brief `StatusBuilder` class is a interface class for the model's `status`
// input, built the way autoware_tensorrt_e2e builds it on the vehicle (and
// planner_io.ego_status in training): route points from a map-frame route,
// motion terms from the localization history. It has no ROS dependency; a
// node feeds it odometry samples and a route polyline, an offline tool the
// same from files.
//
// Route points (K > 1): the route is cleaned of reversals and resampled at
// 1 m once per SetRoute; the ego is projected onto it with a forward hint
// carried across calls (a jump of more than 10 m from the hinted vertex --
// relocalisation, a new route -- falls back to a global search); each point is
// route_points_m[k] of arc length past that vertex, clamped at the route end.
// The single-subgoal contract (K = 1) keeps its original rule (SubgoalAlong
// on the raw route, 50 m ahead)
class StatusBuilder {
 public:
  StatusBuilder()          = default;
  virtual ~StatusBuilder() = default;

  // @brief: Replace the route (map frame, in driving order)
  virtual void SetRoute(const std::vector<Point2d> &route) = 0;
  virtual bool HasRoute() const                            = 0;

  // @brief: Add one localization sample; samples must arrive in time order.
  // A stamp that runs back by more than a second (a looping bag) clears the
  // history and the projection hint
  virtual void PushPose(const PoseSample &sample) = 0;

  // @brief: The status for the instant `stamp_s` with the ego at `ego`
  // (normally the newest pose at or before the camera frame being planned on)
  virtual StatusResult Build(double stamp_s, const PoseSample &ego) = 0;

  // @brief: The same with route points given directly in the map frame, e.g.
  // the ones the vehicle's planner recorded (/tensorrt_e2e/debug/latentdrive/
  // route_points); only the motion terms are computed here
  virtual StatusResult BuildWithRoutePoints(double stamp_s, const PoseSample &ego,
                                            const std::vector<Point2d> &route_points_map) = 0;

  virtual const std::vector<double> &RoutePointsM() const = 0;
  virtual void                       Reset()              = 0;
};

// @brief: This function is used as a factory to generate a StatusBuilder.
// Returns nullptr on an invalid parameter set (e.g. distances that do not
// match the route point count)
std::shared_ptr<StatusBuilder> CreateStatusBuilder(const StatusBuilderParams &params);

}  // namespace status

#endif  //__STATUS_BUILDER_HPP__
