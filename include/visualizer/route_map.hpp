#ifndef __ROUTE_MAP_HPP__
#define __ROUTE_MAP_HPP__

#include <memory>
#include <opencv2/core.hpp>
#include <vector>

namespace visualizer {

// Ego pose in the clip's map frame, straight out of sequence.json
struct MapPose {
  double x   = 0.0;
  double y   = 0.0;
  double yaw = 0.0;  // [rad]
};

// One numbered point along the route. The subgoal the model consumes is a
// single point that slides along the route, so these are milestones cut from
// that trace at a fixed spacing -- something to measure progress against
struct RouteMilestone {
  cv::Point2d xy;               // map frame
  double      s_m     = 0.0;    // arc position along the route
  int         label   = 0;      // 1-based; the goal is unnumbered
  bool        is_goal = false;
};

struct RouteParams {
  // Milestone spacing [m]; 0 picks one that yields a handful of them
  double milestone_step_m = 0.0;

  // The subgoal trace is stitched onto the driven path only when its far end
  // actually lands on the ego's final position; further apart than this and
  // the clip simply ends before its destination
  double join_tolerance_m = 5.0;

  // Routes shorter than this get no milestones -- a clip that barely moves
  // would otherwise sprout markers every few metres
  double min_route_m = 30.0;
};

// @brief `RouteMap` class is a interface class for the route context of a
// clip: where the whole drive goes, which part of it is already behind, and
// how far the ego still has to run
//
// The clips ship no route file. What they do ship is the per-step subgoal in
// the ego frame plus the ego pose, and the subgoal is a point that rides the
// route a fixed distance ahead -- so lifting every step's subgoal into the map
// frame traces out the route ahead, and the driven path fills in the rest.
// Where the two meet, the clip reaches its destination and the trace pins to
// it; where they do not, the destination is outside the clip and `HasGoal()`
// is false
class RouteMap {
 public:
  RouteMap()          = default;
  virtual ~RouteMap() = default;

  // @brief: Whether the reconstructed route ends on a destination the clip
  // actually reaches. When false there are no milestones and no goal distance
  virtual bool HasGoal() const = 0;

  // @brief: Arc length of the whole reconstructed route [m]
  virtual double RouteLength() const = 0;

  // @brief: How far along the route the ego is at `in_step` [m]
  virtual double ArcAt(std::size_t in_step) const = 0;

  // @brief: Milestones in route order, the goal last. Empty for a route the
  // clip never resolves
  virtual const std::vector<RouteMilestone> &Milestones() const = 0;

  // @brief: Route panel of the given size: the whole route, every subgoal the
  // clip issues, the numbered milestones, the destination, and the ego at
  // `in_step` with the path it has driven
  virtual cv::Mat Draw(cv::Size in_size, std::size_t in_step) const = 0;
};

// @brief: This function is used as a factory to generate a RouteMap object.
// `in_subgoal_ego` is the per-step subgoal in the ego frame [m] (that is,
// status columns 0 and 1 already multiplied back up by 10), one entry per pose
std::shared_ptr<RouteMap> CreateRouteMap(const std::vector<MapPose>     &in_poses,
                                         const std::vector<cv::Point2d> &in_subgoal_ego,
                                         const RouteParams              &params = {});

}  // namespace visualizer

#endif  //__ROUTE_MAP_HPP__
