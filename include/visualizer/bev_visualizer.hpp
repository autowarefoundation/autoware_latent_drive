#ifndef __BEV_VISUALIZER_HPP__
#define __BEV_VISUALIZER_HPP__

#include <memory>
#include <opencv2/core.hpp>
#include <string>

#include "trajectory.hpp"

namespace visualizer {

struct BevParams {
  float range_fwd_m  = 45.f;  // metres of road ahead of the ego origin
  float range_back_m = 5.f;   // metres behind
  float range_lat_m  = 16.f;  // metres to each side
  int   px_per_m     = 12;
  float grid_step_m  = 5.f;  // faint grid; labels every second line
};

// Default trajectory colors (BGR), matched to the benchmark report palette
namespace color {
inline const cv::Scalar kPrediction{214, 120, 42};    // blue   #2a78d6
inline const cv::Scalar kReference{0, 131, 0};        // green  #008300
inline const cv::Scalar kGroundTruth{129, 135, 137};  // gray   #898781
inline const cv::Scalar kSubgoal{52, 104, 235};       // orange #eb6834
}  // namespace color

// @brief `BevVisualizer` class is a interface class for the bird's-eye-view
// rendering of planner trajectories. Pure drawing: no file or ROS
// dependencies, the canvas is a BGR cv::Mat that an app writes to disk or a
// ROS 2 node publishes as sensor_msgs/Image. Ego frame convention matches the
// model output (x forward, y left); on the canvas forward points up
class BevVisualizer {
 public:
  BevVisualizer()          = default;
  virtual ~BevVisualizer() = default;

  // @brief: Reset the canvas: background, metric grid, range labels and the
  // ego footprint
  virtual void Clear() = 0;

  // @brief: Polyline from the ego origin through the waypoints, with a dot
  // and a short heading tick per waypoint
  virtual void DrawTrajectory(const common::Trajectory &in_traj,
                              const cv::Scalar &in_color, int in_thickness = 2,
                              bool in_dashed = false) = 0;

  // @brief: Prediction polyline colored along the horizon (green near -> red
  // far), matching the camera ribbon so the two panels read as one plan
  virtual void DrawTrajectoryHorizon(const common::Trajectory &in_traj,
                                     int                       in_thickness = 2) = 0;

  // @brief: The route subgoal (physical metres, ego frame) as a diamond
  // marker, clamped into view with a distance label when outside the canvas
  virtual void DrawSubgoal(float in_x_m, float in_y_m,
                           const cv::Scalar &in_color = color::kSubgoal) = 0;

  // @brief: Velocity vector from the ego origin, 1 m of arrow per m/s
  virtual void DrawVelocity(float in_vel_x, float in_vel_y) = 0;

  // @brief: One line in the legend block (top-right); call in draw order
  virtual void AddLegend(const std::string &in_label, const cv::Scalar &in_color) = 0;

  virtual const cv::Mat &Image() const = 0;

  // @brief: Ego-frame metres -> canvas pixel, exposed for custom marks
  virtual cv::Point ToPx(float in_x_m, float in_y_m) const = 0;
};

// @brief: This function is used as a factory to generate a BevVisualizer object
std::shared_ptr<BevVisualizer> CreateBevVisualizer(const BevParams &params = {});

}  // namespace visualizer

#endif  //__BEV_VISUALIZER_HPP__
