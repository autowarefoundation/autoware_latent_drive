// Bird's-eye-view renderer for planner trajectories.
//
// Pure drawing: no file or ROS dependencies, output is a cv::Mat (BGR) that an
// app writes to disk or a ROS 2 node publishes as sensor_msgs/Image. Ego frame
// convention matches the model output: x forward, y left, heading 0 = forward.
// On the canvas, forward points up and left points left.
#pragma once

#include <string>

#include <opencv2/core.hpp>

#include "latentdrive/types.hpp"

namespace latentdrive {

struct BevConfig {
  float range_fwd_m = 45.f;   // metres of road ahead of the ego origin
  float range_back_m = 5.f;   // metres behind
  float range_lat_m = 16.f;   // metres to each side
  int px_per_m = 12;
  float grid_step_m = 5.f;    // faint grid; labels every second line
};

// Default colors (BGR), matched to the benchmark report's palette.
namespace bev_color {
inline const cv::Scalar kPrediction{214, 120, 42};   // blue   #2a78d6
inline const cv::Scalar kReference{0, 131, 0};       // green  #008300
inline const cv::Scalar kGroundTruth{129, 135, 137}; // gray   #898781
inline const cv::Scalar kSubgoal{52, 104, 235};      // orange #eb6834
}  // namespace bev_color

class BevVisualizer {
 public:
  explicit BevVisualizer(const BevConfig& config = {});

  // Resets the canvas: background, grid, range labels, ego footprint.
  void clear();

  // Polyline from the ego origin through the 8 waypoints, with a dot and a
  // short heading tick per waypoint.
  void drawTrajectory(const Trajectory& traj, const cv::Scalar& color,
                      int thickness = 2, bool dashed = false);

  // The route subgoal (physical metres, ego frame) as a diamond marker.
  void drawSubgoal(float x_m, float y_m, const cv::Scalar& color = bev_color::kSubgoal);

  // Velocity vector from the ego origin, 1 m of arrow per m/s.
  void drawVelocity(float vel_x, float vel_y);

  // One line in the legend block (top-left); call in draw order.
  void addLegend(const std::string& label, const cv::Scalar& color);

  const cv::Mat& image() const { return canvas_; }
  int width() const { return canvas_.cols; }
  int height() const { return canvas_.rows; }

  // Ego-frame metres -> canvas pixel. Exposed so apps can add their own marks.
  cv::Point toPx(float x_m, float y_m) const;

 private:
  BevConfig cfg_;
  cv::Mat canvas_;
  int legend_rows_ = 0;
};

}  // namespace latentdrive
