#ifndef __CAMERA_PROJECTOR_HPP__
#define __CAMERA_PROJECTOR_HPP__

#include <memory>
#include <opencv2/core.hpp>
#include <optional>

#include "trajectory.hpp"

namespace visualizer {

// T4 calibrated_sensor convention: the extrinsics are camera-to-ego, so an
// ego-frame point p projects as
//     p_cam = R^T (p - t);   uv = K p_cam;   pixel = uv.xy / uv.z
struct CameraParams {
  cv::Matx33d k;  // intrinsics for the image size being drawn on
  cv::Matx33d r;  // camera-to-ego rotation
  cv::Vec3d   t;  // camera origin in the ego frame [m]

  // The depth gate is MANDATORY: at a standstill the whole plan sits at the
  // ego origin behind the camera, and an ungated projection sprays lines
  // across the frame
  double min_depth_m = 0.5;
};

// @brief `CameraProjector` class is a interface class for overlaying
// ego-frame trajectories on the camera image (pinhole model). Waypoints are
// projected on the ground plane (z = 0 in the ego frame)
class CameraProjector {
 public:
  CameraProjector()          = default;
  virtual ~CameraProjector() = default;

  // @brief: Ego-frame point -> pixel, or nullopt when it fails the depth gate
  virtual std::optional<cv::Point2d> Project(double in_x_m, double in_y_m,
                                             double in_z_m = 0.0) const = 0;

  // @brief: Polyline through the waypoints (ego origin prepended so the line
  // starts from the car when visible); a segment draws only when both of its
  // endpoints pass the depth gate, plus a dot per visible waypoint
  virtual void DrawTrajectory(cv::Mat &inout_image, const common::Trajectory &in_traj,
                              const cv::Scalar &in_color, int in_thickness = 2,
                              bool in_draw_dots = true) const = 0;
};

// @brief: This function is used as a factory to generate a CameraProjector object
std::shared_ptr<CameraProjector> CreateCameraProjector(const CameraParams &params);

}  // namespace visualizer

#endif  //__CAMERA_PROJECTOR_HPP__
