// Projects ego-frame trajectories onto the camera image (pinhole model).
//
// Convention (T4 calibrated_sensor): extrinsics are camera-to-ego, so an
// ego-frame point p projects as
//     p_cam = R^T (p - t);   uv = K p_cam;   pixel = uv.xy / uv.z
// with a MANDATORY depth gate p_cam.z > min_depth_m -- at a standstill the
// whole plan sits at the ego origin, behind the camera, and an ungated
// projection sprays lines across the frame.
//
// Waypoints are projected on the ground plane (z = 0 in the ego frame).
#pragma once

#include <optional>

#include <opencv2/core.hpp>

#include "latentdrive/types.hpp"

namespace latentdrive {

struct CameraCalibration {
  cv::Matx33d K;          // intrinsics for the image size being drawn on
  cv::Matx33d R;          // camera-to-ego rotation
  cv::Vec3d t;            // camera origin in the ego frame [m]
  double min_depth_m = 0.5;
};

class CameraProjector {
 public:
  explicit CameraProjector(const CameraCalibration& calib);

  // Ego-frame point -> pixel, or nullopt if it fails the depth gate.
  std::optional<cv::Point2d> project(double x_m, double y_m, double z_m = 0.0) const;

  // Polyline through the waypoints (ego origin prepended), each segment drawn
  // only when both endpoints pass the depth gate, plus a dot per visible
  // waypoint. Thickness tapers with distance is left to the caller; this
  // draws uniform width.
  void drawTrajectory(cv::Mat& image, const Trajectory& traj, const cv::Scalar& color,
                      int thickness = 2, bool draw_dots = true) const;

 private:
  CameraCalibration calib_;
  cv::Matx33d R_inv_;  // ego-to-camera
};

}  // namespace latentdrive
