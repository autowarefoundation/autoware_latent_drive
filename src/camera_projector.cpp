#include "latentdrive/camera_projector.hpp"

#include <vector>

#include <opencv2/imgproc.hpp>

namespace latentdrive {

CameraProjector::CameraProjector(const CameraCalibration& calib)
    : calib_(calib), R_inv_(calib.R.t()) {}

std::optional<cv::Point2d> CameraProjector::project(double x_m, double y_m,
                                                    double z_m) const {
  const cv::Vec3d p_cam = R_inv_ * (cv::Vec3d(x_m, y_m, z_m) - calib_.t);
  if (p_cam[2] <= calib_.min_depth_m) return std::nullopt;
  const cv::Vec3d uv = calib_.K * p_cam;
  return cv::Point2d(uv[0] / uv[2], uv[1] / uv[2]);
}

void CameraProjector::drawTrajectory(cv::Mat& image, const Trajectory& traj,
                                     const cv::Scalar& color, int thickness,
                                     bool draw_dots) const {
  // ego origin prepended so the line starts "from the car" when visible
  std::vector<std::optional<cv::Point2d>> px;
  px.push_back(project(0.0, 0.0));
  for (const Waypoint& wp : traj) px.push_back(project(wp.x, wp.y));

  for (std::size_t i = 0; i + 1 < px.size(); ++i) {
    if (px[i] && px[i + 1]) {
      cv::line(image, *px[i], *px[i + 1], color, thickness, cv::LINE_AA);
    }
  }
  if (draw_dots) {
    for (std::size_t i = 1; i < px.size(); ++i) {
      if (px[i]) cv::circle(image, *px[i], thickness + 1, color, cv::FILLED, cv::LINE_AA);
    }
  }
}

}  // namespace latentdrive
