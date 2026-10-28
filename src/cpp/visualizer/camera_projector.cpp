#include "camera_projector.hpp"

#include <cmath>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "color_map.hpp"

namespace visualizer {

// `CameraProjectorImpl` is an implementation class, which implement the
// pinhole projection with the mandatory depth gate
class CameraProjectorImpl : public CameraProjector {
 public:
  explicit CameraProjectorImpl(const CameraParams &params)
      : params_(params), r_inv_(params.r.t()) {}

  CameraProjectorImpl(const CameraProjectorImpl &)            = delete;
  CameraProjectorImpl &operator=(const CameraProjectorImpl &) = delete;
  ~CameraProjectorImpl() override                             = default;

  std::optional<cv::Point2d> Project(double in_x_m, double in_y_m,
                                     double in_z_m) const final {
    const cv::Vec3d p_cam = r_inv_ * (cv::Vec3d(in_x_m, in_y_m, in_z_m) - params_.t);
    if (p_cam[2] <= params_.min_depth_m) return std::nullopt;
    const cv::Vec3d uv = params_.k * p_cam;
    return cv::Point2d(uv[0] / uv[2], uv[1] / uv[2]);
  }

  void DrawTrajectory(cv::Mat &inout_image, const common::Trajectory &in_traj,
                      const cv::Scalar &in_color, int in_thickness,
                      bool in_draw_dots) const final {
    // Ego origin prepended so the line starts "from the car" when visible
    std::vector<std::optional<cv::Point2d>> pixels;
    pixels.push_back(Project(0.0, 0.0, 0.0));
    for (const common::Waypoint &wp : in_traj) pixels.push_back(Project(wp.x, wp.y, 0.0));

    for (std::size_t i = 0; i + 1 < pixels.size(); ++i) {
      if (pixels[i] && pixels[i + 1]) {
        cv::line(inout_image, *pixels[i], *pixels[i + 1], in_color, in_thickness, cv::LINE_AA);
      }
    }
    if (in_draw_dots) {
      for (std::size_t i = 1; i < pixels.size(); ++i) {
        if (pixels[i]) {
          cv::circle(inout_image, *pixels[i], in_thickness + 1, in_color, cv::FILLED, cv::LINE_AA);
        }
      }
    }
  }

  void DrawRibbon(cv::Mat &inout_image, const common::Trajectory &in_traj,
                  double in_half_width_m, double in_alpha) const final {
    if (in_traj.size() < 2) return;

    // Path points in the ego ground plane, ego origin prepended
    std::vector<cv::Point2d> path;
    path.reserve(in_traj.size() + 1);
    path.emplace_back(0.0, 0.0);
    for (const common::Waypoint &wp : in_traj) path.emplace_back(wp.x, wp.y);

    // The corridor is drawn onto a scratch layer and blended once, so the
    // per-segment fills stay translucent without compounding where quads meet
    cv::Mat overlay = inout_image.clone();
    cv::Mat drawn(inout_image.size(), CV_8U, cv::Scalar(0));

    for (std::size_t i = 0; i + 1 < path.size(); ++i) {
      // Lateral offset perpendicular to the local heading, in the ground plane
      const cv::Point2d dir = path[i + 1] - path[i];
      const double      len = std::hypot(dir.x, dir.y);
      if (len < 1e-6) continue;
      const cv::Point2d nrm(-dir.y / len * in_half_width_m, dir.x / len * in_half_width_m);

      const auto l0 = Project(path[i].x + nrm.x, path[i].y + nrm.y, 0.0);
      const auto r0 = Project(path[i].x - nrm.x, path[i].y - nrm.y, 0.0);
      const auto l1 = Project(path[i + 1].x + nrm.x, path[i + 1].y + nrm.y, 0.0);
      const auto r1 = Project(path[i + 1].x - nrm.x, path[i + 1].y - nrm.y, 0.0);
      if (!l0 || !r0 || !l1 || !r1) continue;  // any corner behind the camera

      const cv::Point quad[4] = {cv::Point(*l0), cv::Point(*r0), cv::Point(*r1), cv::Point(*l1)};
      // Color at this segment's horizon fraction (green near -> red far)
      const double frac = static_cast<double>(i) / (path.size() - 1);
      cv::fillConvexPoly(overlay, quad, 4, HorizonColor(frac), cv::LINE_AA);
      cv::fillConvexPoly(drawn, quad, 4, cv::Scalar(255), cv::LINE_AA);
    }

    // Blend only where the corridor was painted
    cv::Mat blended;
    cv::addWeighted(overlay, in_alpha, inout_image, 1.0 - in_alpha, 0.0, blended);
    blended.copyTo(inout_image, drawn);
  }

 private:
  CameraParams params_;
  cv::Matx33d  r_inv_;  // ego-to-camera
};

std::shared_ptr<CameraProjector> CreateCameraProjector(const CameraParams &params) {
  return std::shared_ptr<CameraProjector>(new CameraProjectorImpl(params));
}

}  // namespace visualizer
