#include "camera_projector.hpp"

#include <opencv2/imgproc.hpp>
#include <vector>

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

 private:
  CameraParams params_;
  cv::Matx33d  r_inv_;  // ego-to-camera
};

std::shared_ptr<CameraProjector> CreateCameraProjector(const CameraParams &params) {
  return std::shared_ptr<CameraProjector>(new CameraProjectorImpl(params));
}

}  // namespace visualizer
