#include "bev_visualizer.hpp"

#include <cmath>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "color_map.hpp"

namespace visualizer {

// Chrome palette (BGR), matching the dark theme of the benchmark report
static const cv::Scalar kBackground{25, 26, 26};  // #1a1a19
static const cv::Scalar kGrid{42, 44, 44};        // #2c2c2a
static const cv::Scalar kAxis{53, 56, 56};        // #383835
static const cv::Scalar kText{183, 194, 195};     // #c3c2b7
static const cv::Scalar kEgo{255, 255, 255};

constexpr static int kFont = cv::FONT_HERSHEY_SIMPLEX;

// @brief: This function is used to draw a dashed polyline: 8 px dash / 6 px
// gap, walked along each segment
static void DashedPolyline(cv::Mat &inout_image, const std::vector<cv::Point> &in_points,
                           const cv::Scalar &in_color, int in_thickness) {
  const double kDash = 8.0;
  const double kGap  = 6.0;
  for (std::size_t i = 0; i + 1 < in_points.size(); ++i) {
    const cv::Point2d a   = in_points[i];
    const cv::Point2d b   = in_points[i + 1];
    const double      len = cv::norm(b - a);
    if (len < 1e-6) continue;
    const cv::Point2d dir = (b - a) / len;
    for (double s = 0; s < len; s += kDash + kGap) {
      const cv::Point2d p0 = a + dir * s;
      const cv::Point2d p1 = a + dir * std::min(s + kDash, len);
      cv::line(inout_image, p0, p1, in_color, in_thickness, cv::LINE_AA);
    }
  }
}

// `BevVisualizerImpl` is an implementation class, which owns the canvas and
// implements the detailed drawing
class BevVisualizerImpl : public BevVisualizer {
 public:
  explicit BevVisualizerImpl(const BevParams &params) : params_(params) {
    const int w = static_cast<int>(2 * params_.range_lat_m * params_.px_per_m);
    const int h = static_cast<int>((params_.range_fwd_m + params_.range_back_m) * params_.px_per_m);
    canvas_.create(h, w, CV_8UC3);
    Clear();
  }

  BevVisualizerImpl(const BevVisualizerImpl &)            = delete;
  BevVisualizerImpl &operator=(const BevVisualizerImpl &) = delete;
  ~BevVisualizerImpl() override                           = default;

  cv::Point ToPx(float in_x_m, float in_y_m) const final {
    return {static_cast<int>(std::lround((params_.range_lat_m - in_y_m) * params_.px_per_m)),
            static_cast<int>(std::lround((params_.range_fwd_m - in_x_m) * params_.px_per_m))};
  }

  void Clear() final {
    canvas_.setTo(kBackground);
    legend_rows_ = 0;

    // Grid: faint lines every grid_step_m, forward labels every second line
    for (float x = 0; x <= params_.range_fwd_m; x += params_.grid_step_m) {
      for (const float sx : {x, -x}) {
        if (-sx > params_.range_back_m) continue;
        const int py = ToPx(sx, 0).y;
        cv::line(canvas_, {0, py}, {canvas_.cols, py}, sx == 0.f ? kAxis : kGrid, 1);
        if (sx > 0 && static_cast<int>(x) % (2 * static_cast<int>(params_.grid_step_m)) == 0) {
          cv::putText(canvas_, cv::format("%.0f m", sx), {4, py - 4}, kFont, 0.38,
                      kText, 1, cv::LINE_AA);
        }
      }
    }
    for (float y = 0; y <= params_.range_lat_m; y += params_.grid_step_m) {
      for (const float sy : {y, -y}) {
        const int px = ToPx(0, sy).x;
        cv::line(canvas_, {px, 0}, {px, canvas_.rows}, sy == 0.f ? kAxis : kGrid, 1);
      }
    }

    // Ego footprint (typical passenger car, rear-axle-ish origin)
    const cv::Point p0   = ToPx(3.6f, 0.95f);
    const cv::Point p1   = ToPx(-1.2f, -0.95f);
    const cv::Point nose = ToPx(3.6f, 0.f);
    cv::rectangle(canvas_, p0, p1, kEgo, 1, cv::LINE_AA);
    cv::drawMarker(canvas_, {nose.x, nose.y - 6}, kEgo, cv::MARKER_TRIANGLE_UP, 8, 1, cv::LINE_AA);
  }

  void DrawTrajectory(const common::Trajectory &in_traj, const cv::Scalar &in_color,
                      int in_thickness, bool in_dashed) final {
    std::vector<cv::Point> points;
    points.push_back(ToPx(0.f, 0.f));
    for (const common::Waypoint &wp : in_traj) points.push_back(ToPx(wp.x, wp.y));

    if (in_dashed) {
      DashedPolyline(canvas_, points, in_color, in_thickness);
    } else {
      cv::polylines(canvas_, points, false, in_color, in_thickness, cv::LINE_AA);
    }

    const float kTickLength = 1.2f;  // heading tick per waypoint [m]
    for (const common::Waypoint &wp : in_traj) {
      const cv::Point p   = ToPx(wp.x, wp.y);
      const cv::Point tip = ToPx(wp.x + kTickLength * std::cos(wp.heading),
                                 wp.y + kTickLength * std::sin(wp.heading));
      cv::circle(canvas_, p, in_thickness + 1, in_color, cv::FILLED, cv::LINE_AA);
      cv::line(canvas_, p, tip, in_color, 1, cv::LINE_AA);
    }
  }

  void DrawTrajectoryHorizon(const common::Trajectory &in_traj, int in_thickness) final {
    cv::Point prev = ToPx(0.f, 0.f);
    for (std::size_t i = 0; i < in_traj.size(); ++i) {
      const cv::Point  p    = ToPx(in_traj[i].x, in_traj[i].y);
      const double     frac = static_cast<double>(i) / (in_traj.size() - 1);
      const cv::Scalar c    = HorizonColor(frac);
      cv::line(canvas_, prev, p, c, in_thickness, cv::LINE_AA);
      cv::circle(canvas_, p, in_thickness, c, cv::FILLED, cv::LINE_AA);
      prev = p;
    }
  }

  void DrawSubgoal(float in_x_m, float in_y_m, const cv::Scalar &in_color) final {
    // Clamp into view so an out-of-range subgoal still shows its direction
    const float x = std::min(in_x_m, params_.range_fwd_m - 1.5f);
    const float y = std::max(std::min(in_y_m, params_.range_lat_m - 1.5f),
                             -params_.range_lat_m + 1.5f);
    cv::drawMarker(canvas_, ToPx(x, y), in_color, cv::MARKER_DIAMOND, 14, 2, cv::LINE_AA);
    if (x != in_x_m || y != in_y_m) {
      cv::putText(canvas_, cv::format("%.0fm", std::hypot(in_x_m, in_y_m)),
                  ToPx(x, y) + cv::Point(8, 4), kFont, 0.38, in_color, 1, cv::LINE_AA);
    }
  }

  void DrawVelocity(float in_vel_x, float in_vel_y) final {
    const cv::Point from = ToPx(0.f, 0.f);
    const cv::Point to   = ToPx(in_vel_x, in_vel_y);  // 1 m of arrow per m/s
    if (from == to) return;
    cv::arrowedLine(canvas_, from, to, kText, 1, cv::LINE_AA, 0, 0.18);
  }

  void AddLegend(const std::string &in_label, const cv::Scalar &in_color) final {
    const int y  = 18 + legend_rows_ * 18;
    const int x0 = canvas_.cols - 150;
    ++legend_rows_;
    cv::line(canvas_, {x0, y - 4}, {x0 + 16, y - 4}, in_color, 3, cv::LINE_AA);
    cv::putText(canvas_, in_label, {x0 + 22, y}, kFont, 0.42, kText, 1, cv::LINE_AA);
  }

  const cv::Mat &Image() const final { return canvas_; }

 private:
  BevParams params_;
  cv::Mat   canvas_;
  int       legend_rows_ = 0;
};

std::shared_ptr<BevVisualizer> CreateBevVisualizer(const BevParams &params) {
  return std::shared_ptr<BevVisualizer>(new BevVisualizerImpl(params));
}

}  // namespace visualizer
