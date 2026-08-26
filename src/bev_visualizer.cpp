#include "latentdrive/bev_visualizer.hpp"

#include <cmath>

#include <opencv2/imgproc.hpp>

namespace latentdrive {
namespace {

// Chrome palette (BGR), matching the dark theme of the benchmark report.
const cv::Scalar kBackground{25, 26, 26};   // #1a1a19
const cv::Scalar kGrid{42, 44, 44};         // #2c2c2a
const cv::Scalar kAxis{53, 56, 56};         // #383835
const cv::Scalar kText{183, 194, 195};      // #c3c2b7
const cv::Scalar kEgo{255, 255, 255};

constexpr int kFont = cv::FONT_HERSHEY_SIMPLEX;

void dashedPolyline(cv::Mat& img, const std::vector<cv::Point>& pts,
                    const cv::Scalar& color, int thickness) {
  // 8 px dash / 6 px gap, walked along each segment
  const double dash = 8.0, gap = 6.0;
  for (std::size_t i = 0; i + 1 < pts.size(); ++i) {
    const cv::Point2d a = pts[i], b = pts[i + 1];
    const double len = cv::norm(b - a);
    if (len < 1e-6) continue;
    const cv::Point2d dir = (b - a) / len;
    for (double s = 0; s < len; s += dash + gap) {
      const cv::Point2d p0 = a + dir * s;
      const cv::Point2d p1 = a + dir * std::min(s + dash, len);
      cv::line(img, p0, p1, color, thickness, cv::LINE_AA);
    }
  }
}

}  // namespace

BevVisualizer::BevVisualizer(const BevConfig& config) : cfg_(config) {
  const int w = static_cast<int>(2 * cfg_.range_lat_m * cfg_.px_per_m);
  const int h = static_cast<int>((cfg_.range_fwd_m + cfg_.range_back_m) * cfg_.px_per_m);
  canvas_.create(h, w, CV_8UC3);
  clear();
}

cv::Point BevVisualizer::toPx(float x_m, float y_m) const {
  return {static_cast<int>(std::lround((cfg_.range_lat_m - y_m) * cfg_.px_per_m)),
          static_cast<int>(std::lround((cfg_.range_fwd_m - x_m) * cfg_.px_per_m))};
}

void BevVisualizer::clear() {
  canvas_.setTo(kBackground);
  legend_rows_ = 0;

  // grid: faint lines every grid_step_m, forward labels every second line
  for (float x = 0; x <= cfg_.range_fwd_m; x += cfg_.grid_step_m) {
    for (const float sx : {x, -x}) {
      if (-sx > cfg_.range_back_m) continue;
      const int py = toPx(sx, 0).y;
      cv::line(canvas_, {0, py}, {canvas_.cols, py}, sx == 0.f ? kAxis : kGrid, 1);
      if (sx > 0 && static_cast<int>(x) % (2 * static_cast<int>(cfg_.grid_step_m)) == 0) {
        cv::putText(canvas_, cv::format("%.0f m", sx), {4, py - 4}, kFont, 0.38,
                    kText, 1, cv::LINE_AA);
      }
    }
  }
  for (float y = 0; y <= cfg_.range_lat_m; y += cfg_.grid_step_m) {
    for (const float sy : {y, -y}) {
      const int px = toPx(0, sy).x;
      cv::line(canvas_, {px, 0}, {px, canvas_.rows}, sy == 0.f ? kAxis : kGrid, 1);
    }
  }

  // ego footprint (typical passenger car, rear-axle-ish origin)
  const cv::Point p0 = toPx(3.6f, 0.95f);
  const cv::Point p1 = toPx(-1.2f, -0.95f);
  cv::rectangle(canvas_, p0, p1, kEgo, 1, cv::LINE_AA);
  const cv::Point nose = toPx(3.6f, 0.f);
  cv::drawMarker(canvas_, {nose.x, nose.y - 6}, kEgo, cv::MARKER_TRIANGLE_UP, 8, 1,
                 cv::LINE_AA);
}

void BevVisualizer::drawTrajectory(const Trajectory& traj, const cv::Scalar& color,
                                   int thickness, bool dashed) {
  std::vector<cv::Point> pts;
  pts.push_back(toPx(0.f, 0.f));
  for (const Waypoint& wp : traj) pts.push_back(toPx(wp.x, wp.y));

  if (dashed) {
    dashedPolyline(canvas_, pts, color, thickness);
  } else {
    cv::polylines(canvas_, pts, false, color, thickness, cv::LINE_AA);
  }
  const float tick_m = 1.2f;
  for (const Waypoint& wp : traj) {
    const cv::Point p = toPx(wp.x, wp.y);
    cv::circle(canvas_, p, thickness + 1, color, cv::FILLED, cv::LINE_AA);
    const cv::Point tip = toPx(wp.x + tick_m * std::cos(wp.heading),
                               wp.y + tick_m * std::sin(wp.heading));
    cv::line(canvas_, p, tip, color, 1, cv::LINE_AA);
  }
}

void BevVisualizer::drawSubgoal(float x_m, float y_m, const cv::Scalar& color) {
  // clamp into view so an out-of-range subgoal still shows direction
  const float x = std::min(x_m, cfg_.range_fwd_m - 1.5f);
  const float y = std::max(std::min(y_m, cfg_.range_lat_m - 1.5f),
                           -cfg_.range_lat_m + 1.5f);
  cv::drawMarker(canvas_, toPx(x, y), color, cv::MARKER_DIAMOND, 14, 2, cv::LINE_AA);
  if (x != x_m || y != y_m) {
    cv::putText(canvas_, cv::format("%.0fm", std::hypot(x_m, y_m)),
                toPx(x, y) + cv::Point(8, 4), kFont, 0.38, color, 1, cv::LINE_AA);
  }
}

void BevVisualizer::drawVelocity(float vel_x, float vel_y) {
  const cv::Point from = toPx(0.f, 0.f);
  const cv::Point to = toPx(vel_x, vel_y);  // 1 m per m/s
  if (from == to) return;
  cv::arrowedLine(canvas_, from, to, kText, 1, cv::LINE_AA, 0, 0.18);
}

void BevVisualizer::addLegend(const std::string& label, const cv::Scalar& color) {
  const int y = 18 + legend_rows_ * 18;
  ++legend_rows_;
  const int x0 = canvas_.cols - 150;
  cv::line(canvas_, {x0, y - 4}, {x0 + 16, y - 4}, color, 3, cv::LINE_AA);
  cv::putText(canvas_, label, {x0 + 22, y}, kFont, 0.42, kText, 1, cv::LINE_AA);
}

}  // namespace latentdrive
