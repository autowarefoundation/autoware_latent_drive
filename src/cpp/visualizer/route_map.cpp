#include "route_map.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <opencv2/imgproc.hpp>

namespace visualizer {

// Same chrome as the BEV panel so the two read as one instrument
static const cv::Scalar kBackground{25, 26, 26};
static const cv::Scalar kRoute{58, 62, 62};        // the route not yet driven
static const cv::Scalar kDriven{188, 188, 188};    // the path already behind
static const cv::Scalar kSubgoal{52, 104, 235};    // orange, matching the BEV
static const cv::Scalar kMilestone{150, 190, 200};
static const cv::Scalar kGoal{120, 210, 90};       // green, matching the HUD
static const cv::Scalar kEgo{255, 255, 255};
static const cv::Scalar kText{183, 194, 195};

constexpr static int kFont = cv::FONT_HERSHEY_SIMPLEX;

// @brief: This function is used to accumulate the arc length of a polyline
static std::vector<double> ArcLengths(const std::vector<cv::Point2d> &in_points) {
  std::vector<double> s(in_points.size(), 0.0);
  for (std::size_t i = 1; i < in_points.size(); ++i) {
    s[i] = s[i - 1] + cv::norm(in_points[i] - in_points[i - 1]);
  }
  return s;
}

// `RouteMapImpl` is an implementation class, which reconstructs the route once
// at construction and implements the per-step drawing
class RouteMapImpl : public RouteMap {
 public:
  RouteMapImpl(const std::vector<MapPose> &poses, const std::vector<cv::Point2d> &subgoal_ego,
               const RouteParams &params)
      : params_(params) {
    // Every subgoal the clip issues, lifted from the ego frame into the map
    const std::size_t n = std::min(poses.size(), subgoal_ego.size());
    for (std::size_t k = 0; k < n; ++k) {
      const double c = std::cos(poses[k].yaw);
      const double s = std::sin(poses[k].yaw);
      driven_.emplace_back(poses[k].x, poses[k].y);
      subgoals_.emplace_back(poses[k].x + c * subgoal_ego[k].x - s * subgoal_ego[k].y,
                             poses[k].y + s * subgoal_ego[k].x + c * subgoal_ego[k].y);
    }
    if (driven_.empty()) return;
    driven_s_ = ArcLengths(driven_);

    BuildRoute();
    BuildMilestones();
    BuildView();
  }

  RouteMapImpl(const RouteMapImpl &)            = delete;
  RouteMapImpl &operator=(const RouteMapImpl &) = delete;
  ~RouteMapImpl() override                      = default;

  bool                               HasGoal() const final { return has_goal_; }
  double                             RouteLength() const final { return route_s_.empty() ? 0.0 : route_s_.back(); }
  const std::vector<RouteMilestone> &Milestones() const final { return milestones_; }

  double ArcAt(std::size_t in_step) const final {
    if (driven_s_.empty()) return 0.0;
    return driven_s_[std::min(in_step, driven_s_.size() - 1)];
  }

  cv::Mat Draw(cv::Size in_size, std::size_t in_step) const final {
    cv::Mat canvas(in_size, CV_8UC3, kBackground);
    if (driven_.empty()) return canvas;

    // A near-straight route leaves the top and bottom of an equal-aspect map
    // empty, so the strip below it carries the same route unrolled onto a
    // single axis -- that is where the milestone spacing is actually readable
    const int bar_h  = milestones_.empty() ? 0 : 52;
    const int map_h  = in_size.height - bar_h;

    // The extent is fitted once for the whole clip, so the map holds still
    // while the ego moves across it; only the metres-per-pixel depends on the
    // panel the caller asked for
    constexpr int kMargin = 16;
    scale_  = std::min((in_size.width - 2.0 * kMargin) / span_.x,
                       (map_h - 2.0 * kMargin) / span_.y);
    origin_ = {0.5 * in_size.width, 0.5 * map_h};

    const auto px = [&](const cv::Point2d &p) {
      const cv::Point2d r = Rotate(p - center_) - offset_;
      return cv::Point(static_cast<int>(std::lround(origin_.x + scale_ * r.x)),
                       static_cast<int>(std::lround(origin_.y - scale_ * r.y)));
    };

    Polyline(canvas, route_, px, kRoute, 2);

    const std::size_t k = std::min(in_step, driven_.size() - 1);
    Polyline(canvas, {driven_.begin(), driven_.begin() + static_cast<long>(k) + 1}, px, kDriven, 2);

    // Every subgoal the clip issues, thinned out so they read as a sequence of
    // points rather than a second line on top of the route
    const std::size_t stride = std::max<std::size_t>(subgoals_.size() / 60, 1);
    for (std::size_t i = 0; i < subgoals_.size(); i += stride) {
      cv::circle(canvas, px(subgoals_[i]), 1, kSubgoal, cv::FILLED, cv::LINE_AA);
    }

    // The map carries bare markers; the strip below it is where the milestones
    // are numbered, and repeating the labels here only crowds the route
    for (const RouteMilestone &m : milestones_) {
      const cv::Point p = px(m.xy);
      if (m.is_goal) {
        cv::rectangle(canvas, p - cv::Point(4, 4), p + cv::Point(4, 4), kGoal, cv::FILLED,
                      cv::LINE_AA);
        const int w = cv::getTextSize("GOAL", kFont, 0.34, 1, nullptr).width;
        const int x = (p.x + 8 + w < canvas.cols) ? p.x + 8 : p.x - 8 - w;
        cv::putText(canvas, "GOAL", {x, p.y + 4}, kFont, 0.34, kGoal, 1, cv::LINE_AA);
      } else {
        cv::circle(canvas, p, 3, m.s_m <= ArcAt(k) ? kRoute : kMilestone, cv::FILLED, cv::LINE_AA);
      }
    }

    // Current subgoal and the ego, drawn last so nothing hides them
    cv::drawMarker(canvas, px(subgoals_[k]), kSubgoal, cv::MARKER_DIAMOND, 9, 2, cv::LINE_AA);
    const cv::Point ego = px(driven_[k]);
    cv::circle(canvas, ego, 4, kEgo, cv::FILLED, cv::LINE_AA);
    cv::circle(canvas, ego, 7, kEgo, 1, cv::LINE_AA);

    // Without a destination the polyline is just the path the ego drove, and
    // calling that "route" would overstate what the clip knows
    cv::putText(canvas,
                cv::format(has_goal_ ? "%.0f m route" : "%.0f m driven", RouteLength()), {6, 14},
                kFont, 0.34, kText, 1, cv::LINE_AA);

    if (bar_h > 0) DrawProgressBar(canvas, in_size.height - bar_h, bar_h, ArcAt(k));
    return canvas;
  }

 private:
  // @brief: This function is used to draw the route unrolled onto one axis:
  // the milestones at their true spacing, the destination at the far end, and
  // the ego sliding along it
  void DrawProgressBar(cv::Mat &canvas, int in_top, int in_height, double in_done) const {
    const int x0 = 16;
    const int x1 = canvas.cols - 16;
    const int y  = in_top + in_height / 2 - 4;

    const auto bx = [&](double s) {
      return x0 + static_cast<int>(std::lround(std::clamp(s / RouteLength(), 0.0, 1.0) * (x1 - x0)));
    };

    cv::line(canvas, {x0, y}, {x1, y}, kRoute, 3, cv::LINE_AA);
    cv::line(canvas, {x0, y}, {bx(in_done), y}, kDriven, 3, cv::LINE_AA);

    for (const RouteMilestone &m : milestones_) {
      const int  mx     = bx(m.s_m);
      const bool passed = m.s_m <= in_done;
      if (m.is_goal) {
        cv::rectangle(canvas, {mx - 4, y - 4}, {mx + 4, y + 4}, kGoal, cv::FILLED, cv::LINE_AA);
      } else {
        cv::circle(canvas, {mx, y}, 3, passed ? kRoute : kMilestone, cv::FILLED, cv::LINE_AA);
        cv::putText(canvas, cv::format("%d", m.label), {mx - 3, y + 20}, kFont, 0.30,
                    passed ? kRoute : kMilestone, 1, cv::LINE_AA);
      }
    }

    const int ex = bx(in_done);
    cv::drawMarker(canvas, {ex, y - 9}, kEgo, cv::MARKER_TRIANGLE_DOWN, 9, 2, cv::LINE_AA);
  }

  // @brief: This function is used to stitch the driven path and the part of
  // the subgoal trace that runs past its end into one route polyline
  void BuildRoute() {
    route_ = driven_;

    std::size_t join = 0;
    double      best = std::numeric_limits<double>::max();
    for (std::size_t i = 0; i < subgoals_.size(); ++i) {
      const double d = cv::norm(subgoals_[i] - driven_.back());
      if (d < best) {
        best = d;
        join = i;
      }
    }

    // The trace only reaches the ego's final position once the lookahead has
    // run out of route to look at, which is exactly when a destination exists
    has_goal_ = best <= params_.join_tolerance_m;
    if (has_goal_) {
      route_.insert(route_.end(), subgoals_.begin() + static_cast<long>(join) + 1, subgoals_.end());
    }
    route_s_ = ArcLengths(route_);
  }

  void BuildMilestones() {
    if (!has_goal_ || RouteLength() < params_.min_route_m) return;

    double step = params_.milestone_step_m;
    if (step <= 0.0) {
      // Whichever round spacing lands closest to six markers
      const double kChoices[] = {10.0, 25.0, 50.0, 100.0, 250.0};
      step                    = kChoices[0];
      double best             = std::numeric_limits<double>::max();
      for (const double c : kChoices) {
        const double miss = std::abs(RouteLength() / c - 6.0);
        if (miss < best) {
          best = miss;
          step = c;
        }
      }
    }

    int label = 1;
    for (double s = step; s < RouteLength() - 0.5 * step; s += step) {
      milestones_.push_back({PointAt(s), s, label++, false});
    }
    milestones_.push_back({route_.back(), RouteLength(), 0, true});
  }

  // @brief: This function is used to interpolate the route at an arc position
  cv::Point2d PointAt(double in_s) const {
    const auto it = std::lower_bound(route_s_.begin(), route_s_.end(), in_s);
    if (it == route_s_.begin()) return route_.front();
    if (it == route_s_.end()) return route_.back();
    const std::size_t i   = static_cast<std::size_t>(it - route_s_.begin());
    const double      len = route_s_[i] - route_s_[i - 1];
    const double      t   = len > 1e-9 ? (in_s - route_s_[i - 1]) / len : 0.0;
    return route_[i - 1] + (route_[i] - route_[i - 1]) * t;
  }

  // @brief: This function is used to fit the whole clip -- route, subgoals and
  // driven path -- into the panel once, with equal scale on both axes. The map
  // is turned so the route's long axis lies across the panel and runs left to
  // right: a route that happens to head northeast would otherwise be drawn as
  // a diagonal through a mostly empty box
  void BuildView() {
    cv::Point2d mean{0, 0};
    for (const cv::Point2d &p : route_) mean += p;
    mean *= 1.0 / static_cast<double>(route_.size());

    double cxx = 0, cyy = 0, cxy = 0;
    for (const cv::Point2d &p : route_) {
      const cv::Point2d d = p - mean;
      cxx += d.x * d.x;
      cyy += d.y * d.y;
      cxy += d.x * d.y;
    }
    rot_ = -0.5 * std::atan2(2.0 * cxy, cxx - cyy);

    // Flip end for end when the route would otherwise run backwards
    if (Rotate(route_.back() - mean).x < Rotate(route_.front() - mean).x) rot_ += CV_PI;

    center_ = mean;

    double lo_x = std::numeric_limits<double>::max(), hi_x = -lo_x;
    double lo_y = lo_x, hi_y = hi_x;
    for (const auto &set : {route_, subgoals_}) {
      for (const cv::Point2d &p : set) {
        const cv::Point2d r = Rotate(p - mean);
        lo_x = std::min(lo_x, r.x);
        hi_x = std::max(hi_x, r.x);
        lo_y = std::min(lo_y, r.y);
        hi_y = std::max(hi_y, r.y);
      }
    }
    // Re-centre on the rotated extent, so the fit has no wasted margin
    offset_ = {0.5 * (lo_x + hi_x), 0.5 * (lo_y + hi_y)};
    span_   = {std::max(hi_x - lo_x, 1.0), std::max(hi_y - lo_y, 1.0)};
  }

  cv::Point2d Rotate(const cv::Point2d &in_p) const {
    const double c = std::cos(rot_), s = std::sin(rot_);
    return {c * in_p.x - s * in_p.y, s * in_p.x + c * in_p.y};
  }

  template <typename Px>
  static void Polyline(cv::Mat &canvas, const std::vector<cv::Point2d> &pts, const Px &px,
                       const cv::Scalar &color, int thickness) {
    for (std::size_t i = 0; i + 1 < pts.size(); ++i) {
      cv::line(canvas, px(pts[i]), px(pts[i + 1]), color, thickness, cv::LINE_AA);
    }
  }

  RouteParams                 params_;
  std::vector<cv::Point2d>    driven_;
  std::vector<double>         driven_s_;
  std::vector<cv::Point2d>    subgoals_;
  std::vector<cv::Point2d>    route_;
  std::vector<double>         route_s_;
  std::vector<RouteMilestone> milestones_;
  bool                        has_goal_ = false;

  cv::Point2d center_{0, 0};  // map-frame point the rotation turns about
  cv::Point2d offset_{0, 0};  // centre of the rotated extent
  cv::Point2d span_{1, 1};
  double      rot_ = 0.0;

  // Filled in by Draw, which is the first place the panel size is known
  mutable double      scale_ = 1.0;
  mutable cv::Point2d origin_{0, 0};
};

std::shared_ptr<RouteMap> CreateRouteMap(const std::vector<MapPose>     &in_poses,
                                         const std::vector<cv::Point2d> &in_subgoal_ego,
                                         const RouteParams              &params) {
  return std::shared_ptr<RouteMap>(new RouteMapImpl(in_poses, in_subgoal_ego, params));
}

}  // namespace visualizer
