#include "status_builder.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>

#include "logger.hpp"

namespace status {

namespace {

double Norm(const Point2d &a, const Point2d &b) { return std::hypot(a.x - b.x, a.y - b.y); }

Point2d Lerp(const Point2d &a, const Point2d &b, double t) {
  return {a.x + t * (b.x - a.x), a.y + t * (b.y - a.y)};
}

// A stamp this far behind the previous one means the source restarted
constexpr double kStampJumpBackS = 1.0;
// A projection this far from the hinted vertex is a relocalisation or a new route
constexpr double kHintResetM = 10.0;

}  // namespace

std::vector<Point2d> RemoveReversals(const std::vector<Point2d> &points, double max_turn_rad) {
  std::vector<Point2d> out;
  for (const Point2d &p : points) {
    if (out.empty() || Norm(p, out.back()) > 1e-3) out.push_back(p);
  }
  if (max_turn_rad <= 0.0) return out;
  bool removed = true;
  while (removed && out.size() > 2) {
    removed = false;
    for (std::size_t i = 1; i + 1 < out.size(); ++i) {
      const double ax   = out[i].x - out[i - 1].x;
      const double ay   = out[i].y - out[i - 1].y;
      const double bx   = out[i + 1].x - out[i].x;
      const double by   = out[i + 1].y - out[i].y;
      const double turn = std::abs(std::atan2(ax * by - ay * bx, ax * bx + ay * by));
      if (turn > max_turn_rad) {
        out.erase(out.begin() + static_cast<std::ptrdiff_t>(i));
        removed = true;
        break;
      }
    }
  }
  return out;
}

std::vector<Point2d> ResamplePolyline(const std::vector<Point2d> &points, double step) {
  if (points.empty() || step <= 0.0) return {};
  std::vector<Point2d> unique{points.front()};
  for (std::size_t i = 1; i < points.size(); ++i) {
    if (Norm(points[i], unique.back()) > 1e-3) unique.push_back(points[i]);
  }
  if (unique.size() == 1) return unique;
  std::vector<double> s(unique.size(), 0.0);
  for (std::size_t i = 1; i < unique.size(); ++i) s[i] = s[i - 1] + Norm(unique[i], unique[i - 1]);

  std::vector<Point2d> out;
  std::size_t          j = 0;
  for (double g = 0.0; g <= s.back() + 1e-6; g += step) {
    while (j + 1 < unique.size() - 1 && s[j + 1] < g) ++j;
    const double span = s[j + 1] - s[j];
    const double u    = span > 0.0 ? std::clamp((g - s[j]) / span, 0.0, 1.0) : 0.0;
    out.push_back(Lerp(unique[j], unique[j + 1], u));
  }
  return out;
}

std::size_t ProjectArclength(const std::vector<Point2d> &route_1m, const Point2d &position,
                             std::optional<std::size_t> hint, std::size_t window) {
  if (route_1m.empty()) return 0;
  const std::size_t lo     = hint ? (*hint > 2 ? *hint - 2 : 0) : 0;
  const std::size_t hi     = hint ? std::min(lo + window, route_1m.size()) : route_1m.size();
  std::size_t       best   = std::min(lo, route_1m.size() - 1);
  double            best_d = std::numeric_limits<double>::infinity();
  for (std::size_t i = lo; i < hi; ++i) {
    const double d = Norm(route_1m[i], position);
    if (d < best_d) {
      best_d = d;
      best   = i;
    }
  }
  return best;
}

Point2d PointAhead(const std::vector<Point2d> &route_1m, std::size_t s_now, double ahead_m) {
  if (route_1m.empty()) return {};
  const double      last = static_cast<double>(route_1m.size() - 1);
  const double      s    = std::min(static_cast<double>(s_now) + ahead_m, last);
  const auto        j    = static_cast<std::size_t>(std::floor(s));
  const double      f    = s - static_cast<double>(j);
  const std::size_t k    = std::min(j + 1, route_1m.size() - 1);
  return Lerp(route_1m[j], route_1m[k], f);
}

Point2d SubgoalAlong(const std::vector<Point2d> &reference, const Point2d &position, double ahead_m) {
  if (reference.empty()) return position;
  std::size_t nearest = 0;
  double      best    = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < reference.size(); ++i) {
    const double d = Norm(reference[i], position);
    if (d < best) {
      best    = d;
      nearest = i;
    }
  }
  // Interpolate inside the segment that crosses the distance; a sparse
  // reference (lanelet centerlines every few metres) would otherwise snap the
  // subgoal to a vertex
  double travelled = 0.0;
  for (std::size_t j = nearest; j + 1 < reference.size(); ++j) {
    const double segment = Norm(reference[j + 1], reference[j]);
    if (travelled + segment >= ahead_m) {
      return Lerp(reference[j], reference[j + 1], segment > 0.0 ? (ahead_m - travelled) / segment : 0.0);
    }
    travelled += segment;
  }
  return reference.back();
}

std::array<double, 4> MotionFromPoses(const std::array<PoseSample, 4> &s, double yaw_now) {
  const double c  = std::cos(yaw_now);
  const double sn = std::sin(yaw_now);
  // Chord between two samples, in the current ego frame, over the real time span
  const auto chord = [&](const PoseSample &a, const PoseSample &b) {
    const double span = std::max(b.stamp_s - a.stamp_s, 1e-3);
    const double dx   = (b.x - a.x) / span;
    const double dy   = (b.y - a.y) / span;
    return std::array<double, 2>{c * dx + sn * dy, -sn * dx + c * dy};
  };
  const auto   v_now  = chord(s[1], s[3]);
  const auto   v_prev = chord(s[0], s[2]);
  const double step   = std::max(s[3].stamp_s - s[2].stamp_s, 1e-3);
  return {v_now[0], v_now[1], (v_now[0] - v_prev[0]) / step, (v_now[1] - v_prev[1]) / step};
}

std::optional<std::array<PoseSample, 4>> PickMotionSamples(const std::deque<PoseSample> &history,
                                                           double now_s, double dt_s,
                                                           double tolerance_s) {
  if (history.empty() || dt_s <= 0.0) return std::nullopt;
  std::array<PoseSample, 4> out{};
  for (std::size_t k = 0; k < out.size(); ++k) {
    const double      target = now_s - static_cast<double>(3 - k) * dt_s;
    const PoseSample *best   = nullptr;
    for (const PoseSample &sample : history) {
      if (!best || std::abs(sample.stamp_s - target) < std::abs(best->stamp_s - target)) best = &sample;
    }
    if (std::abs(best->stamp_s - target) > tolerance_s) return std::nullopt;
    out[k] = *best;
  }
  return out;
}

// `StatusBuilderImpl` is an implementation class. It owns the 1 m route, the
// projection hint and the pose history
class StatusBuilderImpl : public StatusBuilder {
 public:
  explicit StatusBuilderImpl(const StatusBuilderParams &params) : params_(params) {}

  void SetRoute(const std::vector<Point2d> &route) final {
    route_raw_ = RemoveReversals(route, params_.route_reversal_deg * M_PI / 180.0);
    route_1m_  = ResamplePolyline(route_raw_, 1.0);
    hint_.reset();
  }

  bool HasRoute() const final { return !route_1m_.empty(); }

  void PushPose(const PoseSample &sample) final {
    if (!history_.empty() && sample.stamp_s < history_.back().stamp_s - kStampJumpBackS) {
      history_.clear();
      hint_.reset();
    }
    history_.push_back(sample);
    while (!history_.empty() && history_.front().stamp_s < sample.stamp_s - params_.retain_s) {
      history_.pop_front();
    }
  }

  StatusResult Build(double stamp_s, const PoseSample &ego) final {
    StatusResult result;
    if (route_1m_.empty()) {
      result.error = "no route";
      return result;
    }
    const Point2d ego_xy{ego.x, ego.y};
    if (params_.num_route_points == 1) {
      result.route_points_map.push_back(SubgoalAlong(route_raw_, ego_xy, params_.route_points_m.front()));
    } else {
      std::size_t s_now = ProjectArclength(route_1m_, ego_xy, hint_);
      if (hint_ && Norm(route_1m_[s_now], ego_xy) > kHintResetM) {
        s_now = ProjectArclength(route_1m_, ego_xy, std::nullopt);
      }
      hint_ = s_now;
      for (const double ahead : params_.route_points_m) {
        result.route_points_map.push_back(PointAhead(route_1m_, s_now, ahead));
      }
    }
    return Finish(stamp_s, ego, std::move(result));
  }

  StatusResult BuildWithRoutePoints(double stamp_s, const PoseSample &ego,
                                    const std::vector<Point2d> &route_points_map) final {
    StatusResult result;
    if (static_cast<int>(route_points_map.size()) != params_.num_route_points) {
      result.error = "got " + std::to_string(route_points_map.size()) + " route points, the engine takes " +
                     std::to_string(params_.num_route_points);
      return result;
    }
    result.route_points_map = route_points_map;
    return Finish(stamp_s, ego, std::move(result));
  }

  const std::vector<double> &RoutePointsM() const final { return params_.route_points_m; }

  void Reset() final {
    history_.clear();
    hint_.reset();
  }

 private:
  // @brief: Route points into the ego frame, then the motion terms
  StatusResult Finish(double stamp_s, const PoseSample &ego, StatusResult result) {
    const double c = std::cos(ego.yaw);
    const double s = std::sin(ego.yaw);
    for (const Point2d &p : result.route_points_map) {
      const double dx = p.x - ego.x;
      const double dy = p.y - ego.y;
      result.ego.route_points.push_back(
          {static_cast<float>(c * dx + s * dy), static_cast<float>(-s * dx + c * dy)});
    }

    std::optional<std::array<PoseSample, 4>> samples;
    if (params_.motion_source == MotionSource::kPoseDifference) {
      samples = PickMotionSamples(history_, stamp_s, params_.motion_dt_s, params_.motion_tolerance_s);
    }
    if (samples) {
      const std::array<double, 4> m = MotionFromPoses(*samples, ego.yaw);
      result.ego.vel_x              = static_cast<float>(m[0]);
      result.ego.vel_y              = static_cast<float>(m[1]);
      result.ego.acc_x              = static_cast<float>(m[2]);
      result.ego.acc_y              = static_cast<float>(m[3]);
      result.motion_from_poses      = true;
    } else {
      // Twist requested, or the pose history does not cover 0.3 s yet
      result.ego.vel_x = static_cast<float>(ego.vel_x);
      result.ego.vel_y = static_cast<float>(ego.vel_y);
      result.ego.acc_x = static_cast<float>(ego.acc_x);
      result.ego.acc_y = static_cast<float>(ego.acc_y);
    }
    result.ok = true;
    return result;
  }

  StatusBuilderParams        params_;
  std::vector<Point2d>       route_raw_;
  std::vector<Point2d>       route_1m_;
  std::optional<std::size_t> hint_;
  std::deque<PoseSample>     history_;
};

std::shared_ptr<StatusBuilder> CreateStatusBuilder(const StatusBuilderParams &in_params) {
  StatusBuilderParams params = in_params;
  if (params.num_route_points < 1) {
    LOGE("StatusBuilder needs at least one route point, got %d", params.num_route_points);
    return nullptr;
  }
  if (params.route_points_m.empty()) {
    if (params.num_route_points == 1) {
      params.route_points_m = {model::planner::kLegacySubgoalAheadM};
    } else if (params.num_route_points ==
               static_cast<int>(std::size(model::planner::kDefaultRoutePointsM))) {
      params.route_points_m.assign(std::begin(model::planner::kDefaultRoutePointsM),
                                   std::end(model::planner::kDefaultRoutePointsM));
    }
  }
  if (static_cast<int>(params.route_points_m.size()) != params.num_route_points) {
    LOGE("The engine takes %d route points but %zu distances were given", params.num_route_points,
         params.route_points_m.size());
    return nullptr;
  }
  if (params.motion_dt_s <= 0.0 || params.motion_tolerance_s < 0.0 ||
      params.retain_s < 3.0 * params.motion_dt_s + params.motion_tolerance_s) {
    LOGE("StatusBuilder: invalid motion sampling (dt %.3f s, tolerance %.3f s, retain %.3f s)",
         params.motion_dt_s, params.motion_tolerance_s, params.retain_s);
    return nullptr;
  }
  return std::shared_ptr<StatusBuilder>(new StatusBuilderImpl(params));
}

}  // namespace status
