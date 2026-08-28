#include "trajectory_smoother.hpp"

#include <cmath>
#include <vector>

namespace visualizer {

// @brief: This function is used to wrap an angle difference into (-pi, pi] so
// headings blend the short way round
static float WrapPi(float in_rad) {
  return std::atan2(std::sin(in_rad), std::cos(in_rad));
}

// @brief: This function is used to sample a plan in waypoint-index space:
// u = -1 is the ego origin the plan starts from, u = i is waypoint i, and
// fractional u interpolates. Beyond the last waypoint the final segment is
// extended, which is what keeps the tail alive when the carried-forward plan
// is asked for a time the old plan did not reach
static common::Waypoint SampleAt(const common::Trajectory &in_traj, double in_u) {
  const int n = static_cast<int>(in_traj.size());

  const auto at = [&](int i) -> common::Waypoint {
    if (i < 0) return {0.f, 0.f, 0.f};
    if (i < n) return in_traj[i];
    if (n < 2) return in_traj[n - 1];
    const common::Waypoint &a = in_traj[n - 2];
    const common::Waypoint &b = in_traj[n - 1];
    const float             s = static_cast<float>(i - (n - 1));
    return {b.x + s * (b.x - a.x), b.y + s * (b.y - a.y), b.heading};
  };

  if (in_u < -1.0) in_u = -1.0;
  const int              i = static_cast<int>(std::floor(in_u));
  const float            t = static_cast<float>(in_u - i);
  const common::Waypoint a = at(i);
  const common::Waypoint b = at(i + 1);
  return {a.x + t * (b.x - a.x), a.y + t * (b.y - a.y),
          a.heading + t * WrapPi(b.heading - a.heading)};
}

// @brief: This function is used to express `in_wp` in the frame of `in_origin`
static common::Waypoint ToLocal(const common::Waypoint &in_wp, const common::Waypoint &in_origin) {
  const float dx = in_wp.x - in_origin.x;
  const float dy = in_wp.y - in_origin.y;
  const float c  = std::cos(in_origin.heading);
  const float s  = std::sin(in_origin.heading);
  return {c * dx + s * dy, -s * dx + c * dy, WrapPi(in_wp.heading - in_origin.heading)};
}

// `TrajectorySmootherImpl` is an implementation class, which holds the last
// displayed plan and implements the carry-forward blend
class TrajectorySmootherImpl : public TrajectorySmoother {
 public:
  explicit TrajectorySmootherImpl(const SmootherParams &params) : params_(params) {}

  TrajectorySmootherImpl(const TrajectorySmootherImpl &)            = delete;
  TrajectorySmootherImpl &operator=(const TrajectorySmootherImpl &) = delete;
  ~TrajectorySmootherImpl() override                                = default;

  common::Trajectory Update(const common::Trajectory &in_traj, double in_dt_s) final {
    // Pass-through cases: filtering off, no state yet, or a plan whose shape
    // the state cannot describe
    if (params_.alpha >= 1.0 || in_traj.empty() || prev_.size() != in_traj.size()) {
      prev_ = in_traj;
      return in_traj;
    }

    const common::Trajectory carried = CarryForward(prev_, in_traj.size(), in_dt_s);

    // A large disagreement is the plan changing its mind, not jitter; smoothing
    // through it would drag a stale trajectory across the canvas for several
    // frames, which looks worse than the jump it hides
    const common::Waypoint &a = carried.back();
    const common::Waypoint &b = in_traj.back();
    if (std::hypot(b.x - a.x, b.y - a.y) > params_.reset_jump_m) {
      prev_ = in_traj;
      return in_traj;
    }

    const float        w = static_cast<float>(params_.alpha);
    common::Trajectory out(in_traj.size());
    for (std::size_t i = 0; i < in_traj.size(); ++i) {
      out[i].x       = carried[i].x + w * (in_traj[i].x - carried[i].x);
      out[i].y       = carried[i].y + w * (in_traj[i].y - carried[i].y);
      out[i].heading = carried[i].heading + w * WrapPi(in_traj[i].heading - carried[i].heading);
    }

    prev_ = out;
    return out;
  }

  void Reset() final { prev_.clear(); }

 private:
  // @brief: This function is used to re-anchor a plan on the ego pose it
  // predicted for +dt and resample it onto the same time grid, so the result
  // is directly comparable with the plan that arrives dt later
  common::Trajectory CarryForward(const common::Trajectory &in_traj, std::size_t in_size,
                                  double in_dt_s) const {
    // Waypoint spacing on the time axis; `shift` is dt measured in waypoints
    const double dt_wp_s = params_.horizon_s / static_cast<double>(in_traj.size());
    const double shift   = (dt_wp_s > 0.0) ? (in_dt_s / dt_wp_s) : 0.0;

    // u = -1 is where the ego was, so u = -1 + shift is where it is now
    const common::Waypoint origin = SampleAt(in_traj, -1.0 + shift);

    common::Trajectory out(in_size);
    for (std::size_t i = 0; i < in_size; ++i) {
      out[i] = ToLocal(SampleAt(in_traj, static_cast<double>(i) + shift), origin);
    }
    return out;
  }

  SmootherParams     params_;
  common::Trajectory prev_;
};

std::shared_ptr<TrajectorySmoother> CreateTrajectorySmoother(const SmootherParams &params) {
  return std::shared_ptr<TrajectorySmoother>(new TrajectorySmootherImpl(params));
}

}  // namespace visualizer
