// Unit tests for the status builder: the route points of the route-point
// models (planner_io.ego_status on a 1 m route), the single-subgoal rule, the
// motion terms from differenced poses, and the encoded layout. Plain asserts,
// like test_view_window.cpp
#include <cmath>
#include <cstdio>
#include <vector>

#include "status_builder.hpp"
#include "trajectory.hpp"

static int g_failures = 0;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++g_failures;                                                        \
    }                                                                      \
  } while (0)

#define CHECK_NEAR(a, b, tol)                                                       \
  do {                                                                              \
    const double _a = (a), _b = (b);                                                \
    if (!(std::abs(_a - _b) <= (tol))) {                                            \
      std::fprintf(stderr, "FAIL %s:%d: %s = %.6f, want %.6f (tol %g)\n", __FILE__, \
                   __LINE__, #a, _a, _b, static_cast<double>(tol));                 \
      ++g_failures;                                                                 \
    }                                                                               \
  } while (0)

using status::Point2d;
using status::PoseSample;

// A straight route along +x from (0, 0) to (100, 0), sparse like a lanelet
// centerline
static std::vector<Point2d> StraightRoute() { return {{0, 0}, {40, 0}, {100, 0}}; }

static void TestResampleAndReversals() {
  const auto route_1m = status::ResamplePolyline(StraightRoute(), 1.0);
  CHECK(route_1m.size() == 101);
  CHECK_NEAR(route_1m[37].x, 37.0, 1e-9);
  // A vertex that doubles back (a junction entry kink) is dropped
  const std::vector<Point2d> kinked = {{0, 0}, {10, 0}, {9, 0.5}, {20, 0}};
  const auto                 clean  = status::RemoveReversals(kinked, 120.0 * M_PI / 180.0);
  CHECK(clean.size() == 3);
  CHECK_NEAR(clean[1].x, 9.0, 1e-9);  // the vertex where it turns back goes
}

static void TestRoutePoints() {
  status::StatusBuilderParams params;
  params.num_route_points = 6;
  params.motion_source    = status::MotionSource::kTwist;
  auto builder            = status::CreateStatusBuilder(params);
  CHECK(builder != nullptr);
  builder->SetRoute(StraightRoute());

  // Ego at (10, 1) heading +x: the points are 5..50 m past x = 10, 1 m to the right
  PoseSample ego{0.0, 10.0, 1.0, 0.0, 3.0, 0.0, 0.5, 0.0};
  const auto r = builder->Build(0.0, ego);
  CHECK(r.ok);
  CHECK(r.ego.route_points.size() == 6);
  const double ahead[] = {5, 10, 15, 20, 30, 50};
  for (int k = 0; k < 6; ++k) {
    CHECK_NEAR(r.ego.route_points[k].x_m, ahead[k], 1e-4);
    CHECK_NEAR(r.ego.route_points[k].y_m, -1.0, 1e-4);
  }
  // Clamped at the route end
  PoseSample near_end{0.0, 90.0, 0.0, 0.0, 0, 0, 0, 0};
  const auto e = builder->Build(0.0, near_end);
  CHECK_NEAR(e.ego.route_points[5].x_m, 10.0, 1e-4);

  // Encoded layout [p/10 ..., v, a], 16 values
  const auto enc = r.ego.Encode();
  CHECK(enc.size() == 16);
  CHECK_NEAR(enc[0], 0.5, 1e-6);
  CHECK_NEAR(enc[11], -0.1, 1e-6);
  CHECK_NEAR(enc[12], 3.0, 1e-6);
  CHECK_NEAR(enc[14], 0.5, 1e-6);
  const auto dec = common::EgoStatus::Decode(enc.data(), 16);
  CHECK_NEAR(dec.Subgoal().x_m, 50.0, 1e-4);

  // Rotated ego: heading +y (90 deg) at the origin of a route along +y
  builder->SetRoute({{0, 0}, {0, 100}});
  PoseSample up{0.0, 0.0, 0.0, M_PI / 2, 0, 0, 0, 0};
  const auto u = builder->Build(0.0, up);
  CHECK_NEAR(u.ego.route_points[0].x_m, 5.0, 1e-4);
  CHECK_NEAR(u.ego.route_points[0].y_m, 0.0, 1e-4);
}

static void TestLegacySubgoal() {
  status::StatusBuilderParams params;
  params.num_route_points = 1;
  params.motion_source    = status::MotionSource::kTwist;
  auto builder            = status::CreateStatusBuilder(params);
  CHECK(builder != nullptr);
  CHECK(builder->RoutePointsM().size() == 1 && builder->RoutePointsM()[0] == 50.0);
  builder->SetRoute(StraightRoute());
  // The original rule walks from the nearest route VERTEX, not the projection
  // (on the vehicle's lanelet centerlines that is the same to within a metre)
  const auto r = builder->Build(0.0, PoseSample{0.0, 40.0, 0.0, 0.0, 0, 0, 0, 0});
  CHECK(r.ok && r.ego.Dim() == 6);
  CHECK_NEAR(r.ego.route_points[0].x_m, 50.0, 1e-4);

  // A distance list that does not match the engine is refused
  params.num_route_points = 6;
  params.route_points_m   = {5, 10};
  CHECK(status::CreateStatusBuilder(params) == nullptr);
}

static void TestMotionFromPoses() {
  status::StatusBuilderParams params;
  params.num_route_points = 6;
  auto builder            = status::CreateStatusBuilder(params);
  builder->SetRoute(StraightRoute());
  // 50 Hz odometry, constant acceleration 1 m/s^2 from 2 m/s along +x; the
  // twist fields carry garbage to show they are not used
  const double t0 = 1787830149.0;
  PoseSample   last;
  for (int i = 0; i <= 50; ++i) {
    const double t = 0.02 * i;
    last           = {t0 + t, 2.0 * t + 0.5 * t * t, 0.0, 0.0, -9, -9, -9, -9};
    builder->PushPose(last);
  }
  const auto r = builder->Build(last.stamp_s, last);
  CHECK(r.ok && r.motion_from_poses);
  // chord over [t-0.2, t] -> the speed at t-0.1 = 2 + 0.9; the acceleration exact
  CHECK_NEAR(r.ego.vel_x, 2.9, 1e-3);
  CHECK_NEAR(r.ego.vel_y, 0.0, 1e-6);
  CHECK_NEAR(r.ego.acc_x, 1.0, 1e-3);

  // Without 0.3 s of history the twist stands in
  builder->Reset();
  builder->PushPose(last);
  const auto t = builder->Build(last.stamp_s, last);
  CHECK(!t.motion_from_poses);
  CHECK_NEAR(t.ego.vel_x, -9.0, 1e-6);
}

int main() {
  TestResampleAndReversals();
  TestRoutePoints();
  TestLegacySubgoal();
  TestMotionFromPoses();
  if (g_failures) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("all status builder checks passed\n");
  return 0;
}
