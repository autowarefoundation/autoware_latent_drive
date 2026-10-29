// Sample: run the LatentDrive planner on a sample clip, offline, with the
// library API alone (no ROS).
//
//   latentdrive-sample-clip --engine_file=E.engine --clip_dir=CLIP
//       [--out_video=CLIP/replay.mp4] [--route_source=recorded|file]
//
// A clip is what export_clip.py cuts out of a rosbag: per-view JPEG frames
// with their stamps, the 50 Hz odometry, a route polyline and (if the bag had
// them) the route points the vehicle fed its model. The loop below is the
// whole deployment pipeline in the order a node runs it:
//
//   1. Push every camera frame into the ViewWindow as it "arrives" -- it is
//      preprocessed once there (crop, resize, normalize).
//   2. Push every odometry sample into the StatusBuilder.
//   3. On each view-0 frame (paced to 10 Hz on its stamps): build the status
//      (route points + motion terms), assemble the [V, 3, 4, H, W] window
//      straight into the engine's pinned input, and run the plan.
//
// The engine sets the contract: 1 or 3 views, status [1, 6] or [1, 16], 8 or
// 40 waypoints. The output is <clip>/plans_<engine>.npy, float32 [T, N, 3]
// (x forward, y left, heading in the ego frame at each step), with the step
// stamps in <clip>/plans_<engine>_t.npy, and optionally an mp4 of the camera
// ribbon + BEV.
#include <gflags/gflags.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <optional>
#include <string>
#include <vector>

#include "bev_visualizer.hpp"
#include "camera_projector.hpp"
#include "color_map.hpp"
#include "frame_window.hpp"
#include "logger.hpp"
#include "npy.hpp"
#include "planner.hpp"
#include "status_builder.hpp"
#include "trajectory.hpp"

namespace fs = std::filesystem;

DEFINE_string(engine_file, "", "Serialized TensorRT engine (latentdrive-trt --mode=build-engine)");
DEFINE_string(clip_dir, "", "Clip directory written by export_clip.py");
DEFINE_string(route_source, "",
              "recorded: the vehicle's route points (route_points.npy); file: build them from route.npy. "
              "Default: file when route.npy is the mission-planning route (export_clip.py --lanelet_map), "
              "else recorded when the clip has them");
DEFINE_string(missing_view_fill, "skip",
              "A side-view slot without a frame within --view_tolerance_s: skip the step or zeros "
              "(the blank frame the camfront3 training used)");
DEFINE_double(view_tolerance_s, model::planner::kDefaultViewToleranceS,
              "Max |side stamp - front stamp| per history slot");
DEFINE_double(max_rate_hz, 10.0, "Plans per second at most, paced on the view-0 stamps (0 = every frame)");
DEFINE_string(out_video, "", "Also render the camera ribbon + BEV to this mp4");

namespace {

struct ClipFrame {
  double      t_s;
  std::string file;
};

// The recorded route points nearest to `t_s` (the vehicle stamped them with
// its tick, a few ms after the camera frame), if within 0.3 s
std::optional<std::vector<status::Point2d>> RecordedAt(const utils::NpyArray &rp, double t_s) {
  if (rp.shape.size() != 2 || rp.shape[0] == 0) return std::nullopt;
  const std::size_t cols = rp.shape[1];
  std::size_t       best = 0;
  for (std::size_t i = 1; i < rp.shape[0]; ++i) {
    if (std::abs(rp.data[i * cols] - t_s) < std::abs(rp.data[best * cols] - t_s)) best = i;
  }
  if (std::abs(rp.data[best * cols] - t_s) > 0.3) return std::nullopt;
  std::vector<status::Point2d> points;
  for (std::size_t c = 1; c + 1 < cols; c += 2) {
    points.push_back({rp.data[best * cols + c], rp.data[best * cols + c + 1]});
  }
  return points;
}

std::shared_ptr<visualizer::CameraProjector> ProjectorFrom(const nlohmann::json &camera) {
  if (camera.is_null()) return nullptr;
  visualizer::CameraParams p;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      p.k(r, c) = camera["K_crop_resized"][r][c].get<double>();
      p.r(r, c) = camera["extrinsics_camera_to_ego"]["rotation_matrix"][r][c].get<double>();
    }
    p.t[r] = camera["extrinsics_camera_to_ego"]["translation"][r].get<double>();
  }
  return visualizer::CreateCameraProjector(p);
}

// Camera panel (view 0 with the plan as a ribbon) beside the BEV (plan, route
// points, velocity)
cv::Mat Render(const cv::Mat &front_bgr, const common::Trajectory &plan, const common::EgoStatus &ego,
               visualizer::CameraProjector *projector, visualizer::BevVisualizer &bev, int num_views,
               double t_s, bool blank_view) {
  cv::Mat camera;
  const int band = front_bgr.cols / 2;
  const cv::Mat cropped =
      front_bgr.rows > band ? front_bgr(cv::Rect(0, (front_bgr.rows - band) / 2, front_bgr.cols, band)) : front_bgr;
  cv::resize(cropped, camera, cv::Size(model::planner::kInputWidth, model::planner::kInputHeight));
  if (projector) projector->DrawRibbon(camera, plan, 0.9, 0.5);

  bev.Clear();
  bev.DrawTrajectoryHorizon(plan, 2);
  for (const common::RoutePoint &p : ego.route_points) bev.DrawSubgoal(p.x_m, p.y_m);
  bev.DrawVelocity(ego.vel_x, ego.vel_y);
  bev.AddLegend("plan now->+4s", visualizer::HorizonColor(0.0));
  bev.AddLegend(ego.route_points.size() == 1 ? "subgoal 50m" : "route points", visualizer::color::kSubgoal);

  const int H = 560;
  cv::Mat   cam_panel, bev_panel;
  cv::resize(camera, cam_panel, cv::Size(H * camera.cols / camera.rows, H));
  cv::resize(bev.Image(), bev_panel, cv::Size(H * bev.Image().cols / bev.Image().rows, H), 0, 0, cv::INTER_AREA);
  cv::putText(cam_panel,
              cv::format("t %.1f s  %dv  v %.1f km/h  |4s| %.1f m%s", t_s, num_views,
                         std::hypot(ego.vel_x, ego.vel_y) * 3.6, std::hypot(plan.back().x, plan.back().y),
                         blank_view ? "  (blank side view)" : ""),
              {12, 28}, cv::FONT_HERSHEY_DUPLEX, 0.6, {230, 230, 230}, 1, cv::LINE_AA);
  cv::Mat frame(H, cam_panel.cols + 2 + bev_panel.cols, CV_8UC3, cv::Scalar(45, 47, 47));
  cam_panel.copyTo(frame(cv::Rect(0, 0, cam_panel.cols, H)));
  bev_panel.copyTo(frame(cv::Rect(cam_panel.cols + 2, 0, bev_panel.cols, H)));
  return frame;
}

}  // namespace

int main(int argc, char **argv) {
  gflags::SetUsageMessage("latentdrive-sample-clip --engine_file=E.engine --clip_dir=CLIP [--out_video=out.mp4]");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_engine_file.empty() || FLAGS_clip_dir.empty()) {
    gflags::ShowUsageWithFlags(argv[0]);
    return 1;
  }
  const fs::path clip_dir(FLAGS_clip_dir);
  std::ifstream  clip_file(clip_dir / "clip.json");
  if (!clip_file) LOGF("%s/clip.json not found (write one with export_clip.py)", clip_dir.c_str());
  const auto clip = nlohmann::json::parse(clip_file);

  // --- the planner: the engine says how many views and route points ---
  auto planner = model::planner::CreatePlanner(FLAGS_engine_file);
  if (!planner) return 1;
  const int num_views = planner->NumViews();
  if (static_cast<int>(clip["views"].size()) < num_views) {
    LOGF("The engine takes %d views, the clip has %zu", num_views, clip["views"].size());
  }

  // --- the preprocessing window ---
  preprocess::ViewWindowParams window_params;
  window_params.num_views          = num_views;
  window_params.view_tolerance_s   = FLAGS_view_tolerance_s;
  window_params.fill_missing_views = FLAGS_missing_view_fill == "zeros";
  auto window                      = preprocess::CreateViewWindow(window_params);

  // --- the status: route points + motion terms ---
  status::StatusBuilderParams status_params;
  status_params.num_route_points = planner->NumRoutePoints();
  auto builder                   = status::CreateStatusBuilder(status_params);
  if (!window || !builder) return 1;

  const auto odom = utils::LoadNpy((clip_dir / clip["odometry"].get<std::string>()).string());  // [M, 8]
  const auto route_np = utils::LoadNpy((clip_dir / clip["route"].get<std::string>()).string());  // [N, 2]
  std::vector<status::Point2d> route;
  for (std::size_t i = 0; i < route_np.shape[0]; ++i) route.push_back({route_np.data[2 * i], route_np.data[2 * i + 1]});
  builder->SetRoute(route);

  utils::NpyArray recorded;
  const bool        has_recorded = clip.contains("route_points") && !clip["route_points"].is_null();
  const std::string route_kind   = clip.value("route_kind", "driven");
  std::string       route_source = FLAGS_route_source;
  if (route_source.empty()) route_source = route_kind == "mission" || !has_recorded ? "file" : "recorded";
  if (route_source == "recorded") {
    if (!has_recorded) LOGF("--route_source=recorded, but the clip has no route_points.npy");
    recorded = utils::LoadNpy((clip_dir / clip["route_points"].get<std::string>()).string());
    if (static_cast<int>((recorded.shape[1] - 1) / 2) != planner->NumRoutePoints()) {
      LOGF("The clip recorded %zu route points per tick, the engine takes %d -- use --route_source=file",
           (recorded.shape[1] - 1) / 2, planner->NumRoutePoints());
    }
  }

  // --- the per-view frame lists ---
  std::vector<std::vector<ClipFrame>> frames(static_cast<std::size_t>(num_views));
  for (int v = 0; v < num_views; ++v) {
    for (const auto &f : clip["views"][v]["frames"]) frames[v].push_back({f[0].get<double>(), f[1].get<std::string>()});
  }
  const auto frame_path = [&](int v, const ClipFrame &f) {
    return (clip_dir / clip["views"][v]["dir"].get<std::string>() / f.file).string();
  };

  LOG("Clip %s: %.0f s, %zu front frames, %d view%s, status [1, %d] with route points from %s",
      clip_dir.c_str(), clip["duration_s"].get<double>(), frames[0].size(), num_views, num_views == 1 ? "" : "s",
      planner->StatusDim(),
      route_source == "recorded" ? "the vehicle (route_points.npy)" : ("route.npy (" + route_kind + " route)").c_str());

  auto                                         bev       = visualizer::CreateBevVisualizer();
  auto                                         projector = ProjectorFrom(clip["camera"]);
  cv::VideoWriter                              writer;
  std::vector<float>                           plans, stamps;
  std::vector<std::size_t>                     next(static_cast<std::size_t>(num_views), 0);
  std::size_t                                  next_odom = 0, planned = 0, skipped = 0, warmup = 0, blank = 0;
  std::optional<double>                        last_tick;
  std::vector<double>                          engine_ms;
  const std::size_t                            odom_cols = odom.shape[1];
  const double                                 period_s  = FLAGS_max_rate_hz > 0 ? 1.0 / FLAGS_max_rate_hz : 0.0;

  // --- the loop: one step per view-0 frame ---
  for (const ClipFrame &front : frames[0]) {
    const double t = front.t_s;

    // Odometry up to this instant; the newest sample is the ego pose
    while (next_odom < odom.shape[0] && odom.data[next_odom * odom_cols] <= t) {
      const float *o = odom.data.data() + next_odom * odom_cols;
      builder->PushPose({o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7]});
      ++next_odom;
    }
    // Side frames up to the tolerance past the front stamp: offline every
    // frame is at hand, so nothing has to wait for a late camera
    for (int v = 1; v < num_views; ++v) {
      while (next[v] < frames[v].size() && frames[v][next[v]].t_s <= t + FLAGS_view_tolerance_s) {
        window->Push(v, cv::imread(frame_path(v, frames[v][next[v]]), cv::IMREAD_COLOR), frames[v][next[v]].t_s);
        ++next[v];
      }
    }
    const cv::Mat front_bgr = cv::imread(frame_path(0, front), cv::IMREAD_COLOR);
    window->Push(0, front_bgr, t);

    // Camera-trigger pacing: at most max_rate_hz, on the stamps
    if (last_tick && t - *last_tick < period_s - 0.05 && t >= *last_tick) continue;
    last_tick = t;
    if (next_odom == 0 || t < 0.0) continue;  // the 2 s before the clip only fill the history

    // The ego: the odometry stamped nearest the frame, as the vehicle's
    // planner sees it (the newest sample when the frame arrives, a few ms on)
    std::size_t i_ego = next_odom - 1;
    if (next_odom < odom.shape[0] && odom.data[next_odom * odom_cols] - t < t - odom.data[i_ego * odom_cols]) {
      i_ego = next_odom;
    }
    const float             *o = odom.data.data() + i_ego * odom_cols;
    const status::PoseSample ego{o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7]};
    status::StatusResult     st;
    if (route_source == "recorded") {
      const auto points = RecordedAt(recorded, t);
      if (!points) {
        ++skipped;
        continue;
      }
      st = builder->BuildWithRoutePoints(t, ego, *points);
    } else {
      st = builder->Build(t, ego);
    }
    if (!st.ok) LOGF("status: %s", st.error.c_str());

    const preprocess::ViewSelection sel = window->Assemble(planner->VideoInput());
    if (!sel.ok) {
      if (sel.failed_view == 0) {
        ++warmup;
      } else {
        ++skipped;
        LOGW("t %.2f s skipped: %s", t, sel.reason.c_str());
      }
      continue;
    }
    blank += sel.filled.size();

    const auto               t0   = std::chrono::steady_clock::now();
    const common::Trajectory plan = planner->ForwardPrepared(st.ego.Encode());
    engine_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    if (plan.empty()) return 1;
    plans.insert(plans.end(), &plan[0].x, &plan[0].x + plan.size() * 3);
    stamps.push_back(static_cast<float>(t));
    ++planned;

    if (planned % 50 == 1) {
      LOG("t %6.2f s  v %4.1f m/s  |4s| %5.1f m  route point 50 m at (%.1f, %.1f)  engine %.1f ms", t,
          std::hypot(st.ego.vel_x, st.ego.vel_y), std::hypot(plan.back().x, plan.back().y), st.ego.Subgoal().x_m,
          st.ego.Subgoal().y_m, engine_ms.back());
    }
    if (!FLAGS_out_video.empty()) {
      const cv::Mat frame =
          Render(front_bgr, plan, st.ego, projector.get(), *bev, num_views, t, !sel.filled.empty());
      if (!writer.isOpened()) {
        writer.open(FLAGS_out_video, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 10.0, frame.size());
        if (!writer.isOpened()) LOGF("Cannot write %s", FLAGS_out_video.c_str());
      }
      writer.write(frame);
    }
  }

  // --- outputs ---
  const std::string stem = fs::path(FLAGS_engine_file).stem().string();
  const fs::path    out  = clip_dir / ("plans_" + stem + ".npy");
  if (planned > 0) {
    utils::SaveNpy(out.string(), plans.data(), {planned, static_cast<std::size_t>(planner->NumWaypoints()), 3});
    utils::SaveNpy((clip_dir / ("plans_" + stem + "_t.npy")).string(), stamps.data(), {planned});
  }
  std::sort(engine_ms.begin(), engine_ms.end());
  LOG("%zu plans -> %s (%zu warm-up steps, %zu skipped, %zu blank side-view slots)", planned, out.c_str(), warmup,
      skipped, blank);
  if (!engine_ms.empty()) {
    LOG("engine call: median %.1f ms, p95 %.1f ms", engine_ms[engine_ms.size() / 2],
        engine_ms[std::min(engine_ms.size() - 1, engine_ms.size() * 95 / 100)]);
  }
  if (writer.isOpened()) LOG("video -> %s", FLAGS_out_video.c_str());
  return 0;
}
