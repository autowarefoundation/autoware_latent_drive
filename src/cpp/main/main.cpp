#include <gflags/gflags.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
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
#include "engine_builder.hpp"
#include "frame_window.hpp"
#include "logger.hpp"
#include "npy.hpp"
#include "planner.hpp"
#include "route_map.hpp"
#include "signal_strip.hpp"
#include "timer.hpp"
#include "trajectory.hpp"
#include "trajectory_smoother.hpp"

using namespace std;

namespace fs      = std::filesystem;
namespace planner = model::planner;

// The model works in m/s; the panels report the number a driver would read
constexpr double kMps2Kmh = 3.6;

// Mode flags
DEFINE_string(mode, "",
              "Operating mode (options: build-engine, run-package, visualize)");

// Network flags
DEFINE_string(engine_file, "", "Path to a serialized TensorRT engine");
DEFINE_string(onnx_file, "", "Path to the ONNX export (build-engine mode)");
DEFINE_string(prec, "fp16", "Engine precision (options: fp32, fp16)");
DEFINE_bool(pure_fp16, false,
            "Disable the fp32 pin on the planner stage (NOT the validated "
            "recipe; one golden sample lands 0.4 mm over tolerance)");

// Input data flags
DEFINE_string(package_dir, "", "Path to the hand-off test package (workspace/)");
DEFINE_string(clip, "follow_lane", "Sequence clip name for visualize mode");
DEFINE_string(clip_dir, "",
              "visualize: use this clip directory directly (sequence.json / status.npy "
              "/ frames / ref / gt), overriding <package_dir>/sequence/<clip>");
DEFINE_string(gt_name, "gt_target.npy",
              "visualize: file name of the ground-truth trajectory in the clip dir");
DEFINE_string(calibration_file, "",
              "Path to calibration.json (default: <package_dir>/calibration.json)");

// Second camera source, consulted only when the engine takes two views
DEFINE_string(far_clip_dir, "",
              "Two-view engines: directory of the telephoto clip (frames/*.jpg + sequence.json "
              "with per-frame t_us), frame-aligned to the wide clip by nearest timestamp. "
              "Default: <clip dir>/far");
DEFINE_double(view_tolerance_s, planner::kDefaultViewToleranceS,
              "Two-view engines: largest |telephoto stamp - wide stamp| accepted per history "
              "slot; a slot without such a frame skips that step (no substitute frame)");
DEFINE_double(history_tolerance_s, planner::kDefaultHistoryToleranceS,
              "Two-view engines: slack on the 1.5 s the wide history must span before a "
              "step can run");

// Output flags
DEFINE_string(out_name, "engine_output.npy",
              "File name of the per-input outputs written next to the references");
DEFINE_string(ref_name, "ref_output_fp32.npy",
              "File name of the PyTorch reference outputs to compare against; "
              "references whose waypoint count differs from the engine are skipped");
DEFINE_string(out_video, "", "Output video path (default: viz_<clip>.mp4)");
DEFINE_int32(fps, 10, "Frames per second of the output video");
DEFINE_int32(png_every, 0, "Also dump a standalone PNG still every N steps (0 = off)");
DEFINE_bool(timeline, true,
            "visualize: draw the route + speed band under the camera/BEV row");
DEFINE_bool(plan_drift, false,
            "visualize: add the plan-drift strip to the band. A development "
            "readout on how much the plan disagrees with itself per step, not "
            "something a demo audience needs");
DEFINE_double(smooth_alpha, 0.35,
              "visualize: weight of the fresh plan when smoothing the DRAWN "
              "trajectory across steps (1.0 = draw the raw plan). Display "
              "only; the accuracy readout always uses the raw plan");
DEFINE_bool(skip_benchmark, false, "Skip the latency benchmark in run-package mode");

// @brief: This function is used to compute the largest per-waypoint xy
// deviation [m] between a plan and its reference row of the same length
static double MaxXyError(const common::Trajectory &in_traj, const float *in_ref) {
  double worst = 0;
  for (std::size_t i = 0; i < in_traj.size(); ++i) {
    const double dx = in_traj[i].x - in_ref[i * 3 + 0];
    const double dy = in_traj[i].y - in_ref[i * 3 + 1];
    worst           = std::max(worst, std::hypot(dx, dy));
  }
  return worst;
}

static common::Trajectory RowToTrajectory(const float *in_row, int in_num_waypoints) {
  common::Trajectory traj(in_num_waypoints);
  std::memcpy(traj.data(), in_row, in_num_waypoints * 3 * sizeof(float));
  return traj;
}

static std::vector<fs::path> SortedDirs(const fs::path &parent) {
  std::vector<fs::path> dirs;
  for (const auto &entry : fs::directory_iterator(parent)) {
    if (entry.is_directory()) dirs.push_back(entry.path());
  }
  std::sort(dirs.begin(), dirs.end());
  return dirs;
}

// One camera stream of a clip as sequence.json lists it: frame number, stamp
// and file name per frame, in time order
struct FrameIndex {
  std::vector<int>         frame;
  std::vector<double>      t_s;
  std::vector<std::string> file;
};

// @brief: This function is used to read the frames[] list of a clip's
// sequence.json. `t_us` is the bag stamp; `file` falls back to the zero-padded
// frame number, which is how the hand-off package names its JPEGs
static FrameIndex LoadFrameIndex(const fs::path &clip_dir) {
  std::ifstream json_file(clip_dir / "sequence.json");
  if (!json_file) LOGF("Cannot open %s", (clip_dir / "sequence.json").c_str());
  const auto sequence = nlohmann::json::parse(json_file);
  FrameIndex index;
  for (const auto &frame : sequence["frames"]) {
    index.frame.push_back(frame["frame"].get<int>());
    index.t_s.push_back(frame.value("t_us", 0L) * 1e-6);
    if (frame.contains("file")) {
      index.file.push_back(frame["file"].get<std::string>());
    } else {
      char name[16];
      std::snprintf(name, sizeof(name), "%05d.jpg", index.frame.back());
      index.file.push_back(name);
    }
  }
  return index;
}

// @brief: This function is used to feed the whole frame list of one clip into
// the sliding window step by step; `on_step` receives each step's plan input
// already assembled, `on_skip` the steps a two-view engine could not run.
// Shared by run-package (numeric check + benchmark) and visualize (rendering),
// so the streaming logic exists exactly once.
//
// A single-view engine streams the way the hand-off package defines it: every
// frame in order through the count-based FrameWindow, whose stride matches the
// package's history_frames. A two-view engine streams by timestamp instead
// (planner_io.pick_history / build_video_views): the wide clip anchors the
// four slots, the telephoto clip -- a second sensor's frames, from
// <clip>/far or --far_clip_dir -- supplies the frame nearest each chosen wide
// stamp within --view_tolerance_s, and a step without one is skipped rather
// than filled with a substitute frame
template <typename StepFn, typename SkipFn>
static void StreamClip(const fs::path &clip_dir, model::planner::Planner &planner,
                       preprocess::FrameWindow &window, timer::Timer &timer,
                       StepFn on_step, SkipFn on_skip) {
  std::ifstream json_file(clip_dir / "sequence.json");
  const auto    sequence   = nlohmann::json::parse(json_file);
  const auto    status_all = utils::LoadNpy((clip_dir / "status.npy").string());
  const auto   &steps      = sequence["steps"];

  const FrameIndex wide = LoadFrameIndex(clip_dir);

  // status.npy rows are the encoded status of each step, [n_steps, 2K + 4];
  // a clip prepared for another contract (K = 1 vs 6) cannot feed this engine
  const std::size_t status_dim = static_cast<std::size_t>(planner.StatusDim());
  if (status_all.shape.size() != 2 || status_all.shape[1] != status_dim) {
    LOGF("%s/status.npy has rows of %zu values, the engine takes status [1, %zu]", clip_dir.c_str(),
         status_all.shape.size() == 2 ? status_all.shape[1] : 0, status_dim);
  }

  // --- two-view: the telephoto clip and the timestamped window -----------
  const int                               num_views = planner.NumViews();
  FrameIndex                              far;
  fs::path                                far_dir;
  std::shared_ptr<preprocess::ViewWindow> views;
  if (num_views >= 2) {
    if (num_views > 2) {
      LOGF("Engine takes %d views; this tool has sources for two (wide + telephoto)", num_views);
    }
    far_dir = FLAGS_far_clip_dir.empty() ? clip_dir / "far" : fs::path(FLAGS_far_clip_dir);
    if (!fs::exists(far_dir / "sequence.json")) {
      LOGF("Two-view engine, but no telephoto clip at %s (frames/*.jpg + sequence.json with "
           "t_us). Point --far_clip_dir at it; the far view is never substituted",
           far_dir.c_str());
    }
    far = LoadFrameIndex(far_dir);
    preprocess::ViewWindowParams params;
    params.num_views           = num_views;
    params.view_tolerance_s    = FLAGS_view_tolerance_s;
    params.history_tolerance_s = FLAGS_history_tolerance_s;
    views                      = preprocess::CreateViewWindow(params);
    if (!views) LOGF("Cannot create the two-view window");
    LOG("  two-view stream: wide %zu frames, telephoto %zu frames from %s (tolerance %.0f ms)",
        wide.frame.size(), far.frame.size(), far_dir.c_str(), FLAGS_view_tolerance_s * 1e3);
  }

  std::size_t next_wide = 0, next_far = 0;
  cv::Mat     camera;

  window.Reset();
  for (std::size_t k = 0; k < steps.size(); ++k) {
    const int step_frame = steps[k]["frame"].get<int>();

    timer.StartCpu();
    while (next_wide < wide.frame.size() && wide.frame[next_wide] <= step_frame) {
      camera = cv::imread((clip_dir / "frames" / wide.file[next_wide]).string(), cv::IMREAD_COLOR);
      if (views) {
        views->Push(0, camera, wide.t_s[next_wide]);
      } else {
        window.Push(camera);
      }
      ++next_wide;
    }
    if (views) {
      // Offline every telephoto frame is at hand, so unlike the live node
      // nothing waits: any far frame up to the tolerance past the wide stamp
      // is a candidate for this step
      const double t_step = views->NewestStamp(0);
      while (next_far < far.frame.size() && far.t_s[next_far] <= t_step + FLAGS_view_tolerance_s) {
        views->Push(1, cv::imread((far_dir / "frames" / far.file[next_far]).string(), cv::IMREAD_COLOR),
                    far.t_s[next_far]);
        ++next_far;
      }
      const preprocess::ViewSelection sel = views->Assemble(planner.VideoInput());
      if (!sel.ok) {
        timer.StopCpu("preprocess");
        on_skip(k, sel.reason);
        continue;
      }
    } else {
      window.Assemble(planner.VideoInput());
    }
    timer.StopCpu("preprocess");

    const std::vector<float> status(status_all.data.begin() + k * status_dim,
                                    status_all.data.begin() + (k + 1) * status_dim);

    timer.StartCpu();
    const common::Trajectory pred = planner.ForwardPrepared(status);
    timer.StopCpu("enqueue");

    on_step(k, steps[k], status, pred, camera);
  }
}

// @brief: This function is used to warm the engine up so cuda runtime
// initialization does not pollute the first timed samples
static void Warmup(model::planner::Planner &planner) {
  const std::vector<float> zeros(planner.VideoNumel(), 0.f);
  const std::vector<float> status(static_cast<std::size_t>(planner.StatusDim()), 0.f);
  for (int i = 0; i < 10; ++i) {
    planner.Forward(zeros.data(), status);
  }
}

// @brief: This function is used to load the clip scene's camera calibration
// from the package's calibration.json. Returns nullptr (overlay disabled)
// when the file or the scene entry is missing
static std::shared_ptr<visualizer::CameraProjector> LoadProjector(
    const std::string &calibration_file, const std::string &scene) {
  std::ifstream file(calibration_file);
  if (!file) {
    LOGW("No %s -- camera overlay disabled", calibration_file.c_str());
    return nullptr;
  }
  const auto calib = nlohmann::json::parse(file);
  if (!calib["scenes"].contains(scene)) {
    LOGW("Scene '%s' not in %s -- camera overlay disabled", scene.c_str(),
         calibration_file.c_str());
    return nullptr;
  }

  const auto              &entry = calib["scenes"][scene];
  visualizer::CameraParams params;
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      params.k(row, col) = entry["K_crop_resized"][row][col].get<double>();
      params.r(row, col) = entry["extrinsics_camera_to_ego"]["rotation_matrix"][row][col].get<double>();
    }
    params.t[row] = entry["extrinsics_camera_to_ego"]["translation"][row].get<double>();
  }
  return visualizer::CreateCameraProjector(params);
}

// @brief: This function replays the hand-off package: the 9 alignment
// samples, a preprocessing self-test against the packaged tensors, and both
// streaming clips, writing the .npy outputs that the package's validate.py
// scores. The follow_lane clip doubles as the batch-1 latency benchmark
static int RunPackage() {
  auto planner_ptr = planner::CreatePlanner(FLAGS_engine_file);
  auto window_ptr  = preprocess::CreateFrameWindow();
  if (!planner_ptr || !window_ptr) return 1;
  Warmup(*planner_ptr);

  const fs::path package(FLAGS_package_dir);

  const auto num_waypoints = static_cast<std::size_t>(planner_ptr->NumWaypoints());

  LOG("== samples/ (packaged video.npy)");
  for (const auto &sample_dir : SortedDirs(package / "samples")) {
    const auto video  = utils::LoadNpy((sample_dir / "video.npy").string());
    const auto status = utils::LoadNpy((sample_dir / "status.npy").string());

    // A packaged video.npy is one complete input: [3,4,H,W] for a single-view
    // engine, [V,3,4,H,W] for a multi-view one. Feeding the wrong one would
    // read past the array
    if (video.Numel() != planner_ptr->VideoNumel()) {
      LOG("  %-32s (video.npy has %zu floats, the %d-view engine wants %zu -- skipped)",
          sample_dir.filename().c_str(), video.Numel(), planner_ptr->NumViews(),
          planner_ptr->VideoNumel());
      continue;
    }

    if (status.Numel() != static_cast<std::size_t>(planner_ptr->StatusDim())) {
      LOG("  %-32s (status.npy has %zu values, the engine takes %d -- skipped)",
          sample_dir.filename().c_str(), status.Numel(), planner_ptr->StatusDim());
      continue;
    }
    const common::Trajectory traj = planner_ptr->Forward(video.data.data(), status.data);
    utils::SaveNpy((sample_dir / FLAGS_out_name).string(), &traj[0].x,
                   {num_waypoints, 3});

    const fs::path ref_path = sample_dir / FLAGS_ref_name;
    if (!fs::exists(ref_path) ||
        utils::LoadNpy(ref_path.string()).shape[0] != num_waypoints) {
      LOG("  %-32s (no %s with %zu waypoints -- comparison skipped)",
          sample_dir.filename().c_str(), FLAGS_ref_name.c_str(), num_waypoints);
      continue;
    }
    const auto ref = utils::LoadNpy(ref_path.string());
    LOG("  %-32s xy max vs ref %.4f m", sample_dir.filename().c_str(),
        MaxXyError(traj, ref.data.data()));
  }

  LOG("== preprocessing self-test (frames/*.jpg vs packaged video.npy)");
  double worst_diff = 0;
  for (const auto &sample_dir : SortedDirs(package / "samples")) {
    const auto video = utils::LoadNpy((sample_dir / "video.npy").string());

    std::vector<fs::path> jpgs;
    for (const auto &entry : fs::directory_iterator(sample_dir / "frames")) {
      jpgs.push_back(entry.path());
    }
    std::sort(jpgs.begin(), jpgs.end());  // slot0..slot3

    std::vector<float>    frame(planner::kFrameNumel);
    double                diff   = 0;
    constexpr std::size_t kPlane = static_cast<std::size_t>(planner::kInputHeight) * planner::kInputWidth;
    for (int t = 0; t < planner::kNumHistFrames; ++t) {
      preprocess::PreprocessFrame(cv::imread(jpgs[t].string(), cv::IMREAD_COLOR), frame.data());
      for (int c = 0; c < 3; ++c) {
        const float *ours = frame.data() + c * kPlane;
        const float *ref  = video.data.data() + (c * planner::kNumHistFrames + t) * kPlane;
        for (std::size_t i = 0; i < kPlane; ++i) {
          diff = std::max(diff, static_cast<double>(std::abs(ours[i] - ref[i])));
        }
      }
    }
    worst_diff = std::max(worst_diff, diff);
    LOG("  %-32s max abs diff %.2e", sample_dir.filename().c_str(), diff);
  }
  LOG("  worst: %.2e (%s)", worst_diff,
      worst_diff < 1e-5 ? "same decoder" : "different JPEG decoder -- fine if ~1e-3");

  LOG("== sequence/ (streaming from frames/*.jpg)");
  for (const auto &clip_dir : SortedDirs(package / "sequence")) {
    const std::size_t row_numel = num_waypoints * 3;

    timer::Timer       timer;
    std::vector<float> plans;
    std::size_t        skipped = 0;
    // Keep the row so the output stays step-aligned with the reference; NaN
    // marks a step the two-view engine did not run
    const auto on_skip = [&](std::size_t k, const std::string &reason) {
      LOGW("  step %zu skipped: %s", k, reason.c_str());
      plans.insert(plans.end(), row_numel, std::numeric_limits<float>::quiet_NaN());
      ++skipped;
    };
    StreamClip(clip_dir, *planner_ptr, *window_ptr, timer,
               [&](std::size_t k, const nlohmann::json &,
                   const std::vector<float> &, const common::Trajectory &pred, const cv::Mat &) {
                 plans.insert(plans.end(), &pred[0].x, &pred[0].x + row_numel);
                 // The first steps carry the window pre-fill (16 decodes) and
                 // engine warmup transients; benchmark steps 5..n only
                 if (k == 4) timer.Reset();
               }, on_skip);
    if (skipped > 0) {
      LOGW("  %-16s %zu step(s) skipped (no telephoto frame within tolerance)",
           clip_dir.filename().c_str(), skipped);
    }

    const std::size_t n_steps = plans.size() / row_numel;
    utils::SaveNpy((clip_dir / FLAGS_out_name).string(), plans.data(),
                   {n_steps, num_waypoints, 3});

    const fs::path ref_path = clip_dir / FLAGS_ref_name;
    bool           compared = false;
    if (fs::exists(ref_path)) {
      const auto ref = utils::LoadNpy(ref_path.string());
      if (ref.shape.size() == 3 && ref.shape[1] == num_waypoints) {
        double worst = 0;
        for (std::size_t k = 0; k < n_steps; ++k) {
          if (std::isnan(plans[k * row_numel])) continue;  // skipped step
          worst = std::max(worst,
                           MaxXyError(RowToTrajectory(plans.data() + k * row_numel, num_waypoints),
                                      ref.data.data() + k * row_numel));
        }
        LOG("  %-16s %zu steps  xy max vs ref %.4f m", clip_dir.filename().c_str(), n_steps, worst);
        compared = true;
      }
    }
    if (!compared) {
      LOG("  %-16s %zu steps  (no %s with %zu waypoints -- comparison skipped)",
          clip_dir.filename().c_str(), n_steps, FLAGS_ref_name.c_str(), num_waypoints);
    }

    if (!FLAGS_skip_benchmark && clip_dir.filename() == "follow_lane") {
      LOG("== benchmark (%s, batch 1, warm cache excluded by design of the clip)",
          clip_dir.filename().c_str());
      timer.Show();
    }
  }
  return 0;
}

// @brief: This function is used to draw a translucent dark HUD panel with a
// list of (label, value, value-color) rows in the top-left of `inout_panel`.
// The label is muted gray, the value carries the color -- so the deployment
// numbers (latency, accuracy) read at a glance
static void DrawHud(cv::Mat                                       &inout_panel,
                    const std::vector<std::array<std::string, 2>> &rows,
                    const std::vector<cv::Scalar>                 &value_colors) {
  constexpr int    kFont     = cv::FONT_HERSHEY_DUPLEX;
  constexpr double kScale    = 0.5;
  constexpr int    kLineH    = 26;
  constexpr int    kPadX     = 14;
  constexpr int    kPadY     = 20;
  constexpr int    kLabelW   = 96;
  const cv::Scalar kLabelInk = {150, 150, 150};

  int max_w = 0;
  for (const auto &row : rows) {
    const cv::Size sz = cv::getTextSize(row[1], kFont, kScale, 1, nullptr);
    max_w             = std::max(max_w, kLabelW + sz.width);
  }
  const int panel_w = kLabelW + max_w + kPadX;
  const int panel_h = kPadY + static_cast<int>(rows.size()) * kLineH;

  cv::Mat roi = inout_panel(cv::Rect(0, 0, std::min(panel_w + kPadX, inout_panel.cols),
                                     std::min(panel_h, inout_panel.rows)));
  cv::Mat shade(roi.size(), roi.type(), cv::Scalar(18, 18, 18));
  cv::addWeighted(shade, 0.62, roi, 0.38, 0.0, roi);

  for (std::size_t i = 0; i < rows.size(); ++i) {
    const int y = kPadY + static_cast<int>(i) * kLineH + 4;
    cv::putText(inout_panel, rows[i][0], {kPadX, y}, kFont, kScale, kLabelInk, 1, cv::LINE_AA);
    cv::putText(inout_panel, rows[i][1], {kPadX + kLabelW, y}, kFont, kScale,
                value_colors[i], 1, cv::LINE_AA);
  }
}

// @brief: This function is used to draw the route ledger: how far along the
// clip's route the ego is and what is still owed to each milestone and to the
// destination. `in_subgoal_m` is the straight-line distance to the subgoal the
// model is actually being fed this step
static void DrawRouteLedger(cv::Mat &inout_panel, const visualizer::RouteMap &in_route,
                            std::size_t in_step, double in_subgoal_m) {
  constexpr int    kFont  = cv::FONT_HERSHEY_SIMPLEX;
  const cv::Scalar kInk   = {225, 225, 225};
  const cv::Scalar kMuted = {120, 128, 130};
  const cv::Scalar kGoal  = {120, 210, 90};
  const cv::Scalar kSub   = {52, 104, 235};

  inout_panel.setTo(cv::Scalar(25, 26, 26));
  int y = 20;

  const double done = in_route.ArcAt(in_step);
  cv::putText(inout_panel, "ROUTE", {8, y}, kFont, 0.42, kInk, 1, cv::LINE_AA);
  // A percentage only means something against a route that has an end
  if (in_route.HasGoal() && in_route.RouteLength() > 0.0) {
    cv::putText(inout_panel, cv::format("%.0f%%", 100.0 * done / in_route.RouteLength()),
                {96, y}, kFont, 0.42, kInk, 1, cv::LINE_AA);
  }
  y += 8;
  cv::line(inout_panel, {8, y}, {inout_panel.cols - 8, y}, cv::Scalar(45, 47, 47), 1);
  y += 20;

  // What the model is steering at right now, before any of the derived numbers
  cv::putText(inout_panel, "subgoal", {8, y}, kFont, 0.38, kSub, 1, cv::LINE_AA);
  cv::putText(inout_panel, cv::format("%.0f m", in_subgoal_m), {96, y}, kFont, 0.38, kSub, 1,
              cv::LINE_AA);
  y += 20;

  if (!in_route.HasGoal()) {
    cv::putText(inout_panel, "goal beyond", {8, y}, kFont, 0.38, kMuted, 1, cv::LINE_AA);
    cv::putText(inout_panel, "this clip", {8, y + 16}, kFont, 0.38, kMuted, 1, cv::LINE_AA);
    return;
  }

  // The next few milestones, then the destination -- a passed milestone drops
  // off the list rather than counting down into negative metres
  int shown = 0;
  for (const visualizer::RouteMilestone &m : in_route.Milestones()) {
    const double left = m.s_m - done;
    if (m.is_goal) {
      cv::putText(inout_panel, "GOAL", {8, y}, kFont, 0.40, kGoal, 1, cv::LINE_AA);
      cv::putText(inout_panel, cv::format("%.0f m", std::max(left, 0.0)), {96, y}, kFont, 0.40,
                  kGoal, 1, cv::LINE_AA);
      break;
    }
    if (left < 0.0 || shown >= 3) continue;
    cv::putText(inout_panel, cv::format("M%d", m.label), {8, y}, kFont, 0.38, kInk, 1, cv::LINE_AA);
    cv::putText(inout_panel, cv::format("%.0f m", left), {96, y}, kFont, 0.38, kInk, 1, cv::LINE_AA);
    y += 18;
    ++shown;
  }
}

// @brief: This function streams one clip and renders each step as a camera
// panel (rainbow prediction ribbon + ground-truth line + deployment HUD) beside
// the BEV, written as an mp4 (PNG directory when no video encoder is built in)
static int Visualize() {
  auto planner_ptr = planner::CreatePlanner(FLAGS_engine_file);
  auto window_ptr  = preprocess::CreateFrameWindow();
  auto bev_ptr     = visualizer::CreateBevVisualizer();
  if (!planner_ptr || !window_ptr || !bev_ptr) return 1;

  // Display-only: the plan the panels draw is the filtered one, everything
  // reported in the HUD stays on the raw engine output
  auto smoother_ptr = visualizer::CreateTrajectorySmoother({FLAGS_smooth_alpha});

  // --clip_dir points straight at a clip directory; otherwise fall back to the
  // package layout <package_dir>/sequence/<clip>
  const fs::path clip_dir = FLAGS_clip_dir.empty()
                                ? (fs::path(FLAGS_package_dir) / "sequence" / FLAGS_clip)
                                : fs::path(FLAGS_clip_dir);
  const auto     ref      = fs::exists(clip_dir / FLAGS_ref_name)
                                ? utils::LoadNpy((clip_dir / FLAGS_ref_name).string())
                                : utils::NpyArray{};
  const auto     gt       = fs::exists(clip_dir / FLAGS_gt_name)
                                ? utils::LoadNpy((clip_dir / FLAGS_gt_name).string())
                                : utils::NpyArray{};

  // A reference/gt trajectory only overlays when its waypoint count matches
  // the engine's; the 8-waypoint package files stay useful for the 2 Hz
  // checkpoints and are skipped for the 10 Hz ones
  const auto num_waypoints = static_cast<std::size_t>(planner_ptr->NumWaypoints());
  const bool ref_usable    = ref.shape.size() == 3 && ref.shape[1] == num_waypoints;
  const bool gt_usable     = gt.shape.size() == 3 && gt.shape[1] == num_waypoints;
  if (!ref_usable) {
    LOGW("%s has %zu waypoints, engine %zu -- reference overlay disabled",
         FLAGS_ref_name.c_str(), ref.shape.size() == 3 ? ref.shape[1] : 0, num_waypoints);
  }

  std::ifstream json_file(clip_dir / "sequence.json");
  const auto    sequence = nlohmann::json::parse(json_file);

  const std::string calibration_file =
      FLAGS_calibration_file.empty()
          ? (fs::path(FLAGS_package_dir) / "calibration.json").string()
          : FLAGS_calibration_file;
  const auto projector_ptr = LoadProjector(calibration_file, sequence["scene"].get<std::string>());

  const std::string out_video =
      FLAGS_out_video.empty() ? "viz_" + FLAGS_clip + ".mp4" : FLAGS_out_video;

  // --- clip-wide context for the timeline band ---------------------------
  // The route, the speed profile and the value axes are all knowable before a
  // single frame is rendered, so the band's geometry is fixed up front and
  // only the playhead moves
  const auto        status_all = utils::LoadNpy((clip_dir / "status.npy").string());
  const int         status_dim = planner_ptr->StatusDim();
  const std::size_t n_steps    = std::min<std::size_t>(sequence["steps"].size(),
                                                       status_all.data.size() / static_cast<std::size_t>(status_dim));

  std::map<int, visualizer::MapPose> frame_pose;
  for (const auto &f : sequence["frames"]) {
    frame_pose[f["frame"].get<int>()] = {f["ego_xy"][0].get<double>(),
                                         f["ego_xy"][1].get<double>(),
                                         f["ego_yaw"].get<double>()};
  }

  std::vector<visualizer::MapPose> poses;
  std::vector<cv::Point2d>         subgoals_ego;
  double                           speed_max = 1.0;
  for (std::size_t k = 0; k < n_steps; ++k) {
    const common::EgoStatus st = common::EgoStatus::Decode(status_all.data.data() + k * status_dim, status_dim);
    poses.push_back(frame_pose[sequence["steps"][k]["frame"].get<int>()]);
    subgoals_ego.emplace_back(st.Subgoal().x_m, st.Subgoal().y_m);
    speed_max = std::max(speed_max, static_cast<double>(std::hypot(st.vel_x, st.vel_y)));
  }

  auto route_ptr = visualizer::CreateRouteMap(poses, subgoals_ego);

  const double speed_axis      = std::ceil(speed_max * kMps2Kmh / 10.0) * 10.0;
  auto         speed_strip_ptr = visualizer::CreateSignalStrip(
      {"speed", "km/h", 0.0, speed_axis, n_steps},
      {{"speed", cv::Scalar(120, 210, 90), false}});

  // How much the +4 s end of the plan shifts from one step to the next. The
  // reach is ego-relative, so a steady drive holds it flat and everything left
  // on this strip is the plan disagreeing with itself -- which is exactly what
  // the smoother is there to take out, so both traces are shown. Off by
  // default: it answers a question about the model, not about the drive
  std::shared_ptr<visualizer::SignalStrip> plan_strip_ptr;
  if (FLAGS_plan_drift) {
    plan_strip_ptr = visualizer::CreateSignalStrip(
        {"plan drift", "m / step", 0.0, 2.0, n_steps},
        {{"raw", cv::Scalar(120, 120, 120), true},
         {"drawn", visualizer::HorizonColor(1.0), false}});
  }
  double prev_reach_raw = -1.0, prev_reach_drawn = -1.0;

  cv::VideoWriter writer;
  bool            use_video = true;
  fs::path        png_dir;
  timer::Timer    timer;
  double          prev_t_rel_s  = 0.0;
  const double    step_period_s = 1.0 / std::max(sequence.value("fps", FLAGS_fps), 1);

  constexpr int    kPanelHeight = 560;
  constexpr int    kBandHeight  = 200;
  const cv::Scalar kGood{120, 210, 90};  // green: within tolerance / fast
  const cv::Scalar kInk{225, 225, 225};  // near-white value text
  const cv::Scalar kSeparator{45, 47, 47};

  // No plan, no frame: a skipped step leaves a gap in the video rather than
  // drawing a stale or invented plan
  const auto on_skip = [&](std::size_t k, const std::string &reason) {
    LOGW("step %zu skipped: %s", k, reason.c_str());
  };

  StreamClip(clip_dir, *planner_ptr, *window_ptr, timer,
             [&](std::size_t k, const nlohmann::json &step,
                 const std::vector<float> &status_vec, const common::Trajectory &pred,
                 const cv::Mat &camera_in) {
               const common::EgoStatus status =
                   common::EgoStatus::Decode(status_vec.data(), static_cast<int>(status_vec.size()));
               const std::size_t row_numel = num_waypoints * 3;

               // Step period straight from the clip; the smoother needs it to
               // know how far to carry the previous plan forward. Older clips
               // without per-step stamps fall back to the sequence rate
               const double t_rel_s = step.value("t_rel_s", 0.0);
               const double dt_s =
                   (k > 0 && t_rel_s > prev_t_rel_s) ? (t_rel_s - prev_t_rel_s) : step_period_s;
               prev_t_rel_s                  = t_rel_s;
               const common::Trajectory shown = smoother_ptr->Update(pred, dt_s);

               // --- camera panel: rainbow prediction ribbon + white GT line ---
               cv::Mat camera = camera_in.clone();
               if (projector_ptr) {
                 projector_ptr->DrawRibbon(camera, shown, 0.9, 0.5);
                 if (gt_usable) {
                   const auto gt_k = RowToTrajectory(gt.data.data() + k * row_numel, num_waypoints);
                   projector_ptr->DrawTrajectory(camera, gt_k, cv::Scalar(255, 255, 255), 1, true);
                 }
               }
               cv::Mat cam_panel;
               cv::resize(camera, cam_panel, cv::Size(kPanelHeight * camera.cols / camera.rows, kPanelHeight),
                          0, 0, cv::INTER_LINEAR);

               // --- BEV panel: gradient prediction + GT + subgoal + velocity ---
               bev_ptr->Clear();
               if (gt_usable) {
                 bev_ptr->DrawTrajectory(RowToTrajectory(gt.data.data() + k * row_numel, num_waypoints),
                                         visualizer::color::kGroundTruth, 1);
               }
               bev_ptr->DrawTrajectoryHorizon(shown, 2);
               for (const common::RoutePoint &p : status.route_points) bev_ptr->DrawSubgoal(p.x_m, p.y_m);
               bev_ptr->DrawVelocity(status.vel_x, status.vel_y);
               // the plan itself is drawn as a green(now)->red(+4s) gradient;
               // the legend swatch takes the near/now end
               bev_ptr->AddLegend("plan now->+4s", visualizer::HorizonColor(0.0));
               bev_ptr->AddLegend("ground truth", visualizer::color::kGroundTruth);
               bev_ptr->AddLegend(status.route_points.size() == 1 ? "subgoal 50m" : "route points",
                                  visualizer::color::kSubgoal);
               const cv::Mat &bev = bev_ptr->Image();
               cv::Mat        bev_panel;
               cv::resize(bev, bev_panel,
                          cv::Size(kPanelHeight * bev.cols / bev.rows, kPanelHeight),
                          0, 0, cv::INTER_AREA);

               // --- deployment HUD on the camera panel ---
               const double engine_ms = timer.Last("enqueue");
               const double pre_ms    = timer.Last("preprocess");
               const double plan_ms   = engine_ms + pre_ms;
               const double med_ms    = timer.Percentile("enqueue", 50) +
                                     timer.Percentile("preprocess", 50);
               const float speed = std::hypot(status.vel_x, status.vel_y);
               const float wp_4s = std::hypot(pred.back().x, pred.back().y);

               std::vector<std::array<std::string, 2>> rows;
               std::vector<cv::Scalar>                 cols;
               // `phase` is optional -- the newer whole-scene exports drop it
               rows.push_back({"LatentDrive",
                               cv::format("TRT fp16 %dv  %s  %zu/%d  %s", planner_ptr->NumViews(),
                                          FLAGS_clip.c_str(), k,
                                          sequence["n_steps"].get<int>() - 1,
                                          step.value("phase", std::string()).c_str())});
               cols.push_back(kInk);
               rows.push_back({"speed",
                               cv::format("%.1f ms/plan  %.0f Hz  (engine %.1f + pre %.1f)",
                                          plan_ms, 1000.0 / std::max(med_ms, 1e-3), engine_ms, pre_ms)});
               cols.push_back(kGood);
               if (ref_usable) {
                 const double err = MaxXyError(pred, ref.data.data() + k * row_numel);
                 rows.push_back({"accuracy", cv::format("%.3f m max vs PyTorch fp32", err)});
                 cols.push_back(err <= 0.10 ? kGood : cv::Scalar(60, 120, 235));
               } else {
                 rows.push_back({"accuracy", "no fp32 reference for this clip"});
                 cols.push_back(kInk);
               }
               std::string intent = cv::format("|4s| %.1f m", wp_4s);
               if (gt_usable) {
                 const float *gt_last = gt.data.data() + k * row_numel + (num_waypoints - 1) * 3;
                 intent += cv::format(" (GT %.1f)", std::hypot(gt_last[0], gt_last[1]));
               }
               intent += cv::format("  v %.1f km/h", speed * kMps2Kmh);
               rows.push_back({"intent", intent});
               cols.push_back(kInk);
               DrawHud(cam_panel, rows, cols);

               // --- timeline band: route ledger + stability strips ---
               speed_strip_ptr->Push({speed * kMps2Kmh});
               if (plan_strip_ptr) {
                 const double reach_raw   = std::hypot(pred.back().x, pred.back().y);
                 const double reach_drawn = std::hypot(shown.back().x, shown.back().y);
                 plan_strip_ptr->Push(
                     {prev_reach_raw < 0 ? 0.0 : std::abs(reach_raw - prev_reach_raw),
                      prev_reach_drawn < 0 ? 0.0 : std::abs(reach_drawn - prev_reach_drawn)});
                 prev_reach_raw   = reach_raw;
                 prev_reach_drawn = reach_drawn;
               }

               // --- compose: camera | separator | BEV, over the timeline band ---
               const int total_w = cam_panel.cols + 2 + bev_panel.cols;
               const int band_h  = FLAGS_timeline ? kBandHeight : 0;
               cv::Mat   frame(kPanelHeight + (band_h > 0 ? band_h + 2 : 0), total_w, CV_8UC3,
                               kSeparator);
               cam_panel.copyTo(frame(cv::Rect(0, 0, cam_panel.cols, kPanelHeight)));
               bev_panel.copyTo(
                   frame(cv::Rect(cam_panel.cols + 2, 0, bev_panel.cols, kPanelHeight)));

               if (band_h > 0) {
                 constexpr int kMapW    = 300;
                 constexpr int kLedgerW = 150;
                 const int     band_y   = kPanelHeight + 2;
                 const int     strips_x = kMapW + kLedgerW + 4;
                 const int     strips_w = std::max(total_w - strips_x, 1);
                 // Speed alone gets the whole height; the drift strip, when
                 // asked for, splits it
                 const int strip_h = plan_strip_ptr ? (band_h - 2) / 2 : band_h;

                 route_ptr->Draw({kMapW, band_h}, k)
                     .copyTo(frame(cv::Rect(0, band_y, kMapW, band_h)));
                 cv::Mat ledger = frame(cv::Rect(kMapW + 2, band_y, kLedgerW, band_h));
                 DrawRouteLedger(ledger, *route_ptr, k, std::hypot(status.Subgoal().x_m, status.Subgoal().y_m));
                 speed_strip_ptr->Draw({strips_w, strip_h})
                     .copyTo(frame(cv::Rect(strips_x, band_y, strips_w, strip_h)));
                 if (plan_strip_ptr) {
                   plan_strip_ptr->Draw({strips_w, band_h - strip_h - 2})
                       .copyTo(frame(cv::Rect(strips_x, band_y + strip_h + 2, strips_w,
                                              band_h - strip_h - 2)));
                 }
               }

               if (k == 0) {
                 // VideoWriter does not create the parent directory; without
                 // this an --out_video into a missing dir fails to open and is
                 // misreported as "no video encoder"
                 const fs::path out_parent = fs::path(out_video).parent_path();
                 if (!out_parent.empty()) fs::create_directories(out_parent);
                 writer.open(out_video, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                             FLAGS_fps, frame.size());
                 if (!writer.isOpened()) {
                   use_video = false;
                   png_dir   = fs::path(out_video).replace_extension("");
                   fs::create_directories(png_dir);
                   LOGW("VideoWriter could not open %s (no encoder?); writing PNGs to %s",
                        out_video.c_str(), png_dir.c_str());
                 }
               }
               if (use_video) {
                 writer << frame;
               } else {
                 cv::imwrite((png_dir / cv::format("step_%04zu.png", k)).string(), frame);
               }
               if (FLAGS_png_every > 0 && k % FLAGS_png_every == 0) {
                 cv::imwrite(fs::path(out_video).replace_extension("").string() +
                                 cv::format("_step%04zu.png", k),
                             frame);
               }
             }, on_skip);

  if (use_video) writer.release();
  LOG("Wrote %s", use_video ? out_video.c_str() : png_dir.c_str());
  return 0;
}

// @brief: This function builds the engine from the ONNX export through the
// TensorRT C++ API -- the in-framework equivalent of the trtexec recipe
static int BuildEngine() {
  if (FLAGS_onnx_file.empty()) {
    LOGE("--mode=build-engine requires --onnx_file");
    return 2;
  }
  planner::BuildParams params;
  params.onnx_path        = FLAGS_onnx_file;
  params.engine_path      = FLAGS_engine_file;
  params.prec             = planner::GetPrecision(FLAGS_prec);
  params.pin_planner_fp32 = !FLAGS_pure_fp16;
  return planner::BuildEngine(params) ? 0 : 1;
}

int main(int argc, char *argv[]) {
  gflags::SetUsageMessage(
      "LatentDrive e2e planner TensorRT tools\n"
      "  --mode=build-engine --onnx_file=O   --engine_file=E       ONNX -> engine\n"
      "  --mode=run-package  --engine_file=E --package_dir=DIR     validate + benchmark\n"
      "  --mode=visualize    --engine_file=E --package_dir=DIR     camera overlay + BEV video\n"
      "A two-view engine (6-D video input) also reads the telephoto clip from <clip>/far\n"
      "or --far_clip_dir; --view_tolerance_s bounds the wide/telephoto stamp mismatch");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_engine_file.empty()) {
    LOGE("--engine_file is required (see --help)");
    return 2;
  }

  if (FLAGS_mode == "build-engine") {
    return BuildEngine();
  }

  if (FLAGS_mode == "run-package") {
    if (FLAGS_package_dir.empty()) {
      LOGE("--package_dir is required for --mode=run-package");
      return 2;
    }
    return RunPackage();
  } else if (FLAGS_mode == "visualize") {
    if (FLAGS_package_dir.empty() && FLAGS_clip_dir.empty()) {
      LOGE("--mode=visualize needs --package_dir or --clip_dir");
      return 2;
    }
    return Visualize();
  }

  LOGE("Unknown --mode '%s' (options: build-engine, run-package, visualize)",
       FLAGS_mode.c_str());
  return 2;
}
