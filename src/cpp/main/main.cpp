#include <gflags/gflags.h>

#include <algorithm>
#include <cmath>
#include <cstring>
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
#include "engine_builder.hpp"
#include "frame_window.hpp"
#include "logger.hpp"
#include "npy.hpp"
#include "planner.hpp"
#include "timer.hpp"
#include "trajectory.hpp"

using namespace std;

namespace fs      = std::filesystem;
namespace planner = model::planner;

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
DEFINE_string(calibration_file, "",
              "Path to calibration.json (default: <package_dir>/calibration.json)");

// Output flags
DEFINE_string(out_name, "engine_output.npy",
              "File name of the per-input outputs written next to the references");
DEFINE_string(out_video, "", "Output video path (default: viz_<clip>.mp4)");
DEFINE_int32(fps, 10, "Frames per second of the output video");
DEFINE_int32(png_every, 0, "Also dump a standalone PNG still every N steps (0 = off)");
DEFINE_bool(skip_benchmark, false, "Skip the latency benchmark in run-package mode");

// @brief: This function is used to compute the largest per-waypoint xy
// deviation [m] between a plan and its reference row
static double MaxXyError(const common::Trajectory &in_traj, const float *in_ref) {
  double worst = 0;
  for (int i = 0; i < common::kNumWaypoints; ++i) {
    const double dx = in_traj[i].x - in_ref[i * 3 + 0];
    const double dy = in_traj[i].y - in_ref[i * 3 + 1];
    worst           = std::max(worst, std::hypot(dx, dy));
  }
  return worst;
}

static common::Trajectory RowToTrajectory(const float *in_row /* [8, 3] */) {
  common::Trajectory traj;
  std::memcpy(traj.data(), in_row, sizeof(common::Trajectory));
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

// @brief: This function is used to feed the whole frame list of one clip into
// the sliding window step by step; the callback receives each step's plan
// input already assembled. Shared by run-package (numeric check + benchmark)
// and visualize (rendering), so the streaming logic exists exactly once
template <typename StepFn>
static void StreamClip(const fs::path &clip_dir, model::planner::Planner &planner,
                       preprocess::FrameWindow &window, timer::Timer &timer,
                       StepFn on_step) {
  std::ifstream json_file(clip_dir / "sequence.json");
  const auto    sequence   = nlohmann::json::parse(json_file);
  const auto    status_all = utils::LoadNpy((clip_dir / "status.npy").string());

  std::vector<int> frame_numbers;
  for (const auto &frame : sequence["frames"]) {
    frame_numbers.push_back(frame["frame"].get<int>());
  }

  const auto &steps      = sequence["steps"];
  std::size_t next_frame = 0;
  cv::Mat     camera;

  window.Reset();
  for (std::size_t k = 0; k < steps.size(); ++k) {
    const int step_frame = steps[k]["frame"].get<int>();

    timer.StartCpu();
    while (next_frame < frame_numbers.size() && frame_numbers[next_frame] <= step_frame) {
      char name[16];
      std::snprintf(name, sizeof(name), "%05d.jpg", frame_numbers[next_frame]);
      camera = cv::imread((clip_dir / "frames" / name).string(), cv::IMREAD_COLOR);
      window.Push(camera);
      ++next_frame;
    }
    window.Assemble(planner.VideoInput());
    timer.StopCpu("preprocess");

    std::array<float, planner::kStatusDim> status;
    std::copy_n(status_all.data.begin() + k * planner::kStatusDim,
                planner::kStatusDim, status.begin());

    timer.StartCpu();
    const common::Trajectory pred = planner.ForwardPrepared(status);
    timer.StopCpu("enqueue");

    on_step(k, steps[k], status, pred, camera);
  }
}

// @brief: This function is used to warm the engine up so cuda runtime
// initialization does not pollute the first timed samples
static void Warmup(model::planner::Planner &planner) {
  const std::vector<float>                     zeros(planner::kVideoNumel, 0.f);
  const std::array<float, planner::kStatusDim> status{};
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

  LOG("== samples/ (packaged video.npy)");
  for (const auto &sample_dir : SortedDirs(package / "samples")) {
    const auto video  = utils::LoadNpy((sample_dir / "video.npy").string());
    const auto status = utils::LoadNpy((sample_dir / "status.npy").string());
    const auto ref    = utils::LoadNpy((sample_dir / "ref_output_fp32.npy").string());

    std::array<float, planner::kStatusDim> status_arr;
    std::copy_n(status.data.begin(), planner::kStatusDim, status_arr.begin());

    const common::Trajectory traj = planner_ptr->Forward(video.data.data(), status_arr);
    utils::SaveNpy((sample_dir / FLAGS_out_name).string(), &traj[0].x,
                   {common::kNumWaypoints, 3});
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
    const auto ref = utils::LoadNpy((clip_dir / "ref_output_fp32.npy").string());

    timer::Timer       timer;
    std::vector<float> plans;
    StreamClip(clip_dir, *planner_ptr, *window_ptr, timer,
               [&](std::size_t k, const nlohmann::json &,
                   const std::array<float, planner::kStatusDim> &,
                   const common::Trajectory &pred, const cv::Mat &) {
                 plans.insert(plans.end(), &pred[0].x, &pred[0].x + planner::kOutputNumel);
                 // The first steps carry the window pre-fill (16 decodes) and
                 // engine warmup transients; benchmark steps 5..n only
                 if (k == 4) timer.Reset();
               });

    const std::size_t n_steps = plans.size() / planner::kOutputNumel;
    utils::SaveNpy((clip_dir / FLAGS_out_name).string(), plans.data(),
                   {n_steps, common::kNumWaypoints, 3});

    double worst = 0;
    for (std::size_t k = 0; k < n_steps; ++k) {
      worst = std::max(worst, MaxXyError(RowToTrajectory(plans.data() + k * planner::kOutputNumel),
                                         ref.data.data() + k * planner::kOutputNumel));
    }
    LOG("  %-16s %zu steps  xy max vs ref %.4f m", clip_dir.filename().c_str(), n_steps, worst);

    if (!FLAGS_skip_benchmark && clip_dir.filename() == "follow_lane") {
      LOG("== benchmark (%s, batch 1, warm cache excluded by design of the clip)",
          clip_dir.filename().c_str());
      timer.Show();
    }
  }
  return 0;
}

// @brief: This function streams one clip and renders each step as camera
// frame (with the waypoint overlay when calibration covers the scene) + HUD
// + BEV, written as an mp4 (PNG directory when no video encoder is built in)
static int Visualize() {
  auto planner_ptr = planner::CreatePlanner(FLAGS_engine_file);
  auto window_ptr  = preprocess::CreateFrameWindow();
  auto bev_ptr     = visualizer::CreateBevVisualizer();
  if (!planner_ptr || !window_ptr || !bev_ptr) return 1;

  const fs::path clip_dir = fs::path(FLAGS_package_dir) / "sequence" / FLAGS_clip;
  const auto     ref      = utils::LoadNpy((clip_dir / "ref_output_fp32.npy").string());
  const auto     gt       = utils::LoadNpy((clip_dir / "gt_target.npy").string());

  std::ifstream json_file(clip_dir / "sequence.json");
  const auto    sequence = nlohmann::json::parse(json_file);

  const std::string calibration_file =
      FLAGS_calibration_file.empty()
          ? (fs::path(FLAGS_package_dir) / "calibration.json").string()
          : FLAGS_calibration_file;
  const auto projector_ptr = LoadProjector(calibration_file, sequence["scene"].get<std::string>());

  const std::string out_video =
      FLAGS_out_video.empty() ? "viz_" + FLAGS_clip + ".mp4" : FLAGS_out_video;

  cv::VideoWriter writer;
  bool            use_video = true;
  fs::path        png_dir;
  timer::Timer    timer;

  StreamClip(clip_dir, *planner_ptr, *window_ptr, timer,
             [&](std::size_t k, const nlohmann::json &step,
                 const std::array<float, planner::kStatusDim> &status,
                 const common::Trajectory &pred, const cv::Mat &camera_in) {
               const common::Trajectory gt_k  = RowToTrajectory(gt.data.data() + k * planner::kOutputNumel);
               const common::Trajectory ref_k = RowToTrajectory(ref.data.data() + k * planner::kOutputNumel);

               cv::Mat camera = camera_in.clone();
               if (projector_ptr) {
                 projector_ptr->DrawTrajectory(camera, gt_k, visualizer::color::kGroundTruth, 1, false);
                 projector_ptr->DrawTrajectory(camera, ref_k, visualizer::color::kReference, 1);
                 projector_ptr->DrawTrajectory(camera, pred, visualizer::color::kPrediction, 2);
               }

               bev_ptr->Clear();
               bev_ptr->DrawTrajectory(gt_k, visualizer::color::kGroundTruth, 1);
               bev_ptr->DrawTrajectory(ref_k, visualizer::color::kReference, 2, true);
               bev_ptr->DrawTrajectory(pred, visualizer::color::kPrediction, 2);
               bev_ptr->DrawSubgoal(status[0] * 10.f, status[1] * 10.f);
               bev_ptr->DrawVelocity(status[2], status[3]);
               bev_ptr->AddLegend("TRT fp16 pred", visualizer::color::kPrediction);
               bev_ptr->AddLegend("PyTorch ref", visualizer::color::kReference);
               bev_ptr->AddLegend("ground truth", visualizer::color::kGroundTruth);
               bev_ptr->AddLegend("subgoal 50m", visualizer::color::kSubgoal);

               // Compose one column: camera resized to the BEV width, a HUD
               // strip, then the BEV canvas
               const cv::Mat &bev = bev_ptr->Image();
               cv::Mat        camera_resized;
               cv::resize(camera, camera_resized,
                          cv::Size(bev.cols, camera.rows * bev.cols / camera.cols),
                          0, 0, cv::INTER_AREA);

               const float speed = std::hypot(status[2], status[3]);
               const float wp8   = std::hypot(pred[7].x, pred[7].y);
               const auto  hud   = cv::format(
                   "%3zu/%d %s  v %4.1f  wp8 %5.2fm  err %.3f", k,
                   sequence["n_steps"].get<int>() - 1,
                   step["phase"].get<std::string>().c_str(), speed, wp8,
                   MaxXyError(pred, ref.data.data() + k * planner::kOutputNumel));

               const int kHudHeight = 26;
               cv::Mat   frame(camera_resized.rows + kHudHeight + bev.rows, bev.cols,
                               CV_8UC3, cv::Scalar(25, 26, 26));
               camera_resized.copyTo(frame(cv::Rect(0, 0, bev.cols, camera_resized.rows)));
               cv::putText(frame, hud, {8, camera_resized.rows + 18},
                           cv::FONT_HERSHEY_SIMPLEX, 0.44, cv::Scalar(183, 194, 195), 1, cv::LINE_AA);
               bev.copyTo(frame(cv::Rect(0, camera_resized.rows + kHudHeight, bev.cols, bev.rows)));

               if (k == 0) {
                 writer.open(out_video, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                             FLAGS_fps, frame.size());
                 if (!writer.isOpened()) {
                   use_video = false;
                   png_dir   = fs::path(out_video).replace_extension("");
                   fs::create_directories(png_dir);
                   LOGW("No video encoder; writing PNGs to %s", png_dir.c_str());
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
             });

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
      "  --mode=visualize    --engine_file=E --package_dir=DIR     camera overlay + BEV video");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_engine_file.empty()) {
    LOGE("--engine_file is required (see --help)");
    return 2;
  }

  if (FLAGS_mode == "build-engine") {
    return BuildEngine();
  }

  if (FLAGS_package_dir.empty()) {
    LOGE("--package_dir is required for --mode=%s", FLAGS_mode.c_str());
    return 2;
  }

  if (FLAGS_mode == "run-package") {
    return RunPackage();
  } else if (FLAGS_mode == "visualize") {
    return Visualize();
  }

  LOGE("Unknown --mode '%s' (options: build-engine, run-package, visualize)",
       FLAGS_mode.c_str());
  return 2;
}
