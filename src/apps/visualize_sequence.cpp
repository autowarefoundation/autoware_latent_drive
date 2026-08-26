// Stream a hand-off package clip through the planner and render each step as
// camera frame (top) + BEV trajectory view (bottom): prediction vs the
// PyTorch reference vs ground truth, plus subgoal, velocity and a HUD.
//
//   visualize_sequence --engine E --package DIR [--clip follow_lane]
//                      [--out out.mp4] [--fps 10] [--png-every N]
//
// Writes an mp4 (falls back to a PNG directory if OpenCV lacks a video
// encoder) and, with --png-every, standalone PNG stills for quick inspection.
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "latentdrive/bev_visualizer.hpp"
#include "latentdrive/camera_projector.hpp"
#include "latentdrive/frame_window.hpp"
#include "latentdrive/planner.hpp"
#include "npy.hpp"

namespace fs = std::filesystem;
using namespace latentdrive;

namespace {

Trajectory rowToTraj(const float* row /* [8,3] */) {
  Trajectory t;
  std::memcpy(t.data(), row, sizeof(Trajectory));
  return t;
}

// Calibration for this clip's scene, from the package's calibration.json
// (K_crop_resized matches the 512x256 package JPEGs). nullopt -> no overlay.
std::optional<CameraProjector> loadProjector(const fs::path& calib_path,
                                             const std::string& scene) {
  std::ifstream f(calib_path);
  if (!f) {
    std::cerr << "no " << calib_path << " -- camera overlay disabled\n";
    return std::nullopt;
  }
  const auto j = nlohmann::json::parse(f);
  if (!j["scenes"].contains(scene)) {
    std::cerr << "scene '" << scene << "' not in calibration.json -- overlay disabled\n";
    return std::nullopt;
  }
  const auto& s = j["scenes"][scene];
  CameraCalibration c;
  for (int r = 0; r < 3; ++r) {
    for (int col = 0; col < 3; ++col) {
      c.K(r, col) = s["K_crop_resized"][r][col].get<double>();
      c.R(r, col) = s["extrinsics_camera_to_ego"]["rotation_matrix"][r][col].get<double>();
    }
    c.t[r] = s["extrinsics_camera_to_ego"]["translation"][r].get<double>();
  }
  return CameraProjector(c);
}

// camera frame resized to the BEV width, HUD strip, BEV panel — one column
cv::Mat compose(const cv::Mat& camera, const cv::Mat& bev, const std::string& hud) {
  const int w = bev.cols;
  cv::Mat cam;
  cv::resize(camera, cam, cv::Size(w, camera.rows * w / camera.cols), 0, 0,
             cv::INTER_AREA);
  const int hud_h = 26;
  cv::Mat out(cam.rows + hud_h + bev.rows, w, CV_8UC3, cv::Scalar(25, 26, 26));
  cam.copyTo(out(cv::Rect(0, 0, w, cam.rows)));
  cv::putText(out, hud, {8, cam.rows + 18}, cv::FONT_HERSHEY_SIMPLEX, 0.44,
              cv::Scalar(183, 194, 195), 1, cv::LINE_AA);
  bev.copyTo(out(cv::Rect(0, cam.rows + hud_h, w, bev.rows)));
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  std::string engine_path, package, clip = "follow_lane", out_path, calib_path;
  int fps = 10, png_every = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--engine" && i + 1 < argc) engine_path = argv[++i];
    else if (a == "--package" && i + 1 < argc) package = argv[++i];
    else if (a == "--clip" && i + 1 < argc) clip = argv[++i];
    else if (a == "--out" && i + 1 < argc) out_path = argv[++i];
    else if (a == "--calibration" && i + 1 < argc) calib_path = argv[++i];
    else if (a == "--fps" && i + 1 < argc) fps = std::stoi(argv[++i]);
    else if (a == "--png-every" && i + 1 < argc) png_every = std::stoi(argv[++i]);
    else {
      std::cerr << "unknown arg: " << a << "\n";
      return 2;
    }
  }
  if (engine_path.empty() || package.empty()) {
    std::cerr << "usage: visualize_sequence --engine E --package DIR "
                 "[--clip C] [--out out.mp4] [--fps 10] [--png-every N]\n";
    return 2;
  }
  if (out_path.empty()) out_path = "viz_" + clip + ".mp4";
  if (calib_path.empty()) calib_path = (fs::path(package) / "calibration.json").string();

  try {
    const fs::path cdir = fs::path(package) / "sequence" / clip;
    std::ifstream jf(cdir / "sequence.json");
    const auto seq = nlohmann::json::parse(jf);
    const auto status_all = npy::load((cdir / "status.npy").string());
    const auto ref = npy::load((cdir / "ref_output_fp32.npy").string());
    const auto gt = npy::load((cdir / "gt_target.npy").string());
    const auto projector = loadProjector(calib_path, seq["scene"].get<std::string>());

    Planner planner(engine_path);
    FrameWindow window;
    BevVisualizer bev;

    std::vector<int> frame_numbers;
    for (const auto& fr : seq["frames"]) frame_numbers.push_back(fr["frame"].get<int>());
    const auto& steps = seq["steps"];

    cv::VideoWriter writer;
    bool use_video = true;
    fs::path png_dir;
    std::size_t next_frame = 0;
    cv::Mat camera;

    for (std::size_t k = 0; k < steps.size(); ++k) {
      const int step_frame = steps[k]["frame"].get<int>();
      while (next_frame < frame_numbers.size() &&
             frame_numbers[next_frame] <= step_frame) {
        char name[16];
        std::snprintf(name, sizeof(name), "%05d.jpg", frame_numbers[next_frame]);
        camera = cv::imread((cdir / "frames" / name).string(), cv::IMREAD_COLOR);
        window.push(camera);
        ++next_frame;
      }
      window.assemble(planner.videoInput());
      std::array<float, kStatusDim> status;
      std::copy_n(status_all.data.begin() + k * kStatusDim, kStatusDim, status.begin());
      const Trajectory pred = planner.inferPrepared(status);

      const Trajectory gt_k = rowToTraj(gt.data.data() + k * kNumWaypoints * 3);
      const Trajectory ref_k = rowToTraj(ref.data.data() + k * kNumWaypoints * 3);

      if (projector) {  // waypoint overlay on the camera frame (z=0 ground)
        projector->drawTrajectory(camera, gt_k, bev_color::kGroundTruth, 1,
                                  /*draw_dots=*/false);
        projector->drawTrajectory(camera, ref_k, bev_color::kReference, 1);
        projector->drawTrajectory(camera, pred, bev_color::kPrediction, 2);
      }

      bev.clear();
      bev.drawTrajectory(gt_k, bev_color::kGroundTruth, 1);
      bev.drawTrajectory(ref_k, bev_color::kReference, 2, /*dashed=*/true);
      bev.drawTrajectory(pred, bev_color::kPrediction, 2);
      bev.drawSubgoal(status[0] * 10.f, status[1] * 10.f);
      bev.drawVelocity(status[2], status[3]);
      bev.addLegend("TRT fp16 pred", bev_color::kPrediction);
      bev.addLegend("PyTorch ref", bev_color::kReference);
      bev.addLegend("ground truth", bev_color::kGroundTruth);
      bev.addLegend("subgoal 50m", bev_color::kSubgoal);

      const float wp8 = std::hypot(pred[7].x, pred[7].y);
      const float speed = std::hypot(status[2], status[3]);
      double err = 0;  // per-step max waypoint deviation vs the PyTorch reference
      for (int i = 0; i < kNumWaypoints; ++i) {
        const float* r = ref.data.data() + (k * kNumWaypoints + i) * 3;
        err = std::max(err, std::hypot(double(pred[i].x - r[0]), double(pred[i].y - r[1])));
      }
      const std::string hud = cv::format(
          "%3zu/%zu %s  v %4.1f  wp8 %5.2fm  err %.3f", k, steps.size() - 1,
          steps[k]["phase"].get<std::string>().c_str(), speed, wp8, err);

      const cv::Mat frame = compose(camera, bev.image(), hud);

      if (k == 0) {
        writer.open(out_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                    fps, frame.size());
        if (!writer.isOpened()) {
          use_video = false;
          png_dir = fs::path(out_path).replace_extension("");
          fs::create_directories(png_dir);
          std::cerr << "no video encoder; writing PNGs to " << png_dir << "\n";
        }
      }
      if (use_video) {
        writer << frame;
      } else {
        cv::imwrite((png_dir / cv::format("step_%04zu.png", k)).string(), frame);
      }
      if (png_every > 0 && k % png_every == 0) {
        cv::imwrite(fs::path(out_path).replace_extension("").string() +
                        cv::format("_step%04zu.png", k),
                    frame);
      }
    }
    if (use_video) writer.release();
    std::cout << "wrote " << (use_video ? out_path : png_dir.string()) << " ("
              << steps.size() << " steps @ " << fps << " fps)\n";
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
