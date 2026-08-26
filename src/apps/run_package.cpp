// Replay the LatentDrive hand-off package (workspace/) through the C++ stack:
// samples for numeric alignment, sequences for the streaming loop, and a
// batch-1 latency benchmark. Writes the engine_output .npy files that the
// package's validate.py scores.
//
//   run_package --engine e2e-fp16-mixed.engine --package /path/to/workspace
//               [--out-name engine_output.npy] [--skip-benchmark]
//
// then:  (cd /path/to/workspace && python3 validate.py)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>

#include "latentdrive/frame_window.hpp"
#include "latentdrive/planner.hpp"
#include "npy.hpp"

namespace fs = std::filesystem;
using namespace latentdrive;
using Clock = std::chrono::steady_clock;

namespace {

double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

double percentile(std::vector<double> v, double p) {
  std::sort(v.begin(), v.end());
  const double idx = p / 100.0 * (v.size() - 1);
  const std::size_t lo = static_cast<std::size_t>(idx);
  const std::size_t hi = std::min(lo + 1, v.size() - 1);
  return v[lo] + (v[hi] - v[lo]) * (idx - lo);
}

double max_xy_err(const Trajectory& t, const float* ref /* [8,3] */) {
  double worst = 0;
  for (int i = 0; i < kNumWaypoints; ++i) {
    const double dx = t[i].x - ref[i * 3 + 0];
    const double dy = t[i].y - ref[i * 3 + 1];
    worst = std::max(worst, std::hypot(dx, dy));
  }
  return worst;
}

std::vector<fs::path> sorted_dirs(const fs::path& parent) {
  std::vector<fs::path> dirs;
  for (const auto& e : fs::directory_iterator(parent)) {
    if (e.is_directory()) dirs.push_back(e.path());
  }
  std::sort(dirs.begin(), dirs.end());
  return dirs;
}

void run_samples(Planner& planner, const fs::path& pkg, const std::string& out_name) {
  std::cout << "== samples/ (packaged video.npy)\n";
  for (const auto& sdir : sorted_dirs(pkg / "samples")) {
    const auto video = npy::load((sdir / "video.npy").string());
    const auto status_arr = npy::load((sdir / "status.npy").string());
    const auto ref = npy::load((sdir / "ref_output_fp32.npy").string());
    std::array<float, kStatusDim> status;
    std::copy_n(status_arr.data.begin(), kStatusDim, status.begin());

    const Trajectory traj = planner.infer(video.data.data(), status);
    npy::save((sdir / out_name).string(), &traj[0].x, {kNumWaypoints, 3});
    std::printf("  %-32s xy max vs ref %.4f m\n", sdir.filename().c_str(),
                max_xy_err(traj, ref.data.data()));
  }
}

void check_preprocessing(const fs::path& pkg) {
  std::cout << "== preprocessing self-test (frames/*.jpg vs packaged video.npy)\n";
  double worst = 0;
  for (const auto& sdir : sorted_dirs(pkg / "samples")) {
    const auto video = npy::load((sdir / "video.npy").string());
    std::vector<fs::path> jpgs;
    for (const auto& e : fs::directory_iterator(sdir / "frames")) jpgs.push_back(e.path());
    std::sort(jpgs.begin(), jpgs.end());  // slot0..slot3

    std::vector<float> frame(kFrameNumel);
    double diff = 0;
    constexpr std::size_t plane = static_cast<std::size_t>(kInputHeight) * kInputWidth;
    for (int t = 0; t < kNumHistFrames; ++t) {
      FrameWindow::preprocess(cv::imread(jpgs[t].string(), cv::IMREAD_COLOR), frame.data());
      for (int c = 0; c < 3; ++c) {
        const float* ours = frame.data() + c * plane;
        const float* ref = video.data.data() + (c * kNumHistFrames + t) * plane;
        for (std::size_t i = 0; i < plane; ++i) {
          diff = std::max(diff, static_cast<double>(std::abs(ours[i] - ref[i])));
        }
      }
    }
    worst = std::max(worst, diff);
    std::printf("  %-32s max abs diff %.2e\n", sdir.filename().c_str(), diff);
  }
  std::printf("  worst: %.2e (%s)\n", worst,
              worst < 1e-5 ? "same decoder" : "different JPEG decoder -- fine if ~1e-3");
}

void run_sequence(Planner& planner, const fs::path& cdir, const std::string& out_name,
                  bool benchmark) {
  std::ifstream jf(cdir / "sequence.json");
  const auto seq = nlohmann::json::parse(jf);
  const auto status_all = npy::load((cdir / "status.npy").string());
  const auto ref = npy::load((cdir / "ref_output_fp32.npy").string());

  // frames are consecutive; push them in order and the stride-5 window in
  // FrameWindow reproduces each step's history_frames exactly
  std::vector<int> frame_numbers;
  for (const auto& fr : seq["frames"]) frame_numbers.push_back(fr["frame"].get<int>());

  FrameWindow window;
  std::vector<float> plans;
  std::vector<double> t_pre, t_inf;
  std::size_t next_frame = 0;
  const auto& steps = seq["steps"];

  for (std::size_t k = 0; k < steps.size(); ++k) {
    const int step_frame = steps[k]["frame"].get<int>();
    const auto t0 = Clock::now();
    while (next_frame < frame_numbers.size() && frame_numbers[next_frame] <= step_frame) {
      char name[16];
      std::snprintf(name, sizeof(name), "%05d.jpg", frame_numbers[next_frame]);
      window.push(cv::imread((cdir / "frames" / name).string(), cv::IMREAD_COLOR));
      ++next_frame;
    }
    window.assemble(planner.videoInput());
    const double pre_ms = ms_since(t0);

    const auto t1 = Clock::now();
    std::array<float, kStatusDim> status;
    std::copy_n(status_all.data.begin() + k * kStatusDim, kStatusDim, status.begin());
    const Trajectory traj = planner.inferPrepared(status);
    t_inf.push_back(ms_since(t1));
    t_pre.push_back(pre_ms);
    plans.insert(plans.end(), &traj[0].x, &traj[0].x + kNumWaypoints * 3);
  }

  const std::size_t n = steps.size();
  npy::save((cdir / out_name).string(), plans.data(), {n, kNumWaypoints, 3});

  double worst = 0, sum = 0;
  for (std::size_t k = 0; k < n; ++k) {
    Trajectory t;
    std::memcpy(t.data(), plans.data() + k * kNumWaypoints * 3, sizeof(Trajectory));
    const double e = max_xy_err(t, ref.data.data() + k * kNumWaypoints * 3);
    worst = std::max(worst, e);
    sum += e;
  }
  std::printf("  %-16s %zu steps  xy max vs ref %.4f m  mean-of-max %.4f m\n",
              cdir.filename().c_str(), n, worst, sum / n);

  if (benchmark) {
    std::vector<double> pre(t_pre.begin() + 5, t_pre.end());
    std::vector<double> inf(t_inf.begin() + 5, t_inf.end());
    std::printf("  benchmark (%s, steps 5..%zu, batch 1):\n", cdir.filename().c_str(), n - 1);
    std::printf("    engine call   median %7.2f ms   p99 %7.2f ms   max %7.2f ms\n",
                percentile(inf, 50), percentile(inf, 99), percentile(inf, 100));
    std::printf("    preprocessing median %7.2f ms   p99 %7.2f ms   (decode+normalize new frames + window assembly)\n",
                percentile(pre, 50), percentile(pre, 99));
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string engine_path, package, out_name = "engine_output.npy";
  bool skip_benchmark = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--engine" && i + 1 < argc) engine_path = argv[++i];
    else if (a == "--package" && i + 1 < argc) package = argv[++i];
    else if (a == "--out-name" && i + 1 < argc) out_name = argv[++i];
    else if (a == "--skip-benchmark") skip_benchmark = true;
    else {
      std::cerr << "unknown arg: " << a << "\n"
                << "usage: run_package --engine E --package DIR "
                   "[--out-name F.npy] [--skip-benchmark]\n";
      return 2;
    }
  }
  if (engine_path.empty() || package.empty()) {
    std::cerr << "usage: run_package --engine E --package DIR "
                 "[--out-name F.npy] [--skip-benchmark]\n";
    return 2;
  }

  try {
    Planner planner(engine_path);
    std::cout << "engine: " << engine_path << "\n";
    // warmup
    std::vector<float> zeros(kVideoNumel, 0.f);
    for (int i = 0; i < 10; ++i) planner.infer(zeros.data(), std::array<float, kStatusDim>{});

    const fs::path pkg(package);
    run_samples(planner, pkg, out_name);
    check_preprocessing(pkg);
    std::cout << "== sequence/ (streaming from frames/*.jpg)\n";
    for (const auto& cdir : sorted_dirs(pkg / "sequence")) {
      run_sequence(planner, cdir, out_name,
                   !skip_benchmark && cdir.filename() == "follow_lane");
    }
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
