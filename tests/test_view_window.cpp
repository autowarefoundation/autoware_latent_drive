// Unit tests for the timestamped multi-view window: history picked by nearest
// stamp (planner_io.pick_history), per-slot alignment of the other views
// within a tolerance, and the [V, 3, T, H, W] tensor layout. Plain asserts, no
// test framework -- the deployment container carries none we want to depend
// on, and the checks are few
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <opencv2/core.hpp>
#include <vector>

#include "frame_window.hpp"
#include "planner_params.hpp"

using model::planner::kFrameNumel;
using model::planner::kImageNetMean;
using model::planner::kImageNetStd;
using model::planner::kInputHeight;
using model::planner::kInputWidth;
using model::planner::kNumHistFrames;
using model::planner::kVideoNumel;

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

// Bag-like epoch: the arithmetic has to hold at 1.8e9 s, not just near zero
constexpr double kEpoch = 1787830149.9;

static std::vector<double> Stamps10Hz(int n, double offset_s = 0.0, double jitter_s = 0.0) {
  std::vector<double> stamps;
  for (int i = 0; i < n; ++i) {
    // deterministic +-jitter, alternating sign
    const double j = (i % 2 ? 1.0 : -1.0) * jitter_s * ((i % 3) / 2.0);
    stamps.push_back(kEpoch + offset_s + 0.1 * i + j);
  }
  return stamps;
}

static void TestPickHistory() {
  // A healthy 10 Hz stream: t-1.5, t-1.0, t-0.5, t are 5 frames apart
  const auto regular = Stamps10Hz(20);
  const auto picked  = preprocess::PickHistory(regular, 19);
  CHECK(picked[0] == 4 && picked[1] == 9 && picked[2] == 14 && picked[3] == 19);

  // Anchored anywhere, not only at the newest frame
  const auto mid = preprocess::PickHistory(regular, 16);
  CHECK(mid[0] == 1 && mid[1] == 6 && mid[2] == 11 && mid[3] == 16);

  // Stamp jitter of +-20 ms does not move a slot
  const auto jittered = preprocess::PickHistory(Stamps10Hz(20, 0.0, 0.02), 19);
  CHECK(jittered[0] == 4 && jittered[1] == 9 && jittered[2] == 14 && jittered[3] == 19);

  // Dropped frames: counting 5 back would land 0.2 s off for every older
  // slot; by time only the slot whose frame is missing moves, to its nearest
  // neighbour. (+0.9 and +1.0 gone: the t-1.0 slot takes +0.8, 0.1 s away,
  // over +1.1, 0.2 s away -- an asymmetric gap, so no floating-point tie)
  std::vector<double> dropped = regular;
  dropped.erase(dropped.begin() + 9, dropped.begin() + 11);    // index 8 = +0.8, 9 = +1.1
  const auto with_gap = preprocess::PickHistory(dropped, 17);  // now = +1.9
  CHECK(with_gap[0] == 4 && with_gap[1] == 8 && with_gap[2] == 12 && with_gap[3] == 17);

  // Unsorted input is fine: selection is by value, not by position
  std::vector<double> shuffled      = {regular[19], regular[4], regular[14], regular[9], regular[0]};
  const auto          from_shuffled = preprocess::PickHistory(shuffled, 0);
  CHECK(from_shuffled[0] == 1 && from_shuffled[1] == 3 && from_shuffled[2] == 2 && from_shuffled[3] == 0);

  // Too little history: the nearest frame is simply the oldest one (no
  // tolerance in pick_history itself -- ViewWindow gates on the span)
  const auto short_hist = preprocess::PickHistory(Stamps10Hz(6), 5);
  CHECK(short_hist[0] == 0 && short_hist[1] == 0 && short_hist[2] == 0 && short_hist[3] == 5);

  const auto bad = preprocess::PickHistory(regular, 20);
  CHECK(bad[0] == -1 && bad[3] == -1);
}

static void TestNearestWithin() {
  const std::vector<double> stamps = {kEpoch + 0.935, kEpoch + 1.07};
  CHECK(preprocess::NearestWithin(stamps, kEpoch + 1.0, 0.06) == -1);  // 65 ms away: rejected
  CHECK(preprocess::NearestWithin(stamps, kEpoch + 1.0, 0.07) == 0);   // inclusive
  CHECK(preprocess::NearestWithin(stamps, kEpoch + 1.05, 0.06) == 1);
  CHECK(preprocess::NearestWithin({}, kEpoch, 1.0) == -1);
}

// A frame whose every pixel is (b, g, r): after preprocessing each channel
// plane is one constant, which makes the layout checkable per plane
static cv::Mat Solid(int b, int g, int r, int rows = kInputHeight, int cols = kInputWidth) {
  return cv::Mat(rows, cols, CV_8UC3, cv::Scalar(b, g, r));
}

static double Normalized(int value, int channel) {
  return (value / 255.0 - kImageNetMean[channel]) / kImageNetStd[channel];
}

static void TestTwoViewAlignmentAndLayout() {
  preprocess::ViewWindowParams params;
  params.num_views        = 2;
  params.view_tolerance_s = 0.06;
  auto window             = preprocess::CreateViewWindow(params);
  CHECK(window != nullptr);

  // Wide: 16 frames at 10 Hz. Telephoto: the same cadence, 30 ms later, as
  // an unsynchronised second camera would deliver. Colour encodes (view,
  // frame) so the tensor can be read back
  const auto wide_stamps = Stamps10Hz(16);
  const auto far_stamps  = Stamps10Hz(16, 0.03);
  for (int i = 0; i < 16; ++i) {
    window->Push(0, Solid(10 + i, 60 + i, 110 + i), wide_stamps[i]);
    if (i < 15) {
      // not ready yet: 1.5 s of wide history takes 16 frames, as with the
      // count-based window
      const auto early = window->Select();
      CHECK(!early.ok && early.failed_view == 0);
    }
    window->Push(1, Solid(160 + i, 200 + i, 240 - i), far_stamps[i]);
  }

  std::vector<float> video(2 * kVideoNumel + 8, 12345.f);  // sentinel tail
  const auto         sel = window->Assemble(video.data());
  CHECK(sel.ok);
  if (!sel.ok) std::fprintf(stderr, "  reason: %s\n", sel.reason.c_str());
  CHECK(sel.stamps_s.size() == 2);
  // Wide slots: frames 0, 5, 10, 15 (t-1.5 .. t); far: the frame 30 ms after each
  const int slots[kNumHistFrames] = {0, 5, 10, 15};
  for (int t = 0; t < kNumHistFrames; ++t) {
    CHECK_NEAR(sel.stamps_s[0][t], wide_stamps[slots[t]], 1e-6);
    CHECK_NEAR(sel.stamps_s[1][t], far_stamps[slots[t]], 1e-6);
  }
  CHECK_NEAR(sel.worst_view_offset_s, 0.03, 1e-6);
  CHECK_NEAR(sel.worst_history_offset_s, 0.0, 1e-6);

  // Layout: [V, 3, T, H, W]. View 1's block follows view 0's; inside a view
  // the planes are channel-major: plane (c, t) at (c*T + t) * H*W
  const std::size_t plane = static_cast<std::size_t>(kInputHeight) * kInputWidth;
  CHECK(2 * kVideoNumel == 2 * 3 * kNumHistFrames * static_cast<int>(plane));
  for (int v = 0; v < 2; ++v) {
    for (int c = 0; c < 3; ++c) {
      for (int t = 0; t < kNumHistFrames; ++t) {
        const int    i      = slots[t];
        const int    bgr[3] = {v == 0 ? 10 + i : 160 + i, v == 0 ? 60 + i : 200 + i, v == 0 ? 110 + i : 240 - i};
        const int    rgb    = bgr[2 - c];  // RGB order in the tensor
        const float *p      = video.data() + ((static_cast<std::size_t>(v) * 3 + c) * kNumHistFrames + t) * plane;
        CHECK_NEAR(p[0], Normalized(rgb, c), 1e-5);
        CHECK_NEAR(p[plane / 2], Normalized(rgb, c), 1e-5);
        CHECK_NEAR(p[plane - 1], Normalized(rgb, c), 1e-5);
      }
    }
  }
  // View 1 starts exactly one single-view tensor after view 0
  CHECK_NEAR(video[kVideoNumel], Normalized(240 - 0, 0), 1e-5);      // v=1, c=0 (R), t=0 -> frame 0
  CHECK_NEAR(video[kVideoNumel - 1], Normalized(10 + 15, 2), 1e-5);  // v=0, c=2 (B), t=3 -> frame 15
  // Nothing written past the tensor
  CHECK(video[2 * kVideoNumel] == 12345.f && video[2 * kVideoNumel + 7] == 12345.f);

  // Tolerance rejection: the telephoto frames sit 50 ms after the wide ones.
  // 60 ms accepts them, 40 ms must not -- and must not fill the slot with
  // anything else
  {
    for (double tol : {0.06, 0.04}) {
      preprocess::ViewWindowParams p2 = params;
      p2.view_tolerance_s             = tol;
      auto       w2                   = preprocess::CreateViewWindow(p2);
      const auto far50                = Stamps10Hz(16, 0.05);
      for (int i = 0; i < 16; ++i) {
        w2->Push(0, Solid(1, 2, 3), wide_stamps[i]);
        w2->Push(1, Solid(4, 5, 6), far50[i]);
      }
      std::vector<float> buf(2 * kVideoNumel, 777.f);
      const auto         s2 = w2->Assemble(buf.data());
      if (tol > 0.045) {
        CHECK(s2.ok);
        CHECK_NEAR(s2.worst_view_offset_s, 0.05, 1e-6);
      } else {
        CHECK(!s2.ok && s2.failed_view == 1 && s2.failed_slot == 0);
        CHECK(buf[0] == 777.f && buf[2 * kVideoNumel - 1] == 777.f);  // untouched
      }
    }
  }

  // A hole in the telephoto stream around one wide slot (t-1.0: the far
  // frames at +0.43 and +0.53 never arrived) fails that slot only, by name
  {
    auto w3 = preprocess::CreateViewWindow(params);
    for (int i = 0; i < 16; ++i) {
      w3->Push(0, Solid(1, 2, 3), wide_stamps[i]);
      if (i == 4 || i == 5) continue;  // far stamps +0.43, +0.53 missing
      w3->Push(1, Solid(4, 5, 6), far_stamps[i]);
    }
    const auto s3 = w3->Select();
    CHECK(!s3.ok && s3.failed_view == 1 && s3.failed_slot == 1);
    CHECK(s3.reason.find("view 1 slot 1") != std::string::npos);
    // the wide slots were resolved before the failure
    CHECK_NEAR(s3.stamps_s[0][1], wide_stamps[5], 1e-6);
    CHECK(std::isnan(s3.stamps_s[1][1]));
  }

  // The newest slot is the one that can still be waiting for its far frame:
  // with the far stream lagging, Select fails on slot 3 and NewestStamp tells
  // the caller whether a nearer frame can still arrive
  {
    auto w4 = preprocess::CreateViewWindow(params);
    for (int i = 0; i < 16; ++i) {
      w4->Push(0, Solid(1, 2, 3), wide_stamps[i]);
      if (i < 15) w4->Push(1, Solid(4, 5, 6), far_stamps[i]);
    }
    const auto s4 = w4->Select();
    CHECK(!s4.ok && s4.failed_view == 1 && s4.failed_slot == 3);
    CHECK_NEAR(w4->NewestStamp(1), far_stamps[14], 1e-6);
    CHECK(w4->NewestStamp(1) < wide_stamps[15] + params.view_tolerance_s);
    w4->Push(1, Solid(4, 5, 6), far_stamps[15]);
    CHECK(w4->Select().ok);
  }
}

static void TestSingleViewWindow() {
  preprocess::ViewWindowParams params;  // one view
  auto                         window = preprocess::CreateViewWindow(params);
  const auto                   stamps = Stamps10Hz(20);
  for (int i = 0; i < 20; ++i) window->Push(0, Solid(i, i, i), stamps[i]);

  std::vector<float> video(kVideoNumel + 4, -1.f);
  const auto         sel = window->Assemble(video.data());
  CHECK(sel.ok && sel.stamps_s.size() == 1);
  // frames 4, 9, 14, 19 -- what the count-based FrameWindow(stride 5) picks
  // on the same healthy stream
  const std::size_t plane                 = static_cast<std::size_t>(kInputHeight) * kInputWidth;
  const int         slots[kNumHistFrames] = {4, 9, 14, 19};
  for (int t = 0; t < kNumHistFrames; ++t) {
    CHECK_NEAR(sel.stamps_s[0][t], stamps[slots[t]], 1e-6);
    for (int c = 0; c < 3; ++c) {
      CHECK_NEAR(video[(c * kNumHistFrames + t) * plane], Normalized(slots[t], c), 1e-5);
    }
  }
  CHECK(video[kVideoNumel] == -1.f);

  // A stamp that jumps back (bag loop) drops the view's history
  window->Push(0, Solid(0, 0, 0), stamps[0] - 10.0);
  CHECK(window->Size(0) == 1);
  CHECK(!window->Select().ok);
  CHECK(std::isnan(preprocess::CreateViewWindow(params)->NewestStamp(0)));

  // Parameter validation
  preprocess::ViewWindowParams bad;
  bad.num_views = 0;
  CHECK(preprocess::CreateViewWindow(bad) == nullptr);
}

static void TestCropBand() {
  // planner_io.preprocess_frame on a 2880x1860 frame: keep the full width,
  // rows [210, 1650) are the 2:1 band. Paint the band white and the rest
  // black: the tensor must come out all white, with no black bleeding in
  cv::Mat frame = Solid(0, 0, 0, 1860, 2880);
  frame(cv::Rect(0, 210, 2880, 1440)).setTo(cv::Scalar(255, 255, 255));
  std::vector<float> chw(kFrameNumel);
  CHECK(preprocess::PreprocessFrame(frame, chw.data()));
  for (int c = 0; c < 3; ++c) {
    const std::size_t plane = static_cast<std::size_t>(kInputHeight) * kInputWidth;
    CHECK_NEAR(chw[c * plane], Normalized(255, c), 1e-5);              // top-left
    CHECK_NEAR(chw[c * plane + plane - 1], Normalized(255, c), 1e-5);  // bottom-right
    CHECK_NEAR(chw[c * plane + (kInputHeight / 2) * kInputWidth + kInputWidth / 2], Normalized(255, c), 1e-5);
  }
  // one row above the band is black in the source and must not survive
  cv::Mat shifted = Solid(255, 255, 255, 1860, 2880);
  shifted(cv::Rect(0, 0, 2880, 210)).setTo(cv::Scalar(0, 0, 0));
  shifted(cv::Rect(0, 1650, 2880, 210)).setTo(cv::Scalar(0, 0, 0));
  std::vector<float> chw2(kFrameNumel);
  CHECK(preprocess::PreprocessFrame(shifted, chw2.data()));
  CHECK_NEAR(chw2[0], Normalized(255, 0), 1e-5);
}

int main() {
  TestPickHistory();
  TestNearestWithin();
  TestTwoViewAlignmentAndLayout();
  TestSingleViewWindow();
  TestCropBand();
  if (g_failures) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("all view window checks passed\n");
  return 0;
}
