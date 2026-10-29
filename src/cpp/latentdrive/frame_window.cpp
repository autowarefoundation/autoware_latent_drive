#include "frame_window.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <limits>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "logger.hpp"

namespace preprocess {

using model::planner::kFrameNumel;
using model::planner::kImageNetMean;
using model::planner::kImageNetStd;
using model::planner::kInputHeight;
using model::planner::kInputWidth;
using model::planner::kNumHistFrames;
using model::planner::kVideoNumel;

constexpr std::size_t kPlane = static_cast<std::size_t>(kInputHeight) * kInputWidth;

bool PreprocessFrame(const cv::Mat &in_frame, float *out_chw, bool in_is_rgb) {
  if (in_frame.empty() || in_frame.channels() != 3) {
    LOGE("Preprocess expects a non-empty 3-channel frame");
    return false;
  }

  // Vehicle preprocessing (planner_io.preprocess_frame): keep the full width,
  // center-crop the rows to a 2:1 band, then resize to the model input. Every
  // camera goes through this; a side or telephoto frame is a separate
  // sensor's full frame, not a crop of the front one. On the hand-off package
  // the JPEGs are already 512x256, so both steps are no-ops there. The colour
  // swap comes last: bilinear resizing treats the channels independently, so
  // swapping on the 512x256 result gives the same numbers as swapping the
  // 2880x1860 frame first, at a fortieth of the cost
  cv::Mat   band     = in_frame;
  const int target_h = band.cols / 2;
  if (band.rows > target_h) {
    const int y0 = (band.rows - target_h) / 2;
    band         = band(cv::Rect(0, y0, band.cols, target_h));
  }
  cv::Mat resized = band;
  if (band.rows != kInputHeight || band.cols != kInputWidth) {
    cv::resize(band, resized, cv::Size(kInputWidth, kInputHeight), 0, 0, cv::INTER_LINEAR);
  }
  cv::Mat rgb;
  if (in_is_rgb) {
    rgb = resized;
  } else {
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
  }

  cv::Mat frame_f32;
  rgb.convertTo(frame_f32, CV_32FC3, 1.0 / 255.0);

  // HWC -> CHW with the per-channel ImageNet normalization
  for (int c = 0; c < 3; ++c) {
    const float mean    = kImageNetMean[c];
    const float inv_std = 1.f / kImageNetStd[c];
    float      *plane   = out_chw + c * kInputHeight * kInputWidth;
    for (int y = 0; y < kInputHeight; ++y) {
      const float *row = frame_f32.ptr<float>(y);
      float       *out = plane + y * kInputWidth;
      for (int x = 0; x < kInputWidth; ++x) {
        out[x] = (row[x * 3 + c] - mean) * inv_std;
      }
    }
  }
  return true;
}

// @brief: This function is used to copy one preprocessed CHW frame into slot
// `t` of view `v` of a [V, 3, T, H, W] tensor: channel-major inside a view,
// so the frame contributes one H*W plane per channel at
// out[((v*3 + c) * T + t) * H*W], and view v+1's block starts right after
// view v's 3*T planes
static void PlaceFrame(float *out_video, int view, int slot, const float *chw) {
  for (int c = 0; c < 3; ++c) {
    std::memcpy(out_video + ((static_cast<std::size_t>(view) * 3 + c) * kNumHistFrames + slot) * kPlane,
                chw + c * kPlane, kPlane * sizeof(float));
  }
}

// `FrameWindowImpl` is an implementation class. The ring buffer holds the
// preprocessed CHW frames; eviction is a pop_front once over capacity
class FrameWindowImpl : public FrameWindow {
 public:
  explicit FrameWindowImpl(int stride_frames)
      : stride_(stride_frames), capacity_(3 * stride_frames + 1) {}

  FrameWindowImpl(const FrameWindowImpl &)            = delete;
  FrameWindowImpl &operator=(const FrameWindowImpl &) = delete;
  ~FrameWindowImpl() override                         = default;

  void Push(const cv::Mat &in_frame, bool in_is_rgb) final {
    std::vector<float> buffer(kFrameNumel);
    if (!PreprocessFrame(in_frame, buffer.data(), in_is_rgb)) return;
    frames_.push_back(std::move(buffer));
    if (static_cast<int>(frames_.size()) > capacity_) {
      frames_.pop_front();
    }
  }

  bool Ready() const final {
    return static_cast<int>(frames_.size()) >= capacity_;
  }

  void Assemble(float *out_video) const final {
    if (!Ready()) {
      LOGF("FrameWindow::Assemble called before Ready (%d/%d frames)",
           static_cast<int>(frames_.size()), capacity_);
    }
    // Slots oldest..newest: frames_[0], [stride], [2*stride], [3*stride],
    // laid out [3, T, H, W] -- the single-view block of PlaceFrame
    for (int t = 0; t < kNumHistFrames; ++t) {
      PlaceFrame(out_video, 0, t, frames_[static_cast<std::size_t>(t) * stride_].data());
    }
  }

  void Reset() final { frames_.clear(); }

  int Size() const final { return static_cast<int>(frames_.size()); }

 private:
  int stride_;
  int capacity_;

  std::deque<std::vector<float>> frames_;
};

std::shared_ptr<FrameWindow> CreateFrameWindow(int stride_frames) {
  if (stride_frames < 1) {
    LOGE("FrameWindow stride must be >= 1, got %d", stride_frames);
    return nullptr;
  }
  return std::shared_ptr<FrameWindow>(new FrameWindowImpl(stride_frames));
}

/****************************************/
/*******Timestamped, multi-view**********/
/****************************************/

// @brief: This function is used to find the index of the stamp nearest to
// `target_s` (first one on a tie, like numpy's argmin); -1 on an empty list
static int Nearest(const std::vector<double> &stamps_s, double target_s) {
  int    best   = -1;
  double best_d = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < stamps_s.size(); ++i) {
    const double d = std::abs(stamps_s[i] - target_s);
    if (d < best_d) {
      best_d = d;
      best   = static_cast<int>(i);
    }
  }
  return best;
}

std::array<int, kNumHistFrames> PickHistory(const std::vector<double> &stamps_s, int now_index,
                                            double hist_dt_s) {
  std::array<int, kNumHistFrames> picked{};
  picked.fill(-1);
  if (now_index < 0 || now_index >= static_cast<int>(stamps_s.size())) return picked;
  const double t0 = stamps_s[static_cast<std::size_t>(now_index)];
  // k = 3..0 -> targets t0-1.5, t0-1.0, t0-0.5, t0, i.e. oldest first
  for (int k = kNumHistFrames - 1; k >= 0; --k) {
    picked[static_cast<std::size_t>(kNumHistFrames - 1 - k)] = Nearest(stamps_s, t0 - k * hist_dt_s);
  }
  return picked;
}

int NearestWithin(const std::vector<double> &stamps_s, double target_s, double tolerance_s) {
  const int i = Nearest(stamps_s, target_s);
  if (i < 0 || std::abs(stamps_s[static_cast<std::size_t>(i)] - target_s) > tolerance_s) return -1;
  return i;
}

// `ViewWindowImpl` is an implementation class. One deque of stamped,
// preprocessed CHW frames per view; eviction is by age relative to the
// view's own newest stamp, so a camera that stops does not lose its history
// while the others move on (its frames simply fall out of tolerance)
class ViewWindowImpl : public ViewWindow {
 public:
  explicit ViewWindowImpl(const ViewWindowParams &params)
      : params_(params), views_(static_cast<std::size_t>(params.num_views)) {
    // An all-zero image after /255 and the ImageNet normalization
    blank_chw_.resize(kFrameNumel);
    for (int c = 0; c < 3; ++c) {
      std::fill_n(blank_chw_.begin() + static_cast<std::ptrdiff_t>(c * kPlane), kPlane,
                  -kImageNetMean[c] / kImageNetStd[c]);
    }
  }

  ViewWindowImpl(const ViewWindowImpl &)            = delete;
  ViewWindowImpl &operator=(const ViewWindowImpl &) = delete;
  ~ViewWindowImpl() override                        = default;

  void Push(int view, const cv::Mat &in_frame, double stamp_s, bool in_is_rgb) final {
    std::vector<float> chw(kFrameNumel);
    if (!PreprocessFrame(in_frame, chw.data(), in_is_rgb)) return;
    PushPreprocessed(view, std::move(chw), stamp_s);
  }

  void PushPreprocessed(int view, std::vector<float> &&in_chw, double stamp_s) final {
    if (view < 0 || view >= params_.num_views) {
      LOGE("ViewWindow::Push: view %d out of range (%d views)", view, params_.num_views);
      return;
    }
    if (in_chw.size() != static_cast<std::size_t>(kFrameNumel)) {
      LOGE("ViewWindow::PushPreprocessed: %zu floats, a frame is %d", in_chw.size(), kFrameNumel);
      return;
    }
    View &v = views_[static_cast<std::size_t>(view)];
    // A stamp that runs backwards by more than the retention span is a bag
    // that looped or seeked: the frames in the deque are from another time
    if (!v.frames.empty() && stamp_s < v.newest_s - params_.retain_s) {
      LOGW("ViewWindow: view %d stamp jumped back %.2f s -- dropping its history", view,
           v.newest_s - stamp_s);
      v.frames.clear();
    }
    Frame frame;
    frame.stamp_s = stamp_s;
    frame.chw     = std::move(in_chw);
    v.frames.push_back(std::move(frame));
    v.newest_s = v.frames.size() == 1 ? stamp_s : std::max(v.newest_s, stamp_s);
    while (!v.frames.empty() && v.frames.front().stamp_s < v.newest_s - params_.retain_s) {
      v.frames.pop_front();
    }
  }

  ViewSelection Select() const final {
    ViewSelection                                sel;
    std::vector<std::array<int, kNumHistFrames>> indices;
    SelectIndices(sel, indices);
    return sel;
  }

  ViewSelection Assemble(float *out_video) const final {
    ViewSelection                                sel;
    std::vector<std::array<int, kNumHistFrames>> indices;
    if (!SelectIndices(sel, indices)) return sel;
    for (int v = 0; v < params_.num_views; ++v) {
      const View &view = views_[static_cast<std::size_t>(v)];
      for (int t = 0; t < kNumHistFrames; ++t) {
        const int i = indices[static_cast<std::size_t>(v)][static_cast<std::size_t>(t)];
        PlaceFrame(out_video, v, t, i < 0 ? blank_chw_.data() : view.frames[static_cast<std::size_t>(i)].chw.data());
      }
    }
    return sel;
  }

  int NumViews() const final { return params_.num_views; }

  int Size(int view) const final {
    if (view < 0 || view >= params_.num_views) return 0;
    return static_cast<int>(views_[static_cast<std::size_t>(view)].frames.size());
  }

  double NewestStamp(int view) const final {
    if (view < 0 || view >= params_.num_views || views_[static_cast<std::size_t>(view)].frames.empty()) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    return views_[static_cast<std::size_t>(view)].newest_s;
  }

  void Reset() final {
    for (View &v : views_) v.frames.clear();
  }

 private:
  struct Frame {
    double             stamp_s = 0.0;
    std::vector<float> chw;
  };
  struct View {
    std::deque<Frame> frames;
    double            newest_s = 0.0;
  };

  std::vector<double> Stamps(const View &view) const {
    std::vector<double> stamps;
    stamps.reserve(view.frames.size());
    for (const Frame &f : view.frames) stamps.push_back(f.stamp_s);
    return stamps;
  }

  // @brief: The selection proper. View 0's newest frame is "now"; its four
  // slots come from PickHistory once the wide history reaches back the full
  // 1.5 s (within history_tolerance_s -- the warm-up gate, the timestamp
  // counterpart of FrameWindow's 16-frame wait). The slots themselves carry
  // no tolerance, exactly like planner_io.pick_history: a dropped wide frame
  // moves one slot to its neighbour rather than skipping the plan. Every
  // other view then takes, per slot, its frame nearest to the chosen wide
  // frame's stamp, within view_tolerance_s -- no frame there and the
  // selection fails, naming the view and slot
  bool SelectIndices(ViewSelection &sel, std::vector<std::array<int, kNumHistFrames>> &indices) const {
    sel.stamps_s.assign(static_cast<std::size_t>(params_.num_views), {});
    for (auto &s : sel.stamps_s) s.fill(std::numeric_limits<double>::quiet_NaN());
    indices.assign(static_cast<std::size_t>(params_.num_views), {});

    const View &wide = views_[0];
    if (wide.frames.empty()) {
      sel.reason      = "no wide frame yet";
      sel.failed_view = 0;
      sel.failed_slot = kNumHistFrames - 1;
      return false;
    }
    const std::vector<double> wide_stamps = Stamps(wide);
    const double              oldest_s    = *std::min_element(wide_stamps.begin(), wide_stamps.end());
    const double              span_s      = wide.newest_s - oldest_s;
    const double              need_s      = (kNumHistFrames - 1) * params_.hist_dt_s;
    if (span_s < need_s - params_.history_tolerance_s) {
      char buf[120];
      std::snprintf(buf, sizeof(buf), "wide history spans %.2f s, need %.2f s", span_s, need_s);
      sel.reason      = buf;
      sel.failed_view = 0;
      sel.failed_slot = 0;
      return false;
    }
    const int                             now_index = Nearest(wide_stamps, wide.newest_s);
    const std::array<int, kNumHistFrames> picked    = PickHistory(wide_stamps, now_index, params_.hist_dt_s);
    for (int t = 0; t < kNumHistFrames; ++t) {
      const double target                          = wide.newest_s - (kNumHistFrames - 1 - t) * params_.hist_dt_s;
      const double stamp                           = wide_stamps[static_cast<std::size_t>(picked[static_cast<std::size_t>(t)])];
      sel.worst_history_offset_s                   = std::max(sel.worst_history_offset_s, std::abs(stamp - target));
      sel.stamps_s[0][static_cast<std::size_t>(t)] = stamp;
      indices[0][static_cast<std::size_t>(t)]      = picked[static_cast<std::size_t>(t)];
    }

    for (int v = 1; v < params_.num_views; ++v) {
      const std::vector<double> stamps = Stamps(views_[static_cast<std::size_t>(v)]);
      for (int t = 0; t < kNumHistFrames; ++t) {
        const double wide_stamp = sel.stamps_s[0][static_cast<std::size_t>(t)];
        const int    i          = NearestWithin(stamps, wide_stamp, params_.view_tolerance_s);
        if (i < 0 && params_.fill_missing_views) {
          sel.filled.emplace_back(v, t);
          indices[static_cast<std::size_t>(v)][static_cast<std::size_t>(t)] = -1;
          continue;
        }
        if (i < 0) {
          const int    nearest = Nearest(stamps, wide_stamp);
          const double off     = nearest < 0 ? std::numeric_limits<double>::infinity()
                                             : std::abs(stamps[static_cast<std::size_t>(nearest)] - wide_stamp);
          char         buf[160];
          if (nearest < 0) {
            std::snprintf(buf, sizeof(buf), "view %d slot %d: no frame at all", v, t);
          } else {
            std::snprintf(buf, sizeof(buf),
                          "view %d slot %d: no frame within %.3f s of the wide frame "
                          "(nearest is %.3f s away)",
                          v, t, params_.view_tolerance_s, off);
          }
          sel.reason      = buf;
          sel.failed_view = v;
          sel.failed_slot = t;
          return false;
        }
        const double off                                                       = std::abs(stamps[static_cast<std::size_t>(i)] - wide_stamp);
        sel.worst_view_offset_s                                                = std::max(sel.worst_view_offset_s, off);
        sel.stamps_s[static_cast<std::size_t>(v)][static_cast<std::size_t>(t)] = stamps[static_cast<std::size_t>(i)];
        indices[static_cast<std::size_t>(v)][static_cast<std::size_t>(t)]      = i;
      }
    }
    sel.ok = true;
    return true;
  }

  ViewWindowParams   params_;
  std::vector<View>  views_;
  std::vector<float> blank_chw_;
};

std::shared_ptr<ViewWindow> CreateViewWindow(const ViewWindowParams &params) {
  if (params.num_views < 1) {
    LOGE("ViewWindow needs at least one view, got %d", params.num_views);
    return nullptr;
  }
  if (params.hist_dt_s <= 0.0 || params.history_tolerance_s < 0.0 || params.view_tolerance_s < 0.0 ||
      params.retain_s < (kNumHistFrames - 1) * params.hist_dt_s) {
    LOGE("ViewWindow: invalid parameters (hist_dt %.3f, history_tol %.3f, view_tol %.3f, retain %.3f)",
         params.hist_dt_s, params.history_tolerance_s, params.view_tolerance_s, params.retain_s);
    return nullptr;
  }
  return std::shared_ptr<ViewWindow>(new ViewWindowImpl(params));
}

}  // namespace preprocess
