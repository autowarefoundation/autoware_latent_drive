#include "frame_window.hpp"

#include <cstring>
#include <deque>
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

bool PreprocessFrame(const cv::Mat &in_frame, float *out_chw, bool in_is_rgb) {
  if (in_frame.empty() || in_frame.channels() != 3) {
    LOGE("Preprocess expects a non-empty 3-channel frame");
    return false;
  }

  cv::Mat rgb;
  if (in_is_rgb) {
    rgb = in_frame;
  } else {
    cv::cvtColor(in_frame, rgb, cv::COLOR_BGR2RGB);
  }

  // Vehicle preprocessing: keep the full width, center-crop the height to
  // width/2, then resize to the model input. On the hand-off package the
  // JPEGs are already 512x256, so both steps are no-ops there
  const int target_h = rgb.cols / 2;
  if (rgb.rows > target_h) {
    const int y0 = (rgb.rows - target_h) / 2;
    rgb          = rgb(cv::Rect(0, y0, rgb.cols, target_h));
  }
  if (rgb.rows != kInputHeight || rgb.cols != kInputWidth) {
    cv::resize(rgb, rgb, cv::Size(kInputWidth, kInputHeight), 0, 0, cv::INTER_LINEAR);
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
    // Slots oldest..newest: frames_[0], [stride], [2*stride], [3*stride].
    // The video layout is [3, T, H, W] -- channel-major -- so each frame
    // contributes one H*W plane per channel at out[(c*T + t) * H*W]
    constexpr std::size_t kPlane = static_cast<std::size_t>(kInputHeight) * kInputWidth;
    for (int t = 0; t < kNumHistFrames; ++t) {
      const std::vector<float> &frame = frames_[static_cast<std::size_t>(t) * stride_];
      for (int c = 0; c < 3; ++c) {
        std::memcpy(out_video + (c * kNumHistFrames + t) * kPlane,
                    frame.data() + c * kPlane, kPlane * sizeof(float));
      }
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

}  // namespace preprocess
