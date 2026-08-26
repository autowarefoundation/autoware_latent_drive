#include "latentdrive/frame_window.hpp"

#include <cstring>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace latentdrive {

FrameWindow::FrameWindow(int stride_frames)
    : stride_(stride_frames), capacity_(3 * stride_frames + 1) {
  if (stride_frames < 1) throw std::invalid_argument("stride_frames must be >= 1");
}

void FrameWindow::preprocess(const cv::Mat& frame, float* dst, bool is_rgb) {
  if (frame.empty()) throw std::invalid_argument("empty frame");
  cv::Mat rgb;
  if (frame.channels() == 3) {
    if (is_rgb) {
      rgb = frame;
    } else {
      cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
    }
  } else {
    throw std::invalid_argument("expected a 3-channel frame");
  }

  // Vehicle preprocessing: keep full width, center-crop height to width/2,
  // then resize to the model input. The hand-off package's JPEGs are already
  // 512x256, so both steps are no-ops there.
  const int target_h = rgb.cols / 2;
  if (rgb.rows > target_h) {
    const int y0 = (rgb.rows - target_h) / 2;
    rgb = rgb(cv::Rect(0, y0, rgb.cols, target_h));
  }
  if (rgb.rows != kInputHeight || rgb.cols != kInputWidth) {
    cv::resize(rgb, rgb, cv::Size(kInputWidth, kInputHeight), 0, 0, cv::INTER_LINEAR);
  }

  cv::Mat f32;
  rgb.convertTo(f32, CV_32FC3, 1.0 / 255.0);
  // HWC -> CHW with per-channel normalization
  for (int c = 0; c < 3; ++c) {
    const float mean = kImageNetMean[c];
    const float inv_std = 1.f / kImageNetStd[c];
    float* plane = dst + c * kInputHeight * kInputWidth;
    for (int y = 0; y < kInputHeight; ++y) {
      const float* row = f32.ptr<float>(y);
      float* out = plane + y * kInputWidth;
      for (int x = 0; x < kInputWidth; ++x) {
        out[x] = (row[x * 3 + c] - mean) * inv_std;
      }
    }
  }
}

void FrameWindow::push(const cv::Mat& frame, bool is_rgb) {
  std::vector<float> buf(kFrameNumel);
  preprocess(frame, buf.data(), is_rgb);
  frames_.push_back(std::move(buf));
  if (static_cast<int>(frames_.size()) > capacity_) frames_.pop_front();
}

bool FrameWindow::ready() const {
  return static_cast<int>(frames_.size()) >= capacity_;
}

void FrameWindow::assemble(float* dst) const {
  if (!ready()) throw std::runtime_error("FrameWindow::assemble before ready()");
  // slots oldest..newest: frames_[0], [stride], [2*stride], [3*stride];
  // layout [3, 4, H, W] -- channel-major, so each frame contributes one
  // H*W plane per channel at dst[(c*4 + t) * H*W].
  constexpr std::size_t plane = static_cast<std::size_t>(kInputHeight) * kInputWidth;
  for (int t = 0; t < kNumHistFrames; ++t) {
    const std::vector<float>& f = frames_[static_cast<std::size_t>(t) * stride_];
    for (int c = 0; c < 3; ++c) {
      std::memcpy(dst + (c * kNumHistFrames + t) * plane, f.data() + c * plane,
                  plane * sizeof(float));
    }
  }
}

void FrameWindow::reset() { frames_.clear(); }

}  // namespace latentdrive
