// Sliding window of preprocessed camera frames.
//
// The model wants 4 frames 0.5 s apart, but the camera delivers 10 Hz; the
// window therefore keeps the last 3*stride+1 preprocessed frames (16 at the
// default stride of 5) and assemble() picks every stride-th one. Each frame
// is preprocessed exactly once, on push -- the ROS 2 image callback maps to
// push(), the planning tick to assemble().
#pragma once

#include <deque>
#include <vector>

#include <opencv2/core.hpp>

#include "latentdrive/types.hpp"

namespace latentdrive {

class FrameWindow {
 public:
  explicit FrameWindow(int stride_frames = kFrameStride);

  // Preprocesses one camera frame (any resolution) and appends it:
  //   BGR -> RGB, center-crop height to width/2, bilinear resize to 512x256,
  //   /255, ImageNet-normalize, HWC -> CHW float32.
  // Pass `is_rgb = true` if the Mat is already RGB (cv::imread gives BGR).
  void push(const cv::Mat& frame, bool is_rgb = false);

  // True once 3*stride+1 frames have been pushed (1.5 s of history at 10 Hz).
  bool ready() const;

  // Writes the [3,4,256,512] video tensor (oldest..newest slots) to `dst`,
  // which must hold kVideoNumel floats. Precondition: ready().
  void assemble(float* dst) const;

  void reset();
  int size() const { return static_cast<int>(frames_.size()); }
  int capacity() const { return capacity_; }

  // One frame's preprocessing, exposed for testing against the training
  // pipeline: writes kFrameNumel floats (CHW) to `dst`.
  static void preprocess(const cv::Mat& frame, float* dst, bool is_rgb = false);

 private:
  int stride_;
  int capacity_;
  std::deque<std::vector<float>> frames_;  // each kFrameNumel, CHW
};

}  // namespace latentdrive
