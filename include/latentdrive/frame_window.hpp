#ifndef __FRAME_WINDOW_HPP__
#define __FRAME_WINDOW_HPP__

#include <memory>
#include <opencv2/core.hpp>

#include "planner_params.hpp"

namespace preprocess {

// @brief `FrameWindow` class is a interface class for the sliding window of
// preprocessed camera frames. The model wants 4 frames 0.5 s apart while the
// camera delivers 10 Hz, so the window keeps the last 3*stride+1 preprocessed
// frames (16 at the default stride) and `Assemble` picks every stride-th one.
// Each frame is preprocessed exactly once, on `Push` -- in the ROS 2 node the
// image callback maps to `Push`, the planning tick to `Assemble`
class FrameWindow {
 public:
  FrameWindow()          = default;
  virtual ~FrameWindow() = default;

  // @brief: Preprocess one camera frame (any resolution) and append it:
  // BGR -> RGB, center-crop the height to width/2, bilinear resize to the
  // model input, /255, ImageNet-normalize, HWC -> CHW float32.
  // Set `in_is_rgb` when the Mat is already RGB (cv::imread gives BGR)
  virtual void Push(const cv::Mat &in_frame, bool in_is_rgb = false) = 0;

  // @brief: True once 3*stride+1 frames were pushed (1.5 s of history at 10 Hz)
  virtual bool Ready() const = 0;

  // @brief: Write the [3, 4, H, W] video tensor (oldest..newest slots) to
  // `out_video`, which must hold kVideoNumel floats. Precondition: Ready()
  virtual void Assemble(float *out_video) const = 0;

  virtual void Reset()      = 0;
  virtual int  Size() const = 0;
};

// @brief: This function is used to run the single-frame preprocessing on its
// own, e.g. to test the pipeline against the training-side reference tensors.
// `out_chw` must hold kFrameNumel floats. Returns false on an invalid frame
bool PreprocessFrame(const cv::Mat &in_frame, float *out_chw, bool in_is_rgb = false);

// @brief: This function is used as a factory to generate a FrameWindow object
std::shared_ptr<FrameWindow> CreateFrameWindow(int stride_frames = model::planner::kFrameStride);

}  // namespace preprocess

#endif  //__FRAME_WINDOW_HPP__
