#ifndef __FRAME_WINDOW_HPP__
#define __FRAME_WINDOW_HPP__

#include <array>
#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <utility>
#include <vector>

#include "planner_params.hpp"

namespace preprocess {

// @brief `FrameWindow` class is a interface class for the sliding window of
// preprocessed camera frames. The model wants 4 frames 0.5 s apart while the
// camera delivers 10 Hz, so the window keeps the last 3*stride+1 preprocessed
// frames (16 at the default stride) and `Assemble` picks every stride-th one.
// Each frame is preprocessed exactly once, on `Push`. This is the count-based
// single-view window the hand-off package validates against (its clips define
// the history by frame index); a stream with timestamps and/or more than one
// camera goes through `ViewWindow` below
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

/****************************************/
/*******Timestamped, multi-view**********/
/****************************************/

// @brief: planner_io.pick_history. For the frame at `now_index` return the
// indices of the frames nearest in TIME to t-1.5, t-1.0, t-0.5 and t (oldest
// first) -- never by counting frames: four consecutive 10 Hz frames span
// 0.3 s, not 1.5 s, and a dropped frame shifts every later slot. Ties go to
// the lower index, as numpy's argmin does. `stamps_s` need not be sorted
std::array<int, model::planner::kNumHistFrames> PickHistory(const std::vector<double> &stamps_s,
                                                            int                        now_index,
                                                            double                     hist_dt_s = model::planner::kHistDtS);

// @brief: Index of the stamp nearest to `target_s`, or -1 when the nearest
// one is more than `tolerance_s` away (or the list is empty). This is how
// every non-wide view finds its frame for a chosen wide frame
int NearestWithin(const std::vector<double> &stamps_s, double target_s, double tolerance_s);

struct ViewWindowParams {
  int    num_views = 1;
  double hist_dt_s = model::planner::kHistDtS;
  // Warm-up gate: the wide history must span 3 * hist_dt_s, minus this
  double history_tolerance_s = model::planner::kDefaultHistoryToleranceS;
  // Per slot, how far a non-wide view's frame may be from the wide frame
  double view_tolerance_s = model::planner::kDefaultViewToleranceS;
  // What a slot without such a frame gets. false: the selection fails and the
  // caller skips the inference. true: the slot is filled with the image the
  // camfront3 training fed for a missing side view -- all zeros before
  // normalization, i.e. -mean/std per channel -- and the selection lists it
  // in `filled`. View 0 is never filled: without it there is no "now"
  bool fill_missing_views = false;
  // Frames older than the view's newest stamp minus this are dropped; the
  // history reaches 1.5 s back, the rest is slack for the tolerances
  double retain_s = 2.5;
};

// @brief: What `ViewWindow::Select` decided for the newest wide frame. When
// `ok` is false, `reason` says why and `failed_view` / `failed_slot` locate
// it: view 0 means the wide history does not reach back 1.5 s yet (warm-up),
// another view means that camera has no frame within tolerance for the slot
struct ViewSelection {
  bool        ok = false;
  std::string reason;
  int         failed_view = -1;
  int         failed_slot = -1;
  // Per view, per slot (oldest first): the chosen frame's stamp [s]. Only
  // views/slots resolved before a failure are filled
  std::vector<std::array<double, model::planner::kNumHistFrames>> stamps_s;
  // Largest |wide slot stamp - target time| and, over the other views,
  // largest |view stamp - wide stamp|, for diagnostics
  double worst_history_offset_s = 0.0;
  double worst_view_offset_s    = 0.0;
  // (view, slot) pairs filled with the blank frame (fill_missing_views only);
  // their stamps_s entries stay NaN
  std::vector<std::pair<int, int>> filled;
};

// @brief `ViewWindow` class is a interface class for the timestamped window
// of one or more camera views. View 0 (the wide front camera) anchors the
// history: its newest frame is "now" and its four slots are picked by nearest
// stamp (PickHistory). Every other view then supplies the frame nearest in
// time to each chosen wide frame, within `view_tolerance_s`; a slot with no
// such frame fails the whole selection (the caller skips that inference) or,
// with fill_missing_views, gets the training's blank frame; a duplicated or
// stale frame is never substituted. Frames are preprocessed once, on `Push`;
// `Assemble` only copies
class ViewWindow {
 public:
  ViewWindow()          = default;
  virtual ~ViewWindow() = default;

  // @brief: Preprocess one frame of `view` (0 = wide) stamped `stamp_s`
  // (seconds, any epoch shared by all views) and add it. A stamp that jumps
  // back by more than `retain_s` (a looping bag) empties that view first
  virtual void Push(int view, const cv::Mat &in_frame, double stamp_s, bool in_is_rgb = false) = 0;

  // @brief: The same for a frame already run through PreprocessFrame
  // (kFrameNumel floats). A multi-threaded caller preprocesses outside its
  // lock and only holds it for this push
  virtual void PushPreprocessed(int view, std::vector<float> &&in_chw, double stamp_s) = 0;

  // @brief: Choose the frames for the newest wide frame without copying
  virtual ViewSelection Select() const = 0;

  // @brief: Select, then write the [V, 3, 4, H, W] tensor (V contiguous
  // [3, 4, H, W] channel-major blocks, view 0 first) to `out_video`, which
  // must hold NumViews() * kVideoNumel floats. Returns the selection; on
  // failure `out_video` is left untouched
  virtual ViewSelection Assemble(float *out_video) const = 0;

  virtual int NumViews() const     = 0;
  virtual int Size(int view) const = 0;
  // @brief: Newest stamp pushed to `view`, NaN while it is empty
  virtual double NewestStamp(int view) const = 0;
  virtual void   Reset()                     = 0;
};

// @brief: This function is used as a factory to generate a ViewWindow object.
// Returns nullptr on an invalid parameter set
std::shared_ptr<ViewWindow> CreateViewWindow(const ViewWindowParams &params);

}  // namespace preprocess

#endif  //__FRAME_WINDOW_HPP__
