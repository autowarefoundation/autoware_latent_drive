#ifndef __PLANNER_PARAMS_HPP__
#define __PLANNER_PARAMS_HPP__

#include "trajectory.hpp"

namespace model {
namespace planner {

/****************************************/
/***********Network related**************/
/****************************************/
// The exports come in two flavours that share everything but the video rank:
//   single-view  video  float32 [B, 3, 4, 256, 512]
//   multi-view   video  float32 [B, V, 3, 4, 256, 512]   view 0 = wide front
//                                                         view 1 = telephoto
//   status float32 [B, 6]
//   plan   float32 [B, N, 3]            (named `traj` in the older exports)
// B is 1 (older exports) or a symbolic axis the engine build freezes to 1.
// The view count V and the waypoint count N are read from the engine at load,
// so one binary serves every checkpoint
constexpr static char kVideoTensorName[]  = "video";
constexpr static char kStatusTensorName[] = "status";
constexpr static char kTrajTensorName[]   = "traj";  // legacy output name

constexpr static int kInputHeight   = 256;
constexpr static int kInputWidth    = 512;
constexpr static int kNumHistFrames = 4;  // video slots, oldest..newest, 0.5 s apart
constexpr static int kStatusDim     = 6;

constexpr static int kFrameNumel = 3 * kInputHeight * kInputWidth;  // one CHW frame
constexpr static int kVideoNumel = kFrameNumel * kNumHistFrames;    // one view: [3, 4, H, W]

// A multi-view tensor is V contiguous single-view blocks, [V, 3, 4, H, W];
// Planner::VideoNumel() is V * kVideoNumel for the loaded engine

/****************************************/
/**********Preprocess related************/
/****************************************/
// The history slots are kHistDtS apart in time. Camera frames run at 10 Hz,
// so on a healthy stream consecutive slots are kFrameStride frames apart --
// the count-based FrameWindow relies on that; the timestamped ViewWindow
// picks each slot by nearest stamp instead (planner_io.pick_history)
constexpr static double kHistDtS     = 0.5;
constexpr static int    kFrameStride = 5;

// ViewWindow tolerances. The wide history must reach back the full 1.5 s
// before a plan runs; kDefaultHistoryToleranceS is the slack on that span
// (half a 10 Hz period, so the warm-up ends with the 16th frame as the
// count-based window's does). The slots themselves are picked by nearest
// stamp with no tolerance, as planner_io.pick_history does. Every other view
// must have a frame within kDefaultViewToleranceS of the chosen wide frame --
// the 60 ms the training-side alignment used; a slot without one skips the
// inference, it is never filled with a substitute frame
constexpr static double kDefaultHistoryToleranceS = 0.05;
constexpr static double kDefaultViewToleranceS    = 0.06;

constexpr static float kImageNetMean[3] = {0.485f, 0.456f, 0.406f};
constexpr static float kImageNetStd[3]  = {0.229f, 0.224f, 0.225f};

/****************************************/
/*********Engine build related***********/
/****************************************/
// Layers pinned to fp32 in the fp16 build (the validated mixed-precision
// recipe). Prefixes match the ONNX node scopes of the planner stage --
// 713 of 11409 nodes, negligible compute next to the ViT encoder. The
// open-loop exports scope the model as `/planner/`, the closed-loop and
// two-view ones (export_onnx.py in the training repo) as `/m/`; the two-view
// model has one image_fc per view (`image_fc`, `image_fc_1`), both pinned.
// The build log says how many layers matched -- 0 means a new export scopes
// its nodes differently and this list needs the new prefix
constexpr static const char *kFp32LayerPrefixes[] = {
    "/planner/transformer/",
    "/planner/trajectory_head/",
    "/planner/image_fc/",
    "/planner/status_encoding/",
    "/m/transformer/",
    "/m/trajectory_head/",
    "/m/image_fc/",
    "/m/image_fc_1/",
    "/m/status_encoding/",
};

}  // namespace planner
}  // namespace model

#endif  //__PLANNER_PARAMS_HPP__
